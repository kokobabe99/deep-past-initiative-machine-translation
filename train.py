"""
Fine-tune google/flan-t5-base for Akkadian → English translation.
"""
import json
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-small")
PROCESSED_DIR = Path(__file__).parent / "processed"
OUTPUT_DIR = Path(__file__).parent / "model_output"
PREFIX = "translate Akkadian to English: "
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    train_data = load_jsonl(PROCESSED_DIR / "train.jsonl")
    val_data = load_jsonl(PROCESSED_DIR / "val.jsonl")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create HF datasets
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Tokenize
    def preprocess(examples):
        inputs = [PREFIX + s for s in examples["source"]]
        targets = examples["target"]

        model_inputs = tokenizer(
            inputs, max_length=MAX_SOURCE_LEN, truncation=True, padding=False
        )
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing data...")
    train_tokenized = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    # Metric
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        chrf = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            word_order=2,  # chrF++
        )

        bleu_score = bleu["score"]
        chrf_score = chrf["score"]
        geo_mean = (bleu_score * chrf_score) ** 0.5 if bleu_score > 0 and chrf_score > 0 else 0.0

        return {
            "bleu": bleu_score,
            "chrf++": chrf_score,
            "geo_mean": geo_mean,
        }

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Training arguments
    # Detect device capabilities
    use_fp16 = torch.cuda.is_available()
    use_mps = (not torch.cuda.is_available()) and torch.backends.mps.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=2,
        fp16=use_fp16,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=0,
        use_mps_device=use_mps,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving best model...")
    best_dir = Path(__file__).parent / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    print(f"Best model saved to: {best_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
