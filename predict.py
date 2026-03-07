"""
Generate predictions on test set and create submission.csv
"""
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "best_model"
PROCESSED_DIR = Path(__file__).parent / "processed"
OUTPUT_FILE = Path(__file__).parent / "submission.csv"
PREFIX = "translate Akkadian to English: "


def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load test data
    test_data = load_jsonl(PROCESSED_DIR / "test.jsonl")
    print(f"Test examples: {len(test_data)}")

    # Generate translations
    results = []
    for item in test_data:
        input_text = PREFIX + item["source"]
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"id": item["id"], "translation": translation})
        print(f"\n--- Test ID {item['id']} ---")
        print(f"Source: {item['source'][:150]}...")
        print(f"Translation: {translation[:200]}")

    # Write submission
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "translation"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSubmission saved to: {OUTPUT_FILE}")

    # Also run on a few validation examples for quality check
    print("\n=== Validation Sample Check ===")
    val_data = load_jsonl(PROCESSED_DIR / "val.jsonl")
    for item in val_data[:3]:
        input_text = PREFIX + item["source"]
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nSource: {item['source'][:120]}...")
        print(f"Predicted: {translation[:200]}")
        print(f"Reference: {item['target'][:200]}")


if __name__ == "__main__":
    main()
