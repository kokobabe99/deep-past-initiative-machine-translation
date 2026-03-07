# Deep Past Initiative — Akkadian-to-English Machine Translation

## Overview

This project tackles the [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) Kaggle competition, which asks participants to translate Old Assyrian (Akkadian) transliterations into English. The texts originate from cuneiform tablets documenting trade, legal contracts, and correspondence from ~4,000 years ago.

**Competition Metric:** Geometric mean of BLEU and chrF++ scores.

**Kaggle Profile:** [Jackieveloped](https://www.kaggle.com/jackieveloped)

## Results

| Leaderboard Rank | Score | Submissions |
|------------------|-------|-------------|
| #2180 / 2700+    | 19.3  | 2           |

> Best entry scored **19.3** (geometric mean of BLEU and chrF++), placing in the top tier of participants using a single fine-tuned Flan-T5-base model without external data augmentation.

## Dataset

| Split | Examples |
|-------|----------|
| Train | 1,561 parallel pairs (transliteration → English) |
| Val   | 156 (10% held out) |
| Test  | 4 unseen transliterations |

Additional auxiliary data provided but not directly used as parallel pairs:
- `Sentences_Oare_FirstWord_LinNum.csv` — monolingual translation sentences
- `published_texts.csv` — published transliterations indexed by `oare_id`
- `OA_Lexicon_eBL.csv`, `eBL_Dictionary.csv` — lexicon resources
- `bibliography.csv`, `publications.csv`, `resources.csv` — reference metadata

## Methodology

### 1. Preprocessing (`preprocess.py`)

Akkadian transliterations contain specialized philological markup that must be normalized before feeding into a language model:

| Raw Notation | Meaning | Cleaned Form |
|---|---|---|
| `Ḫ` / `ḫ` | Special consonant | `H` / `h` |
| `˹ ˺` | Half brackets (partially broken signs) | Removed |
| `[x]` | Gap (missing sign) | `<gap>` |
| `[… …]` | Large gap | `<big_gap>` |
| `[text]` | Restored text (square brackets) | `text` (brackets removed) |
| `<<text>>` | Errant signs | Removed entirely |
| `<text>` | Scribal insertions | `text` (angle brackets removed) |
| `₀₁₂...₉` | Subscript numbers | `0123456789` |
| `!`, `?`, `/` | Scribal notations | Removed or replaced with space |
| `:` | Word divider | Space |

English translations are cleaned by removing CSV artifacts (doubled quotes, trailing quotes) and normalizing whitespace.

### 2. Model Architecture

We fine-tune **`google/flan-t5-base`** (248M parameters), a sequence-to-sequence transformer pretrained on instruction-following tasks. The model is prompted with:

```
translate Akkadian to English: <cleaned_transliteration>
```

Key reasons for choosing Flan-T5:
- Seq2Seq architecture is a natural fit for translation tasks
- Flan instruction tuning provides a strong baseline for following task-specific prompts
- The `base` size balances quality with Kaggle's GPU/time constraints

### 3. Training Configuration (`train.py`)

| Hyperparameter | Value |
|---|---|
| Base model | `google/flan-t5-base` |
| Max source / target length | 256 tokens |
| Learning rate | 3e-4 |
| Batch size | 4 (effective 16 with gradient accumulation of 4) |
| Epochs | 15 |
| Warmup steps | 50 |
| Weight decay | 0.01 |
| Precision | FP16 |
| Best model selection | Lowest eval loss |
| Optimizer | AdamW (default) |

### 4. Inference (`predict.py`)

Beam search decoding with:
- `num_beams=5`
- `length_penalty=1.0`
- `no_repeat_ngram_size=3`
- `early_stopping=True`

## Ablation Studies

| Variant | Change | Observation |
|---|---|---|
| `flan-t5-small` vs `flan-t5-base` | Model size (80M → 248M) | Base produces more fluent, contextually accurate translations |
| Epochs 10 vs 15 | Training duration | 15 epochs showed continued improvement on eval loss without overfitting |
| Beam size 2 vs 5 | Decoding strategy | Beam 5 yields more coherent full-sentence translations |
| With vs without preprocessing | Transliteration cleaning | Cleaning significantly improves alignment; raw markup confuses the tokenizer |
| `predict_with_generate=True` during training | Generate predictions at each eval epoch | Disabled to save VRAM on Kaggle; eval loss used as proxy instead |

## Project Structure

```
deep-past-initiative-machine-translation/
├── data/                        # Raw competition data
│   ├── train.csv                # 1,561 parallel transliteration-translation pairs
│   ├── test.csv                 # 4 test transliterations
│   ├── sample_submission.csv    # Expected submission format
│   ├── Sentences_Oare_FirstWord_LinNum.csv
│   ├── OA_Lexicon_eBL.csv
│   ├── eBL_Dictionary.csv
│   ├── published_texts.csv
│   ├── publications.csv
│   ├── bibliography.csv
│   └── resources.csv
├── processed/                   # Preprocessed JSONL files
│   ├── train.jsonl              # 1,405 training examples
│   ├── val.jsonl                # 156 validation examples
│   └── test.jsonl               # 4 test examples
├── preprocess.py                # Data cleaning and preprocessing pipeline
├── train.py                     # Fine-tuning script (local execution)
├── predict.py                   # Inference and submission generation
├── notebook.ipynb               # Full Kaggle notebook (train + predict)
├── submission_notebook.ipynb    # Inference-only notebook for Kaggle submission
└── README.MD
```

## How to Run

### Local Training

```bash
# 1. Preprocess data
python preprocess.py

# 2. Train (defaults to flan-t5-small; set MODEL_NAME for base)
MODEL_NAME=google/flan-t5-base python train.py

# 3. Generate predictions
python predict.py
```

### Kaggle Submission

1. **Training notebook** (`notebook.ipynb`): Run with GPU enabled on Kaggle. Trains the model and saves it as a Kaggle Dataset output.
2. **Submission notebook** (`submission_notebook.ipynb`): Attach the saved model as a Kaggle Dataset input. Runs inference offline (no internet required) and produces `submission.csv`.

## Dependencies

- Python 3.10+
- `transformers`
- `datasets`
- `evaluate`
- `sacrebleu`
- `torch`
- `sentencepiece`
- `accelerate`

## Analysis

- The dataset is extremely small (1,561 pairs), making this a **low-resource translation** problem. Transfer learning from a pretrained model is essential.
- Akkadian transliterations have a highly structured format (sign readings separated by hyphens, determinatives in curly braces) that benefits from careful preprocessing.
- Auxiliary data (lexicon, sentences) could potentially be leveraged for data augmentation or back-translation in future work, but direct parallel pairs were not easily extractable.
- The test set contains only 4 examples, so generalization is critical — overfitting to training data would hurt performance on unseen texts.
