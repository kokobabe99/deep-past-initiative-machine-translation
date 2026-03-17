# Deep Past Initiative — Akkadian-to-English Machine Translation

## Overview

This project tackles the [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) Kaggle competition. The task is to translate **Old Assyrian (Akkadian) transliterations** into modern English. These texts originate from cuneiform tablets dating back ~4,000 years, documenting trade records, legal contracts, and personal correspondence from ancient Mesopotamia.

**Competition Metric:** Geometric mean of BLEU and chrF++ scores.

**Kaggle Profile:** [Jackieveloped](https://www.kaggle.com/jackieveloped)

## Results

| Submission | Public Score | Approach |
|------------|-------------|----------|
| Dual ByT5 Ensemble + MBR v1 | **34.9** | Dual ByT5 ensemble with MBR consensus decoding |
| Flan-T5-base Baseline | 19.3 | Single fine-tuned Flan-T5-base |

> Best entry scored **34.9** using a dual-model ByT5 ensemble with Minimum Bayes Risk (MBR) decoding, domain-specific pre/post-processing, and proper noun lexicon-guided candidate selection.

![Kaggle Submission Score](doc/image.png)

## Dataset

| Split | Examples |
|-------|----------|
| Train | 1,561 parallel pairs (transliteration → English) |
| Val   | 156 (10% held out) |
| Test  | 4 unseen transliterations |

### Auxiliary Data

The competition provides several auxiliary datasets. The **OA_Lexicon_eBL.csv** is actively used in the current pipeline for proper noun fidelity scoring during MBR selection.

| File | Records | Content | Usage |
|------|---------|---------|-------|
| `OA_Lexicon_eBL.csv` | 39,332 | Akkadian word forms → normalized lemmas | **Active** — 13k+ proper nouns (PN, GN, DN, RN) extracted for MBR fidelity scoring |
| `Sentences_Oare_FirstWord_LinNum.csv` | 9,783 | Sentence-level English translations with metadata | Potential data augmentation if aligned with `published_texts.csv` |
| `published_texts.csv` | 7,992 | Full transliterations of cuneiform tablets | Potential monolingual pretraining / domain adaptation |
| `eBL_Dictionary.csv` | 19,216 | Akkadian words → English definitions | Potential lexical augmentation |
| `publications.csv` | 216,603 | OCR text from academic papers on Assyriology | Potential background knowledge |
| `bibliography.csv` | 908 | Publication metadata | Not used |
| `resources.csv` | 292 | Research paper references and links | Not used |

## Methodology

The project evolved through two major iterations:

### Iteration 1: Fine-tuned Flan-T5-base (Score: 19.3)

Single-model approach using `google/flan-t5-base` (248M parameters) fine-tuned on the 1,561 training pairs. Simple beam search decoding. See [notebook.ipynb](notebook.ipynb) for the training pipeline.

### Iteration 2: Dual ByT5 Ensemble + MBR (Score: 34.9)

The current best submission uses a dual-model ensemble with MBR consensus selection. The pipeline has five stages:

#### 1. Preprocessing

Akkadian transliterations require extensive normalization before model input. The preprocessing pipeline handles:

| Transformation | Example | Purpose |
|---|---|---|
| ASCII → diacritics | `a2` → `á`, `sz` → `š` | Standardize transliteration notation |
| Determinative handling | `(D)` → `D`, `(d)` → `{d}` | Separate semantic classifiers from phonetic content |
| Gap unification | `[x]`, `...`, `(break)` → `<gap>` | Normalize 10+ gap variants into a single token |
| Character cleanup | `ḫ` → `h`, subscript digits → ASCII | Reduce vocabulary complexity |
| Fraction normalization | `0.3333` → `⅓`, `0.5` → `½` | Standardize numerical representations |
| Currency normalization | `KÙ.B.` → `KÙ.BABBAR` | Fix truncated commodity names |

#### 2. Dual-Model Inference

Two pre-trained ByT5 models (~582M parameters each) generate translation candidates independently:

- **Model A** (`byt5-akkadian-optimized-34x`) — optimized for general translation quality
- **Model B** (`byt5-akkadian-mbr`) — trained with MBR-oriented objectives

Each model generates **4 beam search candidates + 2 stochastic sampling candidates** per input, yielding a pool of 12 diverse candidates per sample.

| Parameter | Value | Purpose |
|---|---|---|
| `num_beams` | 8 | Wide beam for diverse hypotheses |
| `num_beam_cands` | 4 | Top-4 beam candidates retained |
| `num_sample_cands` | 2 | Stochastic diversity via nucleus sampling |
| `length_penalty` | 1.3 | Slight preference for longer, more complete translations |
| `repetition_penalty` | 1.2 | Discourage repetitive output |
| `sample_top_p` | 0.92 | Nucleus sampling threshold |
| `sample_temperature` | 0.75 | Moderate randomness for diversity |

Engineering optimizations:
- **BF16 autocast** — reduces VRAM usage by ~40% on supported GPUs
- **Bucket batching** — groups similar-length inputs to minimize padding waste
- **OOM protection** — gracefully skips batches on GPU memory overflow instead of crashing
- **Sequential model loading** — loads one model at a time to fit within 16GB VRAM

#### 3. Postprocessing

Model outputs are cleaned before MBR selection:

- Gap and `PN` token unification
- Grammar marker removal (`fem.`, `pl.`, `sing.`)
- Shekel fraction normalization (e.g., `5/12 shekels` → `⅔ shekel 15 grains`)
- Commodity name expansion (e.g., `-gold` → `pašallum gold`)
- Month numeral conversion (Roman → Arabic)
- Repeated word/phrase deduplication
- Forbidden character stripping (brackets, angle brackets, etc.)

#### 4. MBR Consensus Selection

Instead of simply taking the top-1 beam search result, MBR selects the candidate with the highest average agreement against all other candidates in the pool.

The scoring function combines:
- **chrF++ consensus (80%)** — measures character + word n-gram overlap between candidate pairs, selecting the translation most consistent with the overall pool
- **Proper noun fidelity (20%)** — checks whether expected proper nouns (from a 13k-entry lexicon) appear in the candidate translation
- **Agreement bonus** — small boost for candidates that appear identically from multiple models

The chrF++ implementation follows the sacrebleu algorithm: per-order F-score computation with character 6-gram + word 2-gram, weighted average, beta=2.0.

#### 5. Checkpointing and Validation

- Periodic CSV checkpoints every 200 rows
- Final validation reports empty translation count, length statistics, and sample predictions

## Project Structure

```
deep-past-initiative-machine-translation/
├── data/                        # Raw competition data
│   ├── train.csv                # 1,561 parallel transliteration-translation pairs
│   ├── test.csv                 # Test transliterations
│   ├── sample_submission.csv    # Expected submission format
│   ├── OA_Lexicon_eBL.csv       # Akkadian lexicon (used for proper noun scoring)
│   ├── Sentences_Oare_FirstWord_LinNum.csv
│   ├── eBL_Dictionary.csv
│   ├── published_texts.csv
│   ├── publications.csv
│   ├── bibliography.csv
│   └── resources.csv
├── doc/
│   ├── deep-past.docx           # Project documentation
│   └── image.png                # Kaggle submission score screenshot
├── notebook.ipynb               # Iteration 1: Flan-T5 training pipeline
├── submission_notebook.ipynb    # Iteration 2: Dual ByT5 ensemble + MBR inference
└── README.MD
```

## How to Run

### Kaggle Submission (Current Best — Dual ByT5 Ensemble)

1. Create a new Kaggle notebook and upload `submission_notebook.ipynb`.
2. Attach the following inputs:
   - **Competition data** — Deep Past Initiative Machine Translation
   - **Dataset** — `assiaben/final-byt5` (Model A)
   - **Model** — `mattiaangeli/byt5-akkadian-mbr` (Model B)
3. Enable GPU (P100 or T4), disable internet, and run all cells.
4. Submit the generated `submission.csv`.

### Flan-T5 Training (Iteration 1)

1. Upload `notebook.ipynb` to Kaggle, enable GPU (T4), and run all cells.
2. The notebook trains the model and saves weights to `/kaggle/working/best_model`.

## Dependencies

All dependencies are pre-installed in the Kaggle environment — **no internet required** at submission time.

- `torch` — PyTorch backend
- `transformers` — model loading, tokenization, generation
- `pandas` / `numpy` — data manipulation
- `tqdm` — progress bars

The chrF++ metric for MBR scoring is implemented from scratch (matching the sacrebleu algorithm) to avoid external dependency installation in the offline Kaggle environment.

## Challenges & Lessons Learned

- **Low-resource translation:** With only 1,561 training pairs, transfer learning from pretrained models is essential. The jump from Flan-T5-base (19.3) to pre-trained ByT5 ensemble (34.9) demonstrates the value of domain-specific pretraining.
- **Byte-level tokenization matters:** ByT5 operates at the byte level, which handles Akkadian's special characters (diacritics, subscripts) natively without tokenizer vocabulary issues that affect SentencePiece-based models like Flan-T5.
- **Ensemble diversity:** Using two independently trained models with different objectives produces candidates that fail in different ways, allowing MBR consensus to select more robust translations.
- **MBR > top-1 beam:** Simply taking the highest-probability beam search output misses cases where the model is confidently wrong. MBR consensus corrects this by favoring translations that multiple hypotheses agree on.
- **Proper noun preservation:** Akkadian texts contain many proper nouns (personal names, place names, deity names) that should be transliterated rather than translated. The lexicon-guided fidelity score helps MBR prefer candidates that preserve these correctly.
- **Offline constraints:** The Kaggle submission environment has no internet access, requiring all dependencies to be either pre-installed or implemented from scratch (e.g., chrF++ scoring).
