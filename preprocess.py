"""
Preprocess Akkadian transliterations and English translations
following the Deep Past Initiative competition formatting guidelines.
"""
import csv
import json
import os
import re
import unicodedata
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "processed"


def clean_transliteration(text: str) -> str:
    """Clean Akkadian transliteration following competition guidelines."""
    if not text or not isinstance(text, str):
        return ""

    t = text

    # Ḫ/ḫ → H/h (test data uses H/h only)
    t = t.replace("Ḫ", "H").replace("ḫ", "h")

    # Remove half brackets ˹ ˺ (partially broken signs)
    t = t.replace("˹", "").replace("˺", "")

    # Use placeholders to protect gap tokens from bracket removal
    GAP_PH = "___GAP___"
    BIGGAP_PH = "___BIGGAP___"

    # [x] → gap, [… …] → big_gap
    t = re.sub(r"\[x\]", GAP_PH, t)
    t = re.sub(r"\[\s*…\s*…?\s*\]", BIGGAP_PH, t)
    t = re.sub(r"\[\s*\.\.\.\s*\.?\.?\.?\s*\]", BIGGAP_PH, t)

    # Remove remaining square brackets but keep content
    t = re.sub(r"\[([^\]]*)\]", r"\1", t)

    # Remove << >> (errant signs, remove content too)
    t = re.sub(r"<<[^>]*>>", "", t)
    # Remove < > (scribal insertions, keep inner text)
    t = re.sub(r"<([^>]*)>", r"\1", t)

    # Standalone ellipsis → big_gap
    t = re.sub(r"…", f" {BIGGAP_PH} ", t)

    # Restore gap tokens
    t = t.replace(GAP_PH, " <gap> ")
    t = t.replace(BIGGAP_PH, " <big_gap> ")

    # Remove scribal notations: ! ? /
    t = re.sub(r"!", "", t)
    t = re.sub(r"\?", "", t)
    t = re.sub(r"/", " ", t)

    # Remove word dividers : and . that act as word dividers
    # But keep . inside sign names like KÙ.BABBAR, ḪI.A, etc.
    # Only remove standalone : (word divider)
    t = re.sub(r"\s*:\s*", " ", t)

    # Normalize subscript numbers to regular: ₀-₉ → 0-9
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")
    t = t.translate(subscript_map)

    # Normalize superscript numbers if any
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    t = t.translate(superscript_map)

    # Normalize multiple <gap> or <big_gap> in sequence
    t = re.sub(r"(<big_gap>\s*)+", "<big_gap> ", t)
    t = re.sub(r"(<gap>\s*)+", "<gap> ", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


def clean_translation(text: str) -> str:
    """Clean English translation text."""
    if not text or not isinstance(text, str):
        return ""

    t = text

    # Remove leading/trailing quotes (artifact from CSV)
    t = t.strip().strip('"').strip("'")

    # Remove doubled quotes
    t = t.replace('""', '"')

    # Clean up extra quotes at end
    t = re.sub(r'"\s*$', "", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


def load_train_data():
    """Load and preprocess train.csv."""
    rows = []
    with open(DATA_DIR / "train.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = clean_transliteration(row["transliteration"])
            tgt = clean_translation(row["translation"])
            if src and tgt:
                rows.append({"source": src, "target": tgt, "id": row["oare_id"]})
    return rows


def load_test_data():
    """Load and preprocess test.csv."""
    rows = []
    with open(DATA_DIR / "test.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = clean_transliteration(row["transliteration"])
            rows.append({"source": src, "id": row["id"]})
    return rows


def load_sentences_oare():
    """Load Sentences_Oare data for augmentation (translation-only sentences)."""
    rows = []
    with open(
        DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv", "r", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            translation = clean_translation(row.get("translation", ""))
            if translation and len(translation) > 10:
                rows.append(
                    {
                        "text_uuid": row.get("text_uuid", ""),
                        "translation": translation,
                        "first_word_spelling": row.get("first_word_spelling", ""),
                    }
                )
    return rows


def build_augmented_data(train_data, sentences_oare):
    """
    Try to match Sentences_Oare entries with published_texts
    to create additional parallel training pairs.
    """
    # Load published_texts to get transliterations by text_uuid
    text_transliterations = {}
    with open(DATA_DIR / "published_texts.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oare_id = row.get("oare_id", "")
            translit = row.get("transliteration", "")
            if oare_id and translit:
                text_transliterations[oare_id] = translit

    # Get train IDs to avoid duplication
    train_ids = {r["id"] for r in train_data}

    # Sentences_Oare doesn't have direct transliterations,
    # but we can use it as additional translation-only data for context
    # For now, just return the primary training data
    # The sentences can be used later for back-translation or other augmentation
    print(f"  Published texts with transliterations: {len(text_transliterations)}")
    print(f"  Sentences_Oare entries: {len(sentences_oare)}")

    return train_data


def split_data(data, val_ratio=0.1, seed=42):
    """Split data into train and validation sets."""
    import random

    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    val_size = max(1, int(len(data) * val_ratio))
    val_indices = set(indices[:val_size])

    train_split = [data[i] for i in range(len(data)) if i not in val_indices]
    val_split = [data[i] for i in range(len(data)) if i in val_indices]

    return train_split, val_split


def save_jsonl(data, filepath):
    """Save data as JSONL."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing train data...")
    train_data = load_train_data()
    print(f"  Loaded {len(train_data)} training examples")

    print("Loading Sentences_Oare for augmentation...")
    sentences_oare = load_sentences_oare()

    print("Building augmented dataset...")
    augmented_data = build_augmented_data(train_data, sentences_oare)
    print(f"  Total training examples: {len(augmented_data)}")

    print("Splitting into train/val...")
    train_split, val_split = split_data(augmented_data)
    print(f"  Train: {len(train_split)}, Val: {len(val_split)}")

    print("Loading and preprocessing test data...")
    test_data = load_test_data()
    print(f"  Loaded {len(test_data)} test examples")

    print("Saving processed data...")
    save_jsonl(train_split, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_split, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    # Print sample
    print("\n--- Sample preprocessed training example ---")
    sample = train_data[0]
    print(f"Source: {sample['source'][:200]}")
    print(f"Target: {sample['target'][:200]}")

    print("\n--- Sample test example ---")
    sample = test_data[0]
    print(f"Source: {sample['source'][:200]}")

    print("\nDone! Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
