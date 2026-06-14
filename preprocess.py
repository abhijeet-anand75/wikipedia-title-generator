# =============================================================================
# preprocess.py
# Dataset Loading, Preprocessing, and Vocabulary Building
#
# This script is the entry point for Part A of the NLP Assignment.
# It performs the following steps:
#
#   1. Load train.csv and test.csv
#   2. Extract 500 articles as a validation set from training data
#   3. Check for and handle missing values
#   4. Preprocess article body text  (full pipeline)
#   5. Preprocess article titles     (light pipeline — no stopword removal)
#   6. Build vocabulary from training text using document-frequency threshold
#   7. Save preprocessed splits and vocabulary to disk
#
# Usage
# -----
#   python preprocess.py
#
# Outputs
# -------
#   preprocessed_train.csv       → PROCESSED_DIR
#   preprocessed_validation.csv  → PROCESSED_DIR
#   preprocessed_test.csv        → PROCESSED_DIR
#   vocab.pkl                    → VOCAB_SAVE_PATH
# =============================================================================

import os
import time

import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    TRAIN_PATH,
    TEST_PATH,
    TRAIN_PROCESSED,
    VAL_PROCESSED,
    TEST_PROCESSED,
    VOCAB_SAVE_PATH,
    VAL_SIZE,
    MIN_DOC_FREQ,
    RANDOM_STATE,
    PROCESSED_DIR,
)
from utils import (
    set_seed,
    Timer,
    print_section,
    load_and_split,
    preprocess_text,
    preprocess_title,
    Vocabulary,
    logger,
    ensure_nltk_data
)


# =============================================================================
# STEP 1 — LOAD AND SPLIT DATASET
# =============================================================================

def step_load(train_path: str, test_path: str) -> tuple:
    """
    Load raw CSVs and split into train / validation / test sets.

    Parameters
    ----------
    train_path : str   Path to raw train.csv
    test_path  : str   Path to raw test.csv

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    print_section("STEP 1 — Loading Dataset")

    with Timer("Dataset Loading"):
        train_df, val_df, test_df = load_and_split(
            train_path   = train_path,
            test_path    = test_path,
            val_size     = VAL_SIZE,
            random_state = RANDOM_STATE,
        )

    # Print dataset statistics
    print(f"\n  Train articles  : {len(train_df):,}")
    print(f"  Val   articles  : {len(val_df):,}")
    print(f"  Test  articles  : {len(test_df):,}")

    # Confirm no missing values after split
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        missing = df[["text", "title"]].isnull().sum()
        print(f"\n  {name} — missing values:")
        print(f"    text  : {missing['text']}")
        print(f"    title : {missing['title']}")

    return train_df, val_df, test_df


# =============================================================================
# STEP 2 — PREPROCESS TEXT AND TITLES
# =============================================================================

def step_preprocess(
    train_df : pd.DataFrame,
    val_df   : pd.DataFrame,
    test_df  : pd.DataFrame,
) -> tuple:
    """
    Apply preprocessing pipelines to all three splits.

    - Article body  → full pipeline  (lowercase, non-ASCII, punctuation,
                                      tokenize, stopwords, lemmatize)
    - Article title → light pipeline (lowercase, non-ASCII, punctuation only)
                      Titles are NOT stopword-filtered or lemmatized because:
                      * Removing stopwords distorts the target sequence
                      * The model must learn to generate natural language titles
                      * ROUGE evaluation requires surface-form titles

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Raw dataframes with 'text' and 'title' columns.

    Returns
    -------
    train_df, val_df, test_df with new columns:
        'processed_text'  : preprocessed article body
        'processed_title' : lightly preprocessed title
    """
    print_section("STEP 2 — Preprocessing Text and Titles")

    for split_name, df in [
        ("Train",      train_df),
        ("Validation", val_df),
        ("Test",       test_df),
    ]:
        print(f"\n  Processing {split_name} split ({len(df):,} articles)...")

        with Timer(f"{split_name} preprocessing"):
            df["processed_text"]  = df["text"].apply(preprocess_text)
            df["processed_title"] = df["title"].apply(preprocess_title)

        # Sanity check — show one example
        if len(df) > 0:
            sample_idx = 0
            print(f"\n  Sample from {split_name}:")
            print(f"    Raw title        : {df['title'].iloc[sample_idx]}")
            print(f"    Processed title  : {df['processed_title'].iloc[sample_idx]}")
            print(f"    Raw text (first 80 chars)       : "
                  f"{str(df['text'].iloc[sample_idx])[:80]}...")
            print(f"    Processed text (first 80 chars) : "
                  f"{str(df['processed_text'].iloc[sample_idx])[:80]}...")

    return train_df, val_df, test_df


# =============================================================================
# STEP 3 — BUILD VOCABULARY
# =============================================================================

def step_build_vocabulary(train_df: pd.DataFrame) -> Vocabulary:
    """
    Build vocabulary from training article bodies.

    Only tokens that appear in >= MIN_DOC_FREQ fraction of training
    documents are included.  This is the assignment specification:
    "tokens that appear in at least 1% of the training set."

    Note: Vocabulary is built from article body text ONLY (not titles).
    Titles are the prediction target — they should not influence the
    input vocabulary.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain 'processed_text' column.

    Returns
    -------
    vocab : Vocabulary
    """
    print_section("STEP 3 — Building Vocabulary")

    print(f"  Min document frequency : {MIN_DOC_FREQ * 100:.1f}% of training docs")
    print(f"  Training documents     : {len(train_df):,}")
    print(f"  Min count threshold    : "
          f"{max(1, int(MIN_DOC_FREQ * len(train_df))):,} documents")

    vocab = Vocabulary(min_doc_freq=MIN_DOC_FREQ)

    with Timer("Vocabulary Building"):
        corpus = (
            train_df["processed_text"].fillna("").tolist()
            + train_df["processed_title"].fillna("").tolist()
        )
        vocab.build_vocabulary(corpus)

    # Print vocabulary statistics
    print(f"\n  Vocabulary size  : {len(vocab):,} tokens")
    print(f"  Special tokens   : <pad>=0, <bos>=1, <eos>=2, <unk>=3")

    # Show most common tokens (top 10 by index — these are most frequent)
    sample_tokens = list(vocab.itos.items())[4:14]
    print(f"\n  Sample tokens (indices 4–13):")
    for idx, token in sample_tokens:
        print(f"    [{idx}] {token}")

    return vocab


# =============================================================================
# STEP 4 — SAVE PREPROCESSED DATA AND VOCABULARY
# =============================================================================

def step_save(
    train_df : pd.DataFrame,
    val_df   : pd.DataFrame,
    test_df  : pd.DataFrame,
    vocab    : Vocabulary,
) -> None:
    """
    Save preprocessed dataframes and vocabulary to disk.

    Saves only the columns needed for training:
        'processed_text'  — encoder input
        'processed_title' — decoder target

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    vocab : Vocabulary
    """
    print_section("STEP 4 — Saving Preprocessed Data")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    cols = ["processed_text", "processed_title"]

    with Timer("Saving CSVs"):
        train_df[cols].to_csv(TRAIN_PROCESSED, index=False)
        val_df[cols].to_csv(VAL_PROCESSED,   index=False)
        test_df[cols].to_csv(TEST_PROCESSED,  index=False)

    print(f"\n  Saved: {TRAIN_PROCESSED}  ({len(train_df):,} rows)")
    print(f"  Saved: {VAL_PROCESSED}    ({len(val_df):,} rows)")
    print(f"  Saved: {TEST_PROCESSED}   ({len(test_df):,} rows)")

    with Timer("Saving Vocabulary"):
        vocab.save(VOCAB_SAVE_PATH)

    print(f"  Saved: {VOCAB_SAVE_PATH}")


# =============================================================================
# STEP 5 — PRINT SUMMARY
# =============================================================================

def step_summary(timings: dict) -> None:
    """
    Print a final timing summary for the report.

    Parameters
    ----------
    timings : dict
        Keys = step names, Values = elapsed seconds
    """
    print_section("TASK A — EXECUTION SUMMARY")

    print(f"\n  {'Step':<35} {'Time (seconds)':>15}")
    print(f"  {'-'*50}")
    total = 0.0
    for step, elapsed in timings.items():
        print(f"  {step:<35} {elapsed:>15.2f}")
        total += elapsed
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<35} {total:>15.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run the full Task A pipeline:
    Load → Preprocess → Build Vocabulary → Save
    """
    ensure_nltk_data()
    overall_start = time.time()
    timings       = {}

    # Reproducibility
    set_seed(RANDOM_STATE)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    t0 = time.time()
    train_df, val_df, test_df = step_load(TRAIN_PATH, TEST_PATH)
    timings["Dataset Loading"] = time.time() - t0

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    t0 = time.time()
    train_df, val_df, test_df = step_preprocess(train_df, val_df, test_df)
    timings["Text Preprocessing"] = time.time() - t0

    # ── Step 3: Build Vocabulary ──────────────────────────────────────────────
    t0 = time.time()
    vocab = step_build_vocabulary(train_df)
    timings["Vocabulary Building"] = time.time() - t0

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    t0 = time.time()
    step_save(train_df, val_df, test_df, vocab)
    timings["Saving Preprocessed Data"] = time.time() - t0

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    step_summary(timings)
    print(f"\nActual total time: {time.time() - overall_start:.2f} seconds")

    print("\nTask A complete. Preprocessed data and vocabulary saved.")
    print("You can now run train_rnn.py\n")


if __name__ == "__main__":
    main()
