# =============================================================================
# utils.py
# Shared utility functions used across preprocess.py, train_rnn.py, and train_transformer.py
#
# Contents:
#   - Text preprocessing  (preprocess_text, preprocess_title)
#   - Vocabulary class    (build, save, load, numericalize)
#   - Timer context manager
#   - Seed setter
#   - Data loading & splitting
#   - Model helpers       (save, load, count parameters)
#   - Logging helpers     (print section headers, training progress)
# =============================================================================

import os
import re
import time
import random
import string
import pickle
import logging

import torch
import numpy as np
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Download required NLTK data (silent if already present)
# ---------------------------------------------------------------------------
# IN utils.py — define as function
def ensure_nltk_data():
    """Download required NLTK data. Call explicitly before preprocessing."""
    for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab", "stopwords"]:
        nltk.download(pkg, quiet=True)
    print("NLTK data ready.")

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused everywhere)
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stop_words = None

def get_stop_words():
    global _stop_words
    if _stop_words is None:
        ensure_nltk_data()
        _stop_words = set(stopwords.words("english"))
    return _stop_words

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a visible section header to stdout."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + GPU).
    Call this at the very start of every task script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    logger.info(f"Random seed set to {seed}")


# =============================================================================
# TIMER
# =============================================================================

class Timer:
    """
    Context manager that measures elapsed wall-clock time.

    Usage
    -----
    with Timer("Training"):
        train(...)
    # → [Training] 123.45 seconds
    """

    def __init__(self, label: str = ""):
        self.label   = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"[{self.label}] {self.elapsed:.2f} seconds")

    def __repr__(self):
        return f"Timer(label={self.label!r}, elapsed={self.elapsed:.2f}s)"


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for article body text (encoder input).

    Steps
    -----
    1. Lowercase
    2. Replace newlines with spaces
    3. Remove non-ASCII characters
    4. Remove punctuation
    5. Tokenize (NLTK word_tokenize)
    6. Remove English stopwords
    7. Lemmatize (WordNetLemmatizer)

    Parameters
    ----------
    text : str
        Raw article body text.

    Returns
    -------
    str
        Cleaned, space-joined token string.
    """
    if pd.isnull(text) or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Newlines → spaces
    text = re.sub(r"\n+", " ", text)

    # 3. Non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 4. Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Tokenize
    tokens = word_tokenize(text)
    stop_words = get_stop_words()

    # 6. Stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # 7. Lemmatize
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w.strip()]

    return " ".join(tokens)


def preprocess_title(text: str) -> str:
    """
    Light preprocessing for article titles (decoder target).

    Titles are NOT stopword-filtered or lemmatized because:
    - Removing stopwords distorts the target (e.g. "The Battle of X" → "Battle X")
    - Lemmatization changes surface form needed for ROUGE evaluation
    - We need the title to remain as close to natural language as possible

    Steps
    -----
    1. Lowercase
    2. Replace newlines with spaces
    3. Remove non-ASCII characters
    4. Remove punctuation

    Parameters
    ----------
    text : str
        Raw article title.

    Returns
    -------
    str
        Cleaned title string.
    """
    if pd.isnull(text) or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    return " ".join(text.split())   # normalise whitespace


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
    """
    Word-level vocabulary built from training corpus using document-frequency
    threshold (tokens must appear in >= min_doc_freq fraction of documents).

    Special tokens
    --------------
    Index 0 → <pad>   (padding)
    Index 1 → <bos>   (begin of sequence)
    Index 2 → <eos>   (end of sequence)
    Index 3 → <unk>   (unknown token)

    Attributes
    ----------
    itos : dict[int, str]   index → token
    stoi : dict[str, int]   token → index
    """

    def __init__(self, min_doc_freq: float = 0.01):
        self.min_doc_freq = min_doc_freq
        self.itos = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    # ── Magic methods ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, "
            f"min_doc_freq={self.min_doc_freq})"
        )

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_vocabulary(self, sentence_list: list) -> None:
        """
        Build vocabulary from a list of preprocessed sentences.

        Only tokens whose document frequency >= min_doc_freq * total_docs
        are added.  This matches the assignment spec:
        "tokens that appear in at least 1% of the training set."

        Parameters
        ----------
        sentence_list : list[str]
            Each element is one preprocessed document (space-joined tokens).
        """
        total_docs = len(sentence_list)
        min_count  = max(1, int(self.min_doc_freq * total_docs))

        # Count how many documents each token appears in
        doc_freq = Counter()
        for sentence in sentence_list:
            unique_tokens = set(str(sentence).split())
            doc_freq.update(unique_tokens)

        # Add qualifying tokens (sorted for determinism)
        idx = 4
        for word in sorted(doc_freq.keys()):
            if doc_freq[word] >= min_count:
                self.itos[idx] = word
                self.stoi[word] = idx
                idx += 1

        logger.info(
            f"Vocabulary built — size: {len(self):,} | "
            f"min_count: {min_count} / {total_docs} docs"
        )

    # ── Numericalize ──────────────────────────────────────────────────────────

    def numericalize(self, text: str) -> list:
        """
        Convert a space-joined token string to a list of integer indices.
        Unknown tokens map to UNK_IDX (3).
        """
        unk_idx = self.stoi["<unk>"]
        return [
            self.stoi.get(token, unk_idx)
            for token in str(text).split()
        ]

    def decode(self, indices: list, skip_special: bool = True) -> str:
        """
        Convert a list of integer indices back to a token string.

        Parameters
        ----------
        indices      : list[int]
        skip_special : bool
            If True, <pad>, <bos>, <eos>, <unk> are not included in output.
        """
        special = {"<pad>", "<bos>", "<eos>", "<unk>"}
        tokens  = []
        for idx in indices:
            token = self.itos.get(idx, "<unk>")
            if skip_special and token in special:
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        return " ".join(tokens)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the vocabulary object to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Vocabulary saved → {path}")

    @staticmethod
    def load(path: str) -> "Vocabulary":
        """Load a pickled Vocabulary from disk."""
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        logger.info(f"Vocabulary loaded ← {path} | size: {len(vocab):,}")
        return vocab


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_split(
    train_path   : str,
    test_path    : str,
    val_size     : int  = 500,
    random_state : int  = 42,
) -> tuple:
    """
    Load train and test CSVs, then extract a validation set from training data.

    Parameters
    ----------
    train_path   : str   Path to train.csv
    test_path    : str   Path to test.csv
    val_size     : int   Number of samples to extract as validation set
    random_state : int   Random seed for reproducibility

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Validate required columns
    for col in ["text", "title"]:
        assert col in train_df.columns, f"Column '{col}' missing from train CSV"
        assert col in test_df.columns,  f"Column '{col}' missing from test CSV"

    # Extract validation set
    actual_val_size = min(val_size, len(train_df))
    val_df   = train_df.sample(n=actual_val_size, random_state=random_state)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # Check for missing values
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = df[["text", "title"]].isnull().sum().sum()
        if missing > 0:
            logger.warning(f"{name} set has {missing} missing values — dropping rows")
            df.dropna(subset=["text", "title"], inplace=True)

    logger.info(
        f"Dataset split — "
        f"train: {len(train_df):,} | "
        f"val: {len(val_df):,} | "
        f"test: {len(test_df):,}"
    )

    return train_df, val_df, test_df


# =============================================================================
# MODEL HELPERS
# =============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count and print the number of trainable parameters in a PyTorch model.

    Returns
    -------
    int : total trainable parameter count
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total:,}")
    return total


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved → {path}")


def load_model(
    model  : torch.nn.Module,
    path   : str,
    device : torch.device,
) -> torch.nn.Module:
    """
    Load model weights from a saved state dict.

    Parameters
    ----------
    model  : torch.nn.Module   Instantiated model with matching architecture
    path   : str               Path to saved .pt file
    device : torch.device      Device to map tensors to

    Returns
    -------
    model with loaded weights
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    logger.info(f"Model loaded ← {path}")
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def epoch_time(start_time: float, end_time: float) -> tuple:
    """
    Convert start/end timestamps to minutes and seconds.

    Returns
    -------
    (elapsed_mins, elapsed_secs) : tuple[int, int]
    """
    elapsed = end_time - start_time
    mins    = int(elapsed / 60)
    secs    = int(elapsed - (mins * 60))
    return mins, secs


def print_epoch_stats(
    epoch      : int,
    total      : int,
    train_loss : float,
    val_loss   : float,
    mins       : int,
    secs       : int,
) -> None:
    """Print a formatted training epoch summary line."""
    print(
        f"Epoch [{epoch:02d}/{total:02d}] | "
        f"Time: {mins}m {secs}s | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )
