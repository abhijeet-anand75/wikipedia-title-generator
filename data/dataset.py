# =============================================================================
# data/dataset.py
# PyTorch Dataset and DataLoader setup for Wikipedia title generation
#
# Contents:
#   - WikiDataset         : torch.utils.data.Dataset subclass
#   - collate_fn          : pads variable-length sequences in a batch
#   - get_dataloader      : builds a DataLoader for a given split
#   - get_all_dataloaders : convenience wrapper that returns all three splits
# =============================================================================

import torch
from torch.utils.data   import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

from config import (
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    BATCH_SIZE,
    RANDOM_STATE,
)
from utils import Vocabulary


# =============================================================================
# DATASET
# =============================================================================

class WikiDataset(Dataset):
    """
    PyTorch Dataset for Wikipedia article → title pairs.

    Each item returns:
        text_tensor  : LongTensor of token indices for the article body
                       (no special tokens — encoder handles its own padding)
        title_tensor : LongTensor of token indices for the title,
                       wrapped with <bos> at start and <eos> at end
        text_len     : int, original (un-padded) length of text_tensor

    Parameters
    ----------
    df           : pd.DataFrame
        Must contain columns `text_col` and `title_col`.
    vocab        : Vocabulary
        Shared vocabulary built from the training set.
    text_col     : str
        Column name for preprocessed article body text.
    title_col    : str
        Column name for preprocessed article title.
    max_text_len : int or None
        Truncate article body to this many tokens if set.
        None means no truncation (not recommended for long articles).
    max_title_len : int or None
        Truncate title to this many tokens if set.
    """

    def __init__(
        self,
        df            : pd.DataFrame,
        vocab         : Vocabulary,
        text_col      : str  = "processed_text",
        title_col     : str  = "processed_title",
        max_text_len  : int  = 400,
        max_title_len : int  = 30,
    ):
        self.vocab         = vocab
        self.text_col      = text_col
        self.title_col     = title_col
        self.max_text_len  = max_text_len
        self.max_title_len = max_title_len

        # Drop rows with empty text or title after preprocessing
        self.df = df.dropna(subset=[text_col, title_col]).reset_index(drop=True)
        self.df = self.df[
            (self.df[text_col].str.strip()  != "") &
            (self.df[title_col].str.strip() != "")
        ].reset_index(drop=True)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns
        -------
        text_tensor  : LongTensor  shape (text_len,)
        title_tensor : LongTensor  shape (title_len + 2,)  — includes bos/eos
        text_len     : int
        """
        row = self.df.iloc[idx]

        # ── Numericalize text ──────────────────────────────────────────────
        text_indices = self.vocab.numericalize(str(row[self.text_col]))

        # Truncate if needed
        if self.max_text_len is not None:
            text_indices = text_indices[: self.max_text_len]

        # Must have at least one token
        if len(text_indices) == 0:
            text_indices = [self.vocab.stoi["<unk>"]]

        text_tensor = torch.tensor(text_indices, dtype=torch.long)
        text_len    = len(text_indices)

        # ── Numericalize title ─────────────────────────────────────────────
        # Wrap with <bos> ... <eos> so the decoder sees begin/end markers
        title_indices = self.vocab.numericalize(str(row[self.title_col]))

        if self.max_title_len is not None:
            title_indices = title_indices[: self.max_title_len]

        title_indices = [BOS_IDX] + title_indices + [EOS_IDX]
        title_tensor  = torch.tensor(title_indices, dtype=torch.long)

        return text_tensor, title_tensor, text_len

    # ── Helper ────────────────────────────────────────────────────────────────

    def get_sample(self, idx: int) -> dict:
        """
        Return a human-readable dict for inspection / debugging.

        Example
        -------
        >>> sample = dataset.get_sample(0)
        >>> print(sample["raw_text"][:80])
        >>> print(sample["raw_title"])
        """
        row = self.df.iloc[idx]
        text_tensor, title_tensor, text_len = self[idx]
        return {
            "raw_text"     : row[self.text_col],
            "raw_title"    : row[self.title_col],
            "text_indices" : text_tensor.tolist(),
            "title_indices": title_tensor.tolist(),
            "text_len"     : text_len,
            "title_decoded": self.vocab.decode(title_tensor.tolist()),
        }


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_fn(batch: list) -> tuple:
    """
    Custom collate function for DataLoader.

    Pads variable-length sequences within a batch to the longest sequence
    in that batch.  Texts are sorted by descending length so they can be
    used with pack_padded_sequence in the encoder.

    Parameters
    ----------
    batch : list of (text_tensor, title_tensor, text_len)

    Returns
    -------
    texts_padded  : LongTensor  shape (batch_size, max_text_len)
    titles_padded : LongTensor  shape (batch_size, max_title_len)
    text_lens     : LongTensor  shape (batch_size,)   — original lengths
    """
    texts, titles, text_lens = zip(*batch)

    # Sort by text length descending — required for pack_padded_sequence
    sorted_indices = sorted(
        range(len(text_lens)), key=lambda i: text_lens[i], reverse=True
    )
    texts     = [texts[i]     for i in sorted_indices]
    titles    = [titles[i]    for i in sorted_indices]
    text_lens = [text_lens[i] for i in sorted_indices]

    # Pad sequences — pad_sequence expects list of 1-D tensors
    # batch_first=True → output shape (batch, max_len)
    texts_padded  = pad_sequence(texts,  batch_first=True, padding_value=PAD_IDX)
    titles_padded = pad_sequence(titles, batch_first=True, padding_value=PAD_IDX)
    text_lens     = torch.tensor(text_lens, dtype=torch.long)

    return texts_padded, titles_padded, text_lens


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_dataloader(
    df            : pd.DataFrame,
    vocab         : Vocabulary,
    batch_size    : int  = BATCH_SIZE,
    shuffle       : bool = True,
    text_col      : str  = "processed_text",
    title_col     : str  = "processed_title",
    max_text_len  : int  = 400,
    max_title_len : int  = 30,
    num_workers   : int  = 0,
) -> DataLoader:
    """
    Build a DataLoader for a single data split.

    Parameters
    ----------
    df            : pd.DataFrame   Data for this split
    vocab         : Vocabulary     Shared vocabulary
    batch_size    : int            Mini-batch size
    shuffle       : bool           Shuffle before each epoch (True for train)
    text_col      : str            Column name for article body
    title_col     : str            Column name for article title
    max_text_len  : int            Truncate texts to this length
    max_title_len : int            Truncate titles to this length
    num_workers   : int            DataLoader worker processes (0 = main process)

    Returns
    -------
    DataLoader
    """
    dataset = WikiDataset(
        df            = df,
        vocab         = vocab,
        text_col      = text_col,
        title_col     = title_col,
        max_text_len  = max_text_len,
        max_title_len = max_title_len,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return loader


def get_all_dataloaders(
    train_df      : pd.DataFrame,
    val_df        : pd.DataFrame,
    test_df       : pd.DataFrame,
    vocab         : Vocabulary,
    batch_size    : int  = BATCH_SIZE,
    text_col      : str  = "processed_text",
    title_col     : str  = "processed_title",
    max_text_len  : int  = 400,
    max_title_len : int  = 30,
) -> tuple:
    """
    Convenience wrapper that builds DataLoaders for all three splits.

    Train split  → shuffled
    Val split    → not shuffled
    Test split   → not shuffled, batch_size=1 (for per-sample inference)

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    kwargs = dict(
        vocab         = vocab,
        text_col      = text_col,
        title_col     = title_col,
        max_text_len  = max_text_len,
        max_title_len = max_title_len,
    )

    train_loader = get_dataloader(
        train_df, batch_size=batch_size, shuffle=True,  **kwargs
    )
    val_loader   = get_dataloader(
        val_df,   batch_size=batch_size, shuffle=False, **kwargs
    )
    # Test loader uses batch_size=1 so beam search can run per-sample
    test_loader  = get_dataloader(
        test_df,  batch_size=1,          shuffle=False, **kwargs
    )

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test samples  : {len(test_loader)}")

    return train_loader, val_loader, test_loader