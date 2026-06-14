# =============================================================================
# models/encoder.py
# Encoder architectures for the Seq2Seq title generation model
#
# Contents:
#   - EncoderRNN     : Bidirectional GRU encoder with optional GloVe embeddings
#   - HierEncoderRNN : Hierarchical encoder — word-level GRU + sentence-level GRU
# =============================================================================

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import (
    HIDDEN_DIM,
    EMBEDDING_DIM,
    DROPOUT,
    SENT_CHUNK_SIZE,
    DEVICE,
)


# =============================================================================
# ENCODER RNN
# =============================================================================

class EncoderRNN(nn.Module):
    """
    Bidirectional GRU encoder.

    Architecture
    ------------
    1. Embedding layer        : vocab_size → embedding_dim
    2. Dropout                : applied to embeddings
    3. Bidirectional GRU      : embedding_dim → hidden_dim (each direction)
    4. Projection layer       : hidden_dim * 2 → hidden_dim  (+ tanh)
       Combines forward and backward final hidden states into a single
       hidden vector that the decoder can consume.

    Why Bidirectional?
    ------------------
    A bidirectional GRU reads the article both left-to-right and
    right-to-left, capturing context from both directions.  This is
    important for title generation where relevant information can appear
    anywhere in the article body.

    Why GRU over LSTM?
    ------------------
    GRUs have fewer parameters (no cell state), train faster, and perform
    comparably on short-to-medium length sequences.  Given Colab GPU
    constraints this is a practical advantage.

    Parameters
    ----------
    vocab_size    : int   Size of the shared vocabulary
    embedding_dim : int   Dimension of word embeddings (default 300)
    hidden_dim    : int   GRU hidden state dimension (default 300)
    dropout       : float Dropout probability
    """

    def __init__(
        self,
        vocab_size    : int,
        embedding_dim : int   = EMBEDDING_DIM,
        hidden_dim    : int   = HIDDEN_DIM,
        dropout       : float = DROPOUT,
    ):
        super(EncoderRNN, self).__init__()

        self.hidden_dim    = hidden_dim
        self.embedding_dim = embedding_dim

        # ── Layers ────────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,             # <pad> index — no gradient through padding
        )

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            bidirectional = True,
            batch_first   = True,
        )

        # Projects concatenated [forward; backward] hidden → hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x     : torch.Tensor,   # (batch, seq_len)   token indices
        x_len : torch.Tensor,   # (batch,)            actual lengths
    ) -> torch.Tensor:
        """
        Encode a batch of token sequences.

        Parameters
        ----------
        x     : LongTensor  (batch_size, seq_len)
        x_len : LongTensor  (batch_size,)   un-padded lengths (sorted desc)

        Returns
        -------
        hidden : FloatTensor  (batch_size, hidden_dim)
            Single context vector per sequence, ready to initialise decoder.
        """
        # 1. Embed + dropout
        embedded = self.dropout(self.embedding(x))   # (B, L, embed_dim)

        # 2. Pack padded sequences for efficient GRU computation
        packed = pack_padded_sequence(
            embedded,
            x_len.cpu(),          # lengths must be on CPU
            batch_first   = True,
            enforce_sorted = True, # batch must be sorted by descending length
        )

        # 3. Bidirectional GRU
        # packed_out  : packed sequence of all hidden states
        # hidden      : (num_directions * num_layers, batch, hidden_dim)
        #             = (2, batch, hidden_dim)
        _, hidden = self.gru(packed)

        # 4. Combine forward and backward final hidden states
        # hidden[-2] = forward final hidden   (batch, hidden_dim)
        # hidden[-1] = backward final hidden  (batch, hidden_dim)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, hidden*2)

        # 5. Project to hidden_dim with tanh non-linearity
        hidden = torch.tanh(self.fc(hidden))   # (B, hidden_dim)

        return hidden

    # ── GloVe Loading ─────────────────────────────────────────────────────────

    def load_embeddings(
        self,
        glove_path  : str,
        vocab_stoi  : dict,
        freeze      : bool = False,
    ) -> None:
        """
        Load pre-trained GloVe vectors into the embedding layer.

        For tokens present in GloVe, the embedding is initialised with the
        pre-trained vector.  Tokens not in GloVe keep their random init.

        Parameters
        ----------
        glove_path : str
            Path to GloVe .txt file  (e.g. glove.6B.300d.txt)
        vocab_stoi : dict
            Mapping of token → index from the shared Vocabulary object
            (pass vocab.stoi)
        freeze : bool
            If True, GloVe vectors are frozen during training.
            If False (default), they are fine-tuned on the task data.

        Raises
        ------
        AssertionError
            If the GloVe vector dimension does not match embedding_dim.
        """
        print(f"Loading GloVe embeddings from: {glove_path}")

        # Read GloVe file and build token → vector mapping
        glove_vectors = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if word in vocab_stoi:          # only load what we need
                    vector = list(map(float, parts[1:]))
                    glove_vectors[word] = vector

        # Validate dimension
        if glove_vectors:
            sample_dim = len(next(iter(glove_vectors.values())))
            assert sample_dim == self.embedding_dim, (
                f"GloVe dim ({sample_dim}) does not match "
                f"embedding_dim ({self.embedding_dim}). "
                f"Check you are loading the correct GloVe file."
            )

        # Copy pre-trained vectors into the embedding weight matrix
        weight_matrix = self.embedding.weight.data   # (vocab_size, embed_dim)
        loaded = 0
        for token, idx in vocab_stoi.items():
            if token in glove_vectors:
                weight_matrix[idx] = torch.tensor(
                    glove_vectors[token], dtype=torch.float
                )
                loaded += 1

        self.embedding.weight.data = weight_matrix

        if freeze:
            self.embedding.weight.requires_grad = False
            print(f"GloVe loaded — {loaded:,} / {len(vocab_stoi):,} tokens "
                  f"(embeddings frozen)")
        else:
            print(f"GloVe loaded — {loaded:,} / {len(vocab_stoi):,} tokens "
                  f"(embeddings trainable)")


# =============================================================================
# HIERARCHICAL ENCODER RNN
# =============================================================================

class HierEncoderRNN(nn.Module):
    """
    Two-level hierarchical encoder.

    Architecture
    ------------
    Level 1 — Word GRU (Bidirectional):
        Processes all token embeddings in the sequence.
        Output hidden states are split into fixed-size chunks
        (each chunk represents one approximate sentence).
        The tokens in each chunk are mean-pooled to produce one
        sentence-level embedding.

    Level 2 — Sentence GRU (Unidirectional):
        Takes the sequence of sentence embeddings as input.
        Initialised with the projected final hidden state from the
        word-level GRU (as specified in the assignment).
        Its final hidden state is returned as the context vector.

    Why Hierarchical?
    -----------------
    Wikipedia articles are long.  A flat GRU processes all tokens
    equally and can lose early context by the time it reaches the end.
    A hierarchical encoder first summarises each sentence, then
    summarises the sequence of sentence summaries — preserving structure
    at both levels.

    Parameters
    ----------
    vocab_size    : int   Size of the shared vocabulary
    embedding_dim : int   Dimension of word embeddings (default 300)
    hidden_dim    : int   GRU hidden state dimension (default 300)
    dropout       : float Dropout probability
    chunk_size    : int   Tokens per sentence chunk (proxy for sentence boundary)
    """

    def __init__(
        self,
        vocab_size    : int,
        embedding_dim : int   = EMBEDDING_DIM,
        hidden_dim    : int   = HIDDEN_DIM,
        dropout       : float = DROPOUT,
        chunk_size    : int   = SENT_CHUNK_SIZE,
    ):
        super(HierEncoderRNN, self).__init__()

        self.hidden_dim    = hidden_dim
        self.embedding_dim = embedding_dim
        self.chunk_size    = chunk_size

        # ── Shared embedding ──────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,
        )

        self.dropout = nn.Dropout(dropout)

        # ── Level 1 : Word-level bidirectional GRU ────────────────────────────
        self.word_gru = nn.GRU(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = 1,
            bidirectional = True,
            batch_first   = True,
        )

        # Projects word GRU output (hidden*2) → hidden_dim
        # Used for:  (a) sentence embeddings  (b) sent GRU init hidden state
        self.word_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Level 2 : Sentence-level unidirectional GRU ───────────────────────
        self.sent_gru = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x     : torch.Tensor,   # (batch, seq_len)
        x_len : torch.Tensor,   # (batch,)
    ) -> torch.Tensor:
        """
        Hierarchically encode a batch of token sequences.

        Parameters
        ----------
        x     : LongTensor  (batch_size, seq_len)
        x_len : LongTensor  (batch_size,)

        Returns
        -------
        hidden : FloatTensor  (batch_size, hidden_dim)
        """
        batch_size = x.size(0)

        # ── Step 1 : Embed ────────────────────────────────────────────────────
        embedded = self.dropout(self.embedding(x))   # (B, L, embed_dim)

        # ── Step 2 : Word-level GRU ───────────────────────────────────────────
        packed = pack_padded_sequence(
            embedded, x_len.cpu(),
            batch_first=True, enforce_sorted=True,
        )
        word_out, word_hidden = self.word_gru(packed)
        word_out, _ = pad_packed_sequence(word_out, batch_first=True)
        # word_out    : (B, L, hidden*2)
        # word_hidden : (2, B, hidden)  [forward, backward]

        # Project word outputs to hidden_dim
        word_out_proj = torch.tanh(self.word_proj(word_out))  # (B, L, hidden)

        # ── Step 3 : Build sentence embeddings via chunk mean-pooling ─────────
        # Split the sequence into chunks of size chunk_size along the time axis
        # Each chunk represents one approximate sentence
        # FIXED — mask out padding before averaging
        mask = (
            torch.arange(word_out_proj.size(1), device=x_len.device)
            [None, :] < x_len[:, None]
        )
        chunks      = word_out_proj.split(self.chunk_size, dim=1)
        mask_chunks = mask.split(self.chunk_size, dim=1)

        sent_embeddings = torch.stack([
            (chunk * m.unsqueeze(-1)).sum(dim=1)
            / m.sum(dim=1, keepdim=True).clamp_min(1)
            for chunk, m in zip(chunks, mask_chunks)
        ], dim=1)
        # sent_embeddings : (B, num_sentences, hidden)

        # ── Step 4 : Build sent GRU initial hidden from word GRU final hidden ─
        # Concatenate forward and backward final hidden states, then project
        # word_hidden[-2] = forward  final hidden : (B, hidden)
        # word_hidden[-1] = backward final hidden : (B, hidden)
        word_h_cat = torch.cat(
            (word_hidden[-2], word_hidden[-1]), dim=1
        )                                                   # (B, hidden*2)
        sent_init_hidden = torch.tanh(
            self.word_proj(word_h_cat)
        ).unsqueeze(0)                                      # (1, B, hidden)

        # ── Step 5 : Sentence-level GRU ───────────────────────────────────────
        _, sent_hidden = self.sent_gru(sent_embeddings, sent_init_hidden)
        # sent_hidden : (1, B, hidden)

        return sent_hidden.squeeze(0)   # (B, hidden)

    # ── GloVe Loading (same interface as EncoderRNN) ──────────────────────────

    def load_embeddings(
        self,
        glove_path : str,
        vocab_stoi : dict,
        freeze     : bool = False,
    ) -> None:
        """
        Load GloVe embeddings into the word embedding layer.
        Same interface as EncoderRNN.load_embeddings().
        """
        print(f"Loading GloVe embeddings from: {glove_path}")

        glove_vectors = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if word in vocab_stoi:
                    glove_vectors[word] = list(map(float, parts[1:]))

        if glove_vectors:
            sample_dim = len(next(iter(glove_vectors.values())))
            assert sample_dim == self.embedding_dim, (
                f"GloVe dim ({sample_dim}) != embedding_dim ({self.embedding_dim})"
            )

        weight_matrix = self.embedding.weight.data
        loaded = 0
        for token, idx in vocab_stoi.items():
            if token in glove_vectors:
                weight_matrix[idx] = torch.tensor(
                    glove_vectors[token], dtype=torch.float
                )
                loaded += 1

        self.embedding.weight.data = weight_matrix

        if freeze:
            self.embedding.weight.requires_grad = False

        print(f"GloVe loaded — {loaded:,} / {len(vocab_stoi):,} tokens "
              f"({'frozen' if freeze else 'trainable'})")