# =============================================================================
# models/decoder.py
# Decoder architectures for the Seq2Seq title generation model
#
# Contents:
#   - DecoderRNN   : Single-layer unidirectional GRU decoder
#   - Decoder2RNN  : Two-layer GRU decoder (stacked GRUs)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    HIDDEN_DIM,
    EMBEDDING_DIM,
    DROPOUT,
)


# =============================================================================
# DECODER RNN
# =============================================================================

class DecoderRNN(nn.Module):
    """
    Single-layer unidirectional GRU decoder.

    Architecture
    ------------
    At each time step t:
      1. Embed the input token              : (batch, 1) → (batch, 1, embed_dim)
      2. Apply dropout to embeddings
      3. Pass through unidirectional GRU    : → output (batch, 1, hidden_dim)
                                               hidden (1, batch, hidden_dim)
      4. Linear FFN maps output → vocab     : (batch, vocab_size)
      5. Log-softmax over vocab dimension   : probability distribution

    The decoder is auto-regressive:
      - During training  : input is the ground-truth token (teacher forcing)
      - During inference : input is the token predicted at the previous step

    Parameters
    ----------
    vocab_size    : int   Size of the shared vocabulary
    embedding_dim : int   Word embedding dimension (default 300)
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
        super(DecoderRNN, self).__init__()

        self.hidden_dim    = hidden_dim
        self.vocab_size    = vocab_size

        # ── Layers ────────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )

        # Maps GRU output hidden state → vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x      : torch.Tensor,   # (batch, 1)   single token per step
        hidden : torch.Tensor,   # (batch, hidden_dim)
    ) -> tuple:
        """
        Decode one time step.

        Parameters
        ----------
        x      : LongTensor  (batch_size, 1)
            Input token index at the current time step.
        hidden : FloatTensor  (batch_size, hidden_dim)
            Hidden state from the previous time step
            (or encoder output at step 0).

        Returns
        -------
        log_probs : FloatTensor  (batch_size, vocab_size)
            Log-softmax distribution over vocabulary.
        hidden    : FloatTensor  (batch_size, hidden_dim)
            Updated hidden state for the next time step.
        """
        # 1. Embed + dropout  →  (B, 1, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # 2. GRU expects hidden of shape (num_layers, B, hidden_dim)
        output, hidden = self.gru(embedded, hidden.unsqueeze(0))
        # output : (B, 1, hidden_dim)
        # hidden : (1, B, hidden_dim)

        # 3. Map to vocabulary  →  (B, vocab_size)
        logits = self.fc(output.squeeze(1))

        # 4. Log-softmax — numerically stable probability distribution
        #    Note: during training CrossEntropyLoss expects raw logits,
        #    so we use log_softmax only during inference / beam search.
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs, hidden.squeeze(0)

    def forward_train(
        self,
        x      : torch.Tensor,
        hidden : torch.Tensor,
    ) -> tuple:
        """
        Training-specific forward pass that returns raw logits.

        CrossEntropyLoss applies its own softmax internally, so passing
        raw logits avoids the double-softmax problem.

        Parameters
        ----------
        x      : LongTensor  (batch_size, 1)
        hidden : FloatTensor  (batch_size, hidden_dim)

        Returns
        -------
        logits : FloatTensor  (batch_size, vocab_size)   raw logits
        hidden : FloatTensor  (batch_size, hidden_dim)
        """
        embedded       = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden.unsqueeze(0))
        logits         = self.fc(output.squeeze(1))
        return logits, hidden.squeeze(0)


# =============================================================================
# DECODER 2 RNN
# =============================================================================

class Decoder2RNN(nn.Module):
    """
    Two-layer stacked GRU decoder.

    Architecture
    ------------
    At each time step t:
      1. Embed the input token              : (batch, 1) → (batch, 1, embed_dim)
      2. Apply dropout to embeddings
      3. First  GRU  (gru1)                 : embed   → out1, hidden1
         - Initialised with encoder hidden state
      4. Apply dropout to gru1 output
      5. Second GRU  (gru2)                 : out1    → out2, hidden2
         - Also initialised with the same encoder hidden state (per spec)
      6. Linear FFN maps out2 → vocab       : (batch, vocab_size)
      7. Log-softmax over vocab dimension

    Why Two GRUs?
    -------------
    Stacking GRUs allows the decoder to learn increasingly abstract
    representations at each layer.  The first GRU focuses on local token
    dependencies; the second GRU integrates those into higher-level
    patterns that better match the title generation objective.

    Parameters
    ----------
    vocab_size    : int   Size of the shared vocabulary
    embedding_dim : int   Word embedding dimension (default 300)
    hidden_dim    : int   GRU hidden state dimension (default 300)
    dropout       : float Dropout probability applied between layers
    """

    def __init__(
        self,
        vocab_size    : int,
        embedding_dim : int   = EMBEDDING_DIM,
        hidden_dim    : int   = HIDDEN_DIM,
        dropout       : float = DROPOUT,
    ):
        super(Decoder2RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # ── Layers ────────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,
        )

        self.dropout = nn.Dropout(dropout)

        # First GRU — processes token embeddings
        self.gru1 = nn.GRU(
            input_size  = embedding_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )

        # Second GRU — processes output of gru1
        self.gru2 = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )

        # Maps second GRU output → vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x      : torch.Tensor,   # (batch, 1)
        hidden : torch.Tensor,   # (batch, hidden_dim)  encoder context vector
    ) -> tuple:
        """
        Decode one time step through two stacked GRUs.

        Parameters
        ----------
        x      : LongTensor  (batch_size, 1)
            Input token index at the current time step.
        hidden : FloatTensor  (batch_size, hidden_dim)
            Encoder context vector used to initialise BOTH GRUs.
            After the first step, the hidden state from gru2 is passed
            as the new hidden for subsequent steps.

        Returns
        -------
        log_probs : FloatTensor  (batch_size, vocab_size)
        hidden2   : FloatTensor  (batch_size, hidden_dim)
            Hidden state from gru2 — passed as input to next time step.
        """
        # 1. Embed + dropout  →  (B, 1, embed_dim)
        embedded = self.dropout(self.embedding(x))

        # 2. First GRU
        out1, _ = self.gru1(embedded, hidden.unsqueeze(0))
        # out1 : (B, 1, hidden_dim)

        # 3. Dropout between layers
        out1 = self.dropout(out1)

        # 4. Second GRU — initialised with same encoder hidden (per spec)
        out2, hidden2 = self.gru2(out1, hidden.unsqueeze(0))
        # out2    : (B, 1, hidden_dim)
        # hidden2 : (1, B, hidden_dim)

        # 5. Map to vocabulary
        logits = self.fc(out2.squeeze(1))   # (B, vocab_size)

        # 6. Log-softmax
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs, hidden2.squeeze(0)

    def forward_train(
        self,
        x      : torch.Tensor,
        hidden : torch.Tensor,
    ) -> tuple:
        """
        Training forward — returns raw logits for CrossEntropyLoss.

        Parameters
        ----------
        x      : LongTensor  (batch_size, 1)
        hidden : FloatTensor  (batch_size, hidden_dim)

        Returns
        -------
        logits  : FloatTensor  (batch_size, vocab_size)
        hidden2 : FloatTensor  (batch_size, hidden_dim)
        """
        embedded       = self.dropout(self.embedding(x))
        out1, _        = self.gru1(embedded, hidden.unsqueeze(0))
        out1           = self.dropout(out1)
        out2, hidden2  = self.gru2(out1, hidden.unsqueeze(0))
        logits         = self.fc(out2.squeeze(1))
        return logits, hidden2.squeeze(0)