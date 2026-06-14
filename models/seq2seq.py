# =============================================================================
# models/seq2seq.py
# Seq2Seq model that combines encoder and decoder for title generation
#
# Contents:
#   - Seq2seqRNN : Main model class combining EncoderRNN/HierEncoderRNN
#                  with DecoderRNN/Decoder2RNN
#                  Supports teacher forcing (training) and
#                  greedy / beam search (inference)
# =============================================================================

import random
import torch
import torch.nn as nn

from config import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    MAX_NEW_TOKENS,
    BEAM_WIDTH,
    TEACHER_FORCING,
    DEVICE,
)


# =============================================================================
# SEQ2SEQ RNN
# =============================================================================

class Seq2seqRNN(nn.Module):
    """
    Sequence-to-Sequence model for Wikipedia title generation.

    Combines an encoder (EncoderRNN or HierEncoderRNN) with a decoder
    (DecoderRNN or Decoder2RNN).

    Training  — Teacher forcing:
        The ground-truth token is fed as decoder input at each step with
        probability `teacher_forcing_ratio`.  The model's own prediction
        is used otherwise.  This stabilises early training.

    Inference — Two strategies:
        1. Greedy search  : select the highest-probability token at each step.
        2. Beam search    : maintain top-k candidate sequences and return
                            the globally best one.

    Parameters
    ----------
    encoder              : nn.Module
        EncoderRNN or HierEncoderRNN instance.
    decoder              : nn.Module
        DecoderRNN or Decoder2RNN instance.
    vocab_size           : int
        Size of the shared vocabulary.
    teacher_forcing_ratio: float
        Probability of using ground-truth token during training (default 0.5).
    device               : torch.device
        Target computation device.
    """

    def __init__(
        self,
        encoder               : nn.Module,
        decoder               : nn.Module,
        vocab_size            : int,
        teacher_forcing_ratio : float        = TEACHER_FORCING,
        device                : torch.device = DEVICE,
    ):
        super(Seq2seqRNN, self).__init__()

        self.encoder               = encoder
        self.decoder               = decoder
        self.vocab_size            = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device                = device

        # Validate hidden dimensions match between encoder and decoder
        assert encoder.hidden_dim == decoder.hidden_dim, (
            f"Encoder hidden_dim ({encoder.hidden_dim}) must match "
            f"decoder hidden_dim ({decoder.hidden_dim})"
        )

    # =========================================================================
    # FORWARD — training with teacher forcing
    # =========================================================================

    def forward(
        self,
        text       : torch.Tensor,          # (batch, text_len)
        text_len   : torch.Tensor,          # (batch,)
        title      : torch.Tensor,          # (batch, title_len)  includes bos/eos
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Parameters
        ----------
        text     : LongTensor  (batch_size, text_len)
        text_len : LongTensor  (batch_size,)
        title    : LongTensor  (batch_size, title_len)
            Ground-truth title including <bos> at index 0 and <eos> at end.

        Returns
        -------
        outputs : FloatTensor  (batch_size, title_len - 1, vocab_size)
            Raw logits for each predicted position (position 0 = first token
            after <bos>, position title_len-2 = <eos> prediction).
        """
        batch_size = title.shape[0]
        title_len  = title.shape[1]

        # Storage for decoder outputs at each time step
        # We predict title_len - 1 tokens (everything after <bos>)
        outputs = torch.zeros(
            batch_size, title_len - 1, self.vocab_size
        ).to(self.device)

        # ── Encode ────────────────────────────────────────────────────────────
        hidden = self.encoder(text, text_len)   # (batch, hidden_dim)

        # ── Decode ────────────────────────────────────────────────────────────
        # First decoder input is always the <bos> token
        dec_input = title[:, 0].unsqueeze(1)    # (batch, 1)

        for t in range(1, title_len):
            # One decoder step — use forward_train for raw logits
            logits, hidden = self.decoder.forward_train(dec_input, hidden)
            # logits : (batch, vocab_size)

            outputs[:, t - 1] = logits

            # Teacher forcing decision
            use_teacher = random.random() < self.teacher_forcing_ratio
            if use_teacher:
                # Feed ground-truth token
                dec_input = title[:, t].unsqueeze(1)
            else:
                # Feed model's own best prediction
                dec_input = logits.argmax(dim=1).unsqueeze(1)

        return outputs   # (batch, title_len - 1, vocab_size)

    # =========================================================================
    # GREEDY DECODING — inference
    # =========================================================================

    def greedy_decode(
        self,
        
        text          : torch.Tensor,          # (1, text_len)
        text_len      : torch.Tensor,          # (1,)
        max_new_tokens: int = MAX_NEW_TOKENS,
        vocab_itos    : dict = None,
    ) -> str:
        """
        Generate a title using greedy decoding (argmax at each step).

        Parameters
        ----------
        text           : LongTensor  (1, text_len)  — single sample
        text_len       : LongTensor  (1,)
        max_new_tokens : int    Maximum tokens to generate
        vocab_itos     : dict   index → token mapping for decoding to string

        Returns
        -------
        str : generated title as a space-joined token string
        """
        self.eval()

        text = text.to(self.device)
        text_len = text_len.to(self.device)

        with torch.no_grad():
            # Encode
            hidden = self.encoder(text, text_len)   # (1, hidden_dim)

            # Start with <bos>
            dec_input = torch.tensor([[BOS_IDX]], dtype=torch.long).to(self.device)

            generated = []

            for _ in range(max_new_tokens):
                log_probs, hidden = self.decoder(dec_input, hidden)
                # log_probs : (1, vocab_size)

                # Greedy — pick the highest probability token
                top_token = log_probs.argmax(dim=1).item()

                # Stop at <eos>
                if top_token == EOS_IDX:
                    break

                # Skip padding and special tokens in output
                if top_token not in (PAD_IDX, BOS_IDX, EOS_IDX):
                    generated.append(top_token)

                dec_input = torch.tensor(
                    [[top_token]], dtype=torch.long
                ).to(self.device)

        # Convert indices to tokens
        if vocab_itos is not None:
            tokens = [vocab_itos.get(idx, "<unk>") for idx in generated]
            return " ".join(tokens)

        return generated   # return indices if no vocab provided

    # =========================================================================
    # BEAM SEARCH DECODING — inference
    # =========================================================================

    def beam_search(
        self,
        text          : torch.Tensor,          # (1, text_len)
        text_len      : torch.Tensor,          # (1,)
        beam_width    : int  = BEAM_WIDTH,
        max_new_tokens: int  = MAX_NEW_TOKENS,
        vocab_itos    : dict = None,
    ) -> str:
        """
        Generate a title using beam search decoding.

        Maintains `beam_width` candidate sequences at each step.
        At each time step, all active beams are expanded over the full
        vocabulary; the top-k (beam_width) are retained by cumulative
        log-probability score.  A beam is marked complete when it
        generates <eos> or reaches max_new_tokens.

        Parameters
        ----------
        text           : LongTensor  (1, text_len)
        text_len       : LongTensor  (1,)
        beam_width     : int   Number of beams to maintain
        max_new_tokens : int   Maximum tokens to generate per beam
        vocab_itos     : dict  index → token mapping

        Returns
        -------
        str : best generated title
        """
        self.eval()

        text = text.to(self.device)
        text_len = text_len.to(self.device)

        with torch.no_grad():

            # ── Encode ────────────────────────────────────────────────────────
            hidden = self.encoder(text, text_len)   # (1, hidden_dim)

            # ── Initialise beams ──────────────────────────────────────────────
            # Each beam is a tuple:
            #   (cumulative_log_prob, token_list, hidden_state)
            beams = [(0.0, [], hidden)]

            completed = []   # finished beams (hit <eos>)

            # ── Expand beams ──────────────────────────────────────────────────
            for _ in range(max_new_tokens):

                if not beams:
                    break

                candidates = []

                for score, tokens, h in beams:

                    # Last generated token (or <bos> at the first step)
                    last_token = tokens[-1] if tokens else BOS_IDX
                    dec_input  = torch.tensor(
                        [[last_token]], dtype=torch.long
                    ).to(self.device)

                    # One decoder step
                    log_probs, new_h = self.decoder(dec_input, h)
                    # log_probs : (1, vocab_size)

                    # Get top beam_width tokens for this beam
                    top_log_probs, top_indices = log_probs[0].topk(beam_width)

                    for log_prob, token_idx in zip(
                        top_log_probs.tolist(), top_indices.tolist()
                    ):
                        new_score  = score + log_prob
                        new_tokens = tokens + [token_idx]

                        if token_idx == EOS_IDX:
                            # Beam is complete
                            # Length-normalise score to avoid bias toward short
                            length           = max(len(new_tokens), 1)
                            normalised_score = new_score / length
                            completed.append((normalised_score, new_tokens))
                        else:
                            candidates.append((new_score, new_tokens, new_h))

                if not candidates:
                    break

                # Keep only the top beam_width candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]

            # ── Select best beam ──────────────────────────────────────────────
            # If any beam completed, pick the highest-scoring completed beam.
            # Otherwise fall back to the best active beam.
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_tokens = completed[0][1]
            else:
                # No completed beams — take best active beam
                beams.sort(key=lambda x: x[0], reverse=True)
                best_tokens = beams[0][1]

            # Filter out special tokens from output
            best_tokens = [
                t for t in best_tokens
                if t not in (PAD_IDX, BOS_IDX, EOS_IDX)
            ]

        # Convert indices to token string
        if vocab_itos is not None:
            tokens = [vocab_itos.get(idx, "<unk>") for idx in best_tokens]
            return " ".join(tokens)

        return best_tokens

    # =========================================================================
    # GENERATE — unified inference entry point
    # =========================================================================

    def generate(
        self,
        text           : torch.Tensor,
        text_len       : torch.Tensor,
        decode_method  : str  = "greedy",
        beam_width     : int  = BEAM_WIDTH,
        max_new_tokens : int  = MAX_NEW_TOKENS,
        vocab_itos     : dict = None,
    ) -> str:
        """
        Unified inference entry point.

        Parameters
        ----------
        text           : LongTensor  (1, text_len)
        text_len       : LongTensor  (1,)
        decode_method  : str    "greedy" or "beam"
        beam_width     : int    Beam width (used only when decode_method="beam")
        max_new_tokens : int    Maximum tokens to generate
        vocab_itos     : dict   index → token mapping

        Returns
        -------
        str : generated title
        """
        if decode_method == "greedy":
            return self.greedy_decode(
                text, text_len,
                max_new_tokens = max_new_tokens,
                vocab_itos     = vocab_itos,
            )
        elif decode_method == "beam":
            return self.beam_search(
                text, text_len,
                beam_width     = beam_width,
                max_new_tokens = max_new_tokens,
                vocab_itos     = vocab_itos,
            )
        else:
            raise ValueError(
                f"Unknown decode_method: '{decode_method}'. "
                f"Choose 'greedy' or 'beam'."
            )

    # =========================================================================
    # UTILITY
    # =========================================================================

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total:,}")
        return total

    def __repr__(self) -> str:
        enc_name = self.encoder.__class__.__name__
        dec_name = self.decoder.__class__.__name__
        return (
            f"Seq2seqRNN(\n"
            f"  encoder = {enc_name}(hidden={self.encoder.hidden_dim})\n"
            f"  decoder = {dec_name}(hidden={self.decoder.hidden_dim})\n"
            f"  vocab_size = {self.vocab_size:,}\n"
            f"  teacher_forcing = {self.teacher_forcing_ratio}\n"
            f")"
        )