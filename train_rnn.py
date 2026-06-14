# =============================================================================
# train_rnn.py
# RNN Seq2Seq Model Training and Evaluation
#
# This script is the entry point for Part B of the NLP Assignment.
# It covers B1 (basic RNN) and B2 (four improvements) as defined in config.py
#
# Pipeline per experiment variant:
#   1. Load preprocessed data and vocabulary  (output of preprocess.py)
#   2. Build DataLoaders
#   3. Instantiate model  (encoder + decoder combination per variant)
#   4. Optionally load GloVe embeddings
#   5. Train with teacher forcing
#   6. Evaluate on test set  (greedy or beam search)
#   7. Compute and report ROUGE-1, ROUGE-2, ROUGE-L F1 scores
#
# Variants trained (defined in config.EXPERIMENT_VARIANTS):
#   B1_basic     : EncoderRNN    + DecoderRNN   | no GloVe | greedy
#   B2_glove     : EncoderRNN    + DecoderRNN   | GloVe    | greedy
#   B2_hier      : HierEncoderRNN + DecoderRNN  | no GloVe | greedy
#   B2_decoder2  : EncoderRNN    + Decoder2RNN  | no GloVe | greedy
#   B2_beam      : EncoderRNN    + DecoderRNN   | no GloVe | beam
#   B2_all       : HierEncoderRNN + Decoder2RNN | GloVe    | beam
#
# Usage
# -----
#   python train_rnn.py
#
# Prerequisites
# -------------
#   Run preprocess.py first to generate preprocessed CSVs and vocab.pkl
# =============================================================================

import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    DEVICE,
    RANDOM_STATE,
    TRAIN_PROCESSED,
    VAL_PROCESSED,
    TEST_PROCESSED,
    VOCAB_SAVE_PATH,
    GLOVE_PATH,
    RNN_MODEL_PATH,
    HIDDEN_DIM,
    EMBEDDING_DIM,
    DROPOUT,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS_BASIC,
    EPOCHS_IMPROVED,
    CLIP,
    TEACHER_FORCING,
    MAX_NEW_TOKENS,
    BEAM_WIDTH,
    PAD_IDX,
    EXPERIMENT_VARIANTS,
    RESULTS_SAVE_PATH,
)
from utils import (
    set_seed,
    Timer,
    print_section,
    Vocabulary,
    save_model,
    load_model,
    count_parameters,
    epoch_time,
    print_epoch_stats,
    logger,
)
from data.dataset      import get_all_dataloaders
from models.encoder    import EncoderRNN, HierEncoderRNN
from models.decoder    import DecoderRNN, Decoder2RNN
from models.seq2seq    import Seq2seqRNN
from evaluation.metrics import (
    evaluate_rnn,
    evaluate_all_variants,
    print_rouge_table,
    save_results,
)


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def step_load_data() -> tuple:
    """
    Load preprocessed CSVs and vocabulary produced by preprocess.py.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    vocab                     : Vocabulary
    """
    print_section("STEP 1 — Loading Preprocessed Data")

    with Timer("Loading CSVs"):
        train_df = pd.read_csv(TRAIN_PROCESSED)
        val_df   = pd.read_csv(VAL_PROCESSED)
        test_df  = pd.read_csv(TEST_PROCESSED)

    with Timer("Loading Vocabulary"):
        vocab = Vocabulary.load(VOCAB_SAVE_PATH)

    print(f"\n  Train : {len(train_df):,} | Val : {len(val_df):,} | "
          f"Test : {len(test_df):,}")
    print(f"  Vocab size : {len(vocab):,}")

    return train_df, val_df, test_df, vocab


# =============================================================================
# STEP 2 — BUILD DATALOADERS
# =============================================================================

def step_build_loaders(
    train_df : pd.DataFrame,
    val_df   : pd.DataFrame,
    test_df  : pd.DataFrame,
    vocab    : Vocabulary,
) -> tuple:
    """
    Build PyTorch DataLoaders for all three splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    print_section("STEP 2 — Building DataLoaders")

    with Timer("DataLoader construction"):
        train_loader, val_loader, test_loader = get_all_dataloaders(
            train_df   = train_df,
            val_df     = val_df,
            test_df    = test_df,
            vocab      = vocab,
            batch_size = BATCH_SIZE,
        )

    return train_loader, val_loader, test_loader


# =============================================================================
# STEP 3 — BUILD MODEL
# =============================================================================

def build_model(
    encoder_type : str,
    decoder_type : str,
    vocab_size   : int,
    use_glove    : bool,
    vocab_stoi   : dict,
) -> Seq2seqRNN:
    """
    Instantiate a Seq2seqRNN model from named encoder and decoder types.

    Parameters
    ----------
    encoder_type : str    "EncoderRNN" or "HierEncoderRNN"
    decoder_type : str    "DecoderRNN" or "Decoder2RNN"
    vocab_size   : int    Size of shared vocabulary
    use_glove    : bool   Whether to load GloVe embeddings into encoder
    vocab_stoi   : dict   token → index mapping (needed for GloVe loading)

    Returns
    -------
    model : Seq2seqRNN  (on DEVICE)
    """
    # ── Encoder ───────────────────────────────────────────────────────────────
    if encoder_type == "EncoderRNN":
        encoder = EncoderRNN(
            vocab_size    = vocab_size,
            embedding_dim = EMBEDDING_DIM,
            hidden_dim    = HIDDEN_DIM,
            dropout       = DROPOUT,
        )
    elif encoder_type == "HierEncoderRNN":
        encoder = HierEncoderRNN(
            vocab_size    = vocab_size,
            embedding_dim = EMBEDDING_DIM,
            hidden_dim    = HIDDEN_DIM,
            dropout       = DROPOUT,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # ── Decoder ───────────────────────────────────────────────────────────────
    if decoder_type == "DecoderRNN":
        decoder = DecoderRNN(
            vocab_size    = vocab_size,
            embedding_dim = EMBEDDING_DIM,
            hidden_dim    = HIDDEN_DIM,
            dropout       = DROPOUT,
        )
    elif decoder_type == "Decoder2RNN":
        decoder = Decoder2RNN(
            vocab_size    = vocab_size,
            embedding_dim = EMBEDDING_DIM,
            hidden_dim    = HIDDEN_DIM,
            dropout       = DROPOUT,
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    # ── Seq2Seq wrapper ───────────────────────────────────────────────────────
    model = Seq2seqRNN(
        encoder               = encoder,
        decoder               = decoder,
        vocab_size            = vocab_size,
        teacher_forcing_ratio = TEACHER_FORCING,
        device                = DEVICE,
    ).to(DEVICE)

    # ── Load GloVe (optional) ─────────────────────────────────────────────────
    if use_glove:
        if not os.path.exists(GLOVE_PATH):
            logger.warning(
                f"GloVe file not found at {GLOVE_PATH}. "
                f"Skipping GloVe loading — using random embeddings."
            )
        else:
            with Timer("GloVe Loading"):
                model.encoder.load_embeddings(
                    glove_path = GLOVE_PATH,
                    vocab_stoi = vocab_stoi,
                    freeze     = False,   # Fine-tune GloVe vectors during training
                )

    count_parameters(model)
    print(model)

    return model


# =============================================================================
# STEP 4 — TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model        : Seq2seqRNN,
    loader       : torch.utils.data.DataLoader,
    optimizer    : torch.optim.Optimizer,
    criterion    : nn.Module,
    clip         : float = CLIP,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model     : Seq2seqRNN
    loader    : DataLoader   Training data loader
    optimizer : Optimizer
    criterion : Loss function  (CrossEntropyLoss with ignore_index=PAD_IDX)
    clip      : float          Gradient clipping norm

    Returns
    -------
    float : mean training loss for this epoch
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (text, title, text_len) in enumerate(loader):
        text     = text.to(DEVICE)
        title    = title.to(DEVICE)
        text_len = text_len.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass — returns (batch, title_len-1, vocab_size)
        output = model(text, text_len, title)

        # Reshape for CrossEntropyLoss:
        # output : (batch, title_len-1, vocab_size) → (batch*(title_len-1), vocab_size)
        # target : title[:, 1:]  — everything after <bos>
        #        : (batch, title_len-1) → (batch*(title_len-1),)
        output_flat = output.reshape(-1, output.shape[-1])
        target_flat = title[:, 1:].reshape(-1)

        loss = criterion(output_flat, target_flat)

        loss.backward()

        # Gradient clipping — prevents exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(
    model     : Seq2seqRNN,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
) -> float:
    """
    Compute mean loss on a validation loader (no gradient computation).

    Parameters
    ----------
    model     : Seq2seqRNN
    loader    : DataLoader
    criterion : Loss function

    Returns
    -------
    float : mean validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for text, title, text_len in loader:
            text     = text.to(DEVICE)
            title    = title.to(DEVICE)
            text_len = text_len.to(DEVICE)

            output      = model(text, text_len, title)
            output_flat = output.reshape(-1, output.shape[-1])
            target_flat = title[:, 1:].reshape(-1)

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()

    return total_loss / len(loader)


def step_train(
    model        : Seq2seqRNN,
    train_loader : torch.utils.data.DataLoader,
    val_loader   : torch.utils.data.DataLoader,
    variant_name : str,
    n_epochs     : int,
) -> tuple:
    """
    Full training loop with validation and model checkpointing.

    Saves the best model (lowest validation loss) to disk.

    Parameters
    ----------
    model        : Seq2seqRNN
    train_loader : DataLoader
    val_loader   : DataLoader
    variant_name : str          Used for checkpoint filename
    n_epochs     : int          Number of training epochs

    Returns
    -------
    model       : Seq2seqRNN  Best model loaded from checkpoint
    train_losses: list[float]
    val_losses  : list[float]
    """
    print_section(f"TRAINING — {variant_name}")

    # Loss function: CrossEntropy ignoring padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss   = float("inf")
    checkpoint_path = RNN_MODEL_PATH.replace(".pt", f"_{variant_name}.pt")

    train_losses = []
    val_losses   = []

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate_loss(model, val_loader, criterion)

        epoch_end = time.time()
        mins, secs = epoch_time(epoch_start, epoch_end)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, checkpoint_path)
            improved = "✓ (saved)"
        else:
            improved = ""

        print_epoch_stats(epoch, n_epochs, train_loss, val_loss, mins, secs)
        if improved:
            print(f"         {improved}")

    # Load the best checkpoint before returning
    model = load_model(model, checkpoint_path, DEVICE)
    print(f"\nBest validation loss : {best_val_loss:.4f}")

    return model, train_losses, val_losses


# =============================================================================
# STEP 5 — RUN ONE FULL EXPERIMENT VARIANT
# =============================================================================

def run_variant(
    variant_name  : str,
    encoder_type  : str,
    decoder_type  : str,
    use_glove     : bool,
    decode_method : str,
    train_loader  : torch.utils.data.DataLoader,
    val_loader    : torch.utils.data.DataLoader,
    test_loader   : torch.utils.data.DataLoader,
    vocab         : Vocabulary,
) -> dict:
    """
    Run one full experiment:  build → train → evaluate → return ROUGE scores.

    Parameters
    ----------
    variant_name  : str    Label from EXPERIMENT_VARIANTS
    encoder_type  : str    "EncoderRNN" or "HierEncoderRNN"
    decoder_type  : str    "DecoderRNN" or "Decoder2RNN"
    use_glove     : bool   Load GloVe embeddings
    decode_method : str    "greedy" or "beam"
    train_loader  : DataLoader
    val_loader    : DataLoader
    test_loader   : DataLoader
    vocab         : Vocabulary

    Returns
    -------
    dict : ROUGE scores (rouge1, rouge2, rougeL)
    """
    print_section(f"EXPERIMENT: {variant_name}")
    print(f"  Encoder      : {encoder_type}")
    print(f"  Decoder      : {decoder_type}")
    print(f"  GloVe        : {use_glove}")
    print(f"  Decode method: {decode_method}")

    # Number of epochs depends on variant
    n_epochs = EPOCHS_BASIC if variant_name == "B1_basic" else EPOCHS_IMPROVED

    # Build model
    with Timer(f"Model build — {variant_name}"):
        model = build_model(
            encoder_type = encoder_type,
            decoder_type = decoder_type,
            vocab_size   = len(vocab),
            use_glove    = use_glove,
            vocab_stoi   = vocab.stoi,
        )

    # Train
    with Timer(f"Training — {variant_name}"):
        model, train_losses, val_losses = step_train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            variant_name = variant_name,
            n_epochs     = n_epochs,
        )

    # Evaluate
    scores = evaluate_rnn(
        model         = model,
        test_loader   = test_loader,
        vocab_itos    = vocab.itos,
        variant_name  = variant_name,
        decode_method = decode_method,
        beam_width    = BEAM_WIDTH,
        max_new_tokens= MAX_NEW_TOKENS,
        device        = DEVICE,
        show_examples = 5,
    )

    return scores


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run the full Task B pipeline for all experiment variants.

    Steps
    -----
    1.  Set random seed
    2.  Load preprocessed data + vocabulary
    3.  Build DataLoaders
    4.  For each variant in EXPERIMENT_VARIANTS:
        a. Build model
        b. Load GloVe (if applicable)
        c. Train
        d. Evaluate on test set
        e. Record ROUGE scores
    5.  Print comparison table
    6.  Save all results to JSON
    """
    overall_start = time.time()
    all_results   = {}

    # Reproducibility
    set_seed(RANDOM_STATE)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df, vocab = step_load_data()

    # ── Build loaders ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = step_build_loaders(
        train_df, val_df, test_df, vocab
    )

    # ── Run all variants ──────────────────────────────────────────────────────
    for variant_name, (enc_type, dec_type, use_glove, decode_method) \
            in EXPERIMENT_VARIANTS.items():

        variant_start = time.time()

        scores = run_variant(
            variant_name  = variant_name,
            encoder_type  = enc_type,
            decoder_type  = dec_type,
            use_glove     = use_glove,
            decode_method = decode_method,
            train_loader  = train_loader,
            val_loader    = val_loader,
            test_loader   = test_loader,
            vocab         = vocab,
        )

        all_results[variant_name] = scores

        variant_elapsed = time.time() - variant_start
        print(f"\n  [{variant_name}] Total time: {variant_elapsed:.2f} seconds")

        # Free GPU memory between variants
        torch.cuda.empty_cache()

    # ── Final results table ───────────────────────────────────────────────────
    print_rouge_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(all_results, RESULTS_SAVE_PATH)

    total_elapsed = time.time() - overall_start
    print(f"\nTask B complete. Total time: {total_elapsed:.2f} seconds")
    print(f"Results saved to: {RESULTS_SAVE_PATH}")
    print("You can now run train_transformer.py\n")


if __name__ == "__main__":
    main()