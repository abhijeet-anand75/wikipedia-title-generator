# =============================================================================
# config.py
# Central configuration file for the NLP Seq2Seq Title Generation Project
#
# All hyperparameters, paths, and constants are defined here.
# To change any setting, modify only this file — not the task files.
# =============================================================================

import os
import torch

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# PATHS
# =============================================================================

# ── Raw Data ──────────────────────────────────────────────────────────────────
# Paths are workspace-relative by default so the project can run outside Colab.
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
# TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
# TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
# GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.300d.txt")

# # ── Processed Data ────────────────────────────────────────────────────────────
# PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
# TRAIN_PROCESSED = os.path.join(PROCESSED_DIR, "preprocessed_train.csv")
# VAL_PROCESSED   = os.path.join(PROCESSED_DIR, "preprocessed_validation.csv")
# TEST_PROCESSED  = os.path.join(PROCESSED_DIR, "preprocessed_test.csv")

# # ── Saved Artifacts ───────────────────────────────────────────────────────────
# VOCAB_SAVE_PATH   = os.path.join(PROCESSED_DIR, "vocab.pkl")
# RNN_MODEL_PATH    = os.path.join(BASE_DIR, "checkpoints", "best_rnn_model.pt")
# T5_OUTPUT_DIR     = os.path.join(BASE_DIR, "checkpoints", "t5-title-gen")
# RESULTS_SAVE_PATH = os.path.join(BASE_DIR, "results", "results_all.json")

# =============================================================================
# PATHS
# =============================================================================

# ── Google Drive File IDs ─────────────────────────────────────────────────────
# Used by gdown to download files directly from Drive
DRIVE_TRAIN_ID = "1cYBtTR0d6Iv15giUQIySY7MfBZXVCEXC"
DRIVE_TEST_ID  = "1cVHnB0Hz6wXUpzgVV3Z2iqfFWSTtb7UI"
DRIVE_GLOVE_ID = "1cTO_lr8-ttiQbjb5Na7TXd51M6kgpxV_"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Raw Data (local project storage) ─────────────────────────────────────────
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.300d.txt")

# ── Processed Data ────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PROCESSED = os.path.join(PROCESSED_DIR, "preprocessed_train.csv")
VAL_PROCESSED   = os.path.join(PROCESSED_DIR, "preprocessed_validation.csv")
TEST_PROCESSED  = os.path.join(PROCESSED_DIR, "preprocessed_test.csv")

# ── Saved Artifacts ────────────────────────────────────────────────────────────
VOCAB_SAVE_PATH   = os.path.join(PROCESSED_DIR, "vocab.pkl")
RNN_MODEL_PATH    = os.path.join(BASE_DIR, "checkpoints", "best_rnn_model.pt")
T5_OUTPUT_DIR     = os.path.join(BASE_DIR, "checkpoints", "t5-title-gen")
RESULTS_SAVE_PATH = os.path.join(BASE_DIR, "results_all.json")

# ── Google Drive backup folder ────────────────────────────────────────────────
# Results are copied here after training to survive session disconnect
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/wikipedia-title-generator-results"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# PREPROCESSING — preprocess.py
# =============================================================================

VAL_SIZE     = 500    # Articles to extract as validation set from train
MIN_DOC_FREQ = 0.01   # Vocabulary: keep tokens in >= 1% of training documents
                      # On ~13,500 train docs → min_count = 135 documents

# =============================================================================
# RNN SEQ2SEQ MODEL — train_rnn.py
# =============================================================================

# ── Architecture ──────────────────────────────────────────────────────────────
HIDDEN_DIM    = 300   # GRU hidden state dimension
EMBEDDING_DIM = 300   # Word embedding dimension (must match GloVe 6B 300d)
DROPOUT       = 0.3   # Dropout probability for regularization
N_LAYERS      = 1     # Number of GRU layers

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16      # Reduce to 8 if CUDA OOM on Colab
LEARNING_RATE   = 0.001   # Adam optimizer learning rate
EPOCHS_BASIC    = 30      # Epochs for B1 basic RNN
EPOCHS_IMPROVED = 15      # Epochs for B2 improved variants
                          # (fewer needed — GloVe gives head start)
CLIP            = 1.0     # Gradient clipping norm (prevents exploding gradients)
TEACHER_FORCING = 0.5     # 50% teacher forcing during training

# ── Inference ─────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 40   # Maximum tokens to generate per title
BEAM_WIDTH     = 3    # Beam width for RNN beam search

# ── Special Tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_IDX   = 0         # Padding index — loss ignores this
BOS_IDX   = 1         # Begin of sequence — fed as first decoder input
EOS_IDX   = 2         # End of sequence — decoding stops here
UNK_IDX   = 3         # Unknown token — used for OOV words

# ── HierEncoderRNN ────────────────────────────────────────────────────────────
SENT_CHUNK_SIZE = 20  # Tokens per chunk (proxy for sentence boundary)
                      # Avg Wikipedia sentence ~ 20 tokens

# =============================================================================
# TRANSFORMER MODELS — train_transformer.py
# =============================================================================

# ── Model Names (HuggingFace Hub) ─────────────────────────────────────────────
T5_MODEL_NAME = "google-t5/t5-small"      # C1 — fine-tuned
FLAN_T5_BASE  = "google/flan-t5-base"     # C2 — zero-shot
FLAN_T5_LARGE = "google/flan-t5-large"    # C2 — zero-shot

# ── Tokenization ──────────────────────────────────────────────────────────────
T5_MAX_INPUT_LENGTH  = 512   # T5 max sequence length — truncate beyond this
T5_MAX_TARGET_LENGTH = 30    # Max title tokens (Wikipedia titles are short)

# ── Fine-tuning (T5-small) ────────────────────────────────────────────────────
T5_BATCH_SIZE       = 8      # Per-device batch size (fits Colab 12-16GB GPU)
T5_LEARNING_RATE    = 5e-5   # Standard fine-tuning LR for T5
T5_EPOCHS           = 3      # Increase to 3-5 if Colab session allows
                             # 1 epoch ≈ 3.5 hours on Colab free GPU
T5_WEIGHT_DECAY     = 0.01
T5_GRAD_ACCUM_STEPS = 4      # Effective batch = 8 * 4 = 32
T5_SAVE_TOTAL_LIMIT = 2      # Keep only 2 checkpoints to save Drive space
T5_LOGGING_STEPS    = 100

# ── Inference ─────────────────────────────────────────────────────────────────
T5_BEAM_WIDTH  = 5    # Beam width for transformer beam search
T5_MAX_GEN_LEN = 50   # Max tokens to generate during inference

# ── Prompt Templates for Flan-T5 Zero-Shot ───────────────────────────────────
# Two meaningfully different prompt strategies:
#
# Variant 1 — Direct instruction (task-prefix style):
#   Matches T5 pre-training convention ("summarize: ...")
#   Simple, reliable, no extra tokens wasted
#
# Variant 2 — Few-shot with one example:
#   Shows the model the expected input→output format before the actual input
#   Typically helps smaller models (base) more than larger ones
#   (larger models already understand the task from pre-training)
#
# Both variants use {} as placeholder for the article text
PROMPT_VARIANTS = {
    "variant1": (
        "summarize: {}"
    ),
    "variant2": (
        "Given a Wikipedia article, write a short title for it.\n"
        "Article: Mount Everest is the Earth's highest mountain above sea level, "
        "located in the Himalayas on the border between Nepal and Tibet.\n"
        "Title: Mount Everest\n"
        "Article: {}\n"
        "Title:"
    ),
}

# =============================================================================
# EXPERIMENT VARIANTS — train_rnn.py
# All 6 RNN model combinations to train and evaluate.
#
# Format: variant_name: (encoder_type, decoder_type, use_glove, decode_method)
#
# encoder_type  : "EncoderRNN"     → standard bidirectional GRU
#               : "HierEncoderRNN" → hierarchical word+sentence GRU
# decoder_type  : "DecoderRNN"     → single-layer GRU decoder
#               : "Decoder2RNN"    → two-layer stacked GRU decoder
# use_glove     : True  → load GloVe 6B 300d into encoder embedding
#               : False → random embedding initialisation
# decode_method : "greedy" → argmax at each step
#               : "beam"   → beam search with BEAM_WIDTH beams
# =============================================================================

EXPERIMENT_VARIANTS = {
    "B1_basic"   : ("EncoderRNN",     "DecoderRNN",  False, "greedy"),
    "B2_glove"   : ("EncoderRNN",     "DecoderRNN",  True,  "greedy"),
    "B2_hier"    : ("HierEncoderRNN", "DecoderRNN",  False, "greedy"),
    "B2_decoder2": ("EncoderRNN",     "Decoder2RNN", False, "greedy"),
    "B2_beam"    : ("EncoderRNN",     "DecoderRNN",  False, "beam"),
    "B2_all"     : ("HierEncoderRNN", "Decoder2RNN", True,  "beam"),
}
