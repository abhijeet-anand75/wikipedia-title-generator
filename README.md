# Wikipedia Title Generator

> Generating concise, accurate titles for Wikipedia articles using Seq2Seq RNN models and state-of-the-art Transformer architectures — trained and evaluated on a curated Wikipedia article dataset.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Vocabulary Construction](#vocabulary-construction)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Comparative Analysis](#comparative-analysis)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements and compares multiple sequence-to-sequence (Seq2Seq) architectures for the task of **automatic Wikipedia article title generation**. Given the full body text of a Wikipedia article, the system generates a concise, accurate title — framing the problem as an extreme document compression and abstractive summarisation task.

The project covers a complete NLP pipeline:

| Stage | Description |
|---|---|
| **Preprocessing** | Text cleaning, stopword removal, lemmatisation, vocabulary building |
| **RNN Seq2Seq** | Six GRU-based encoder-decoder variants with progressive improvements |
| **Transformer** | Fine-tuned T5-small and zero-shot Flan-T5 evaluation |
| **Evaluation** | ROUGE-1, ROUGE-2, ROUGE-L F1 scores across all models |

---

## Problem Statement

Wikipedia contains millions of articles, each with a title that accurately and concisely describes its content. Automating title generation has applications in:

- **Document indexing** — automatically cataloguing new articles
- **Content summarisation** — extreme compression of long documents
- **Information retrieval** — generating descriptive labels for unstructured text
- **NLP benchmarking** — evaluating language model capabilities on a well-defined generation task

The challenge lies in producing titles that are semantically accurate, grammatically correct, and appropriately concise — capturing the essence of a potentially long article in just a few words.

---

## Dataset Description

The dataset consists of Wikipedia article body–title pairs provided as CSV files.

| Split | Samples | Description |
|---|---|---|
| Training | 13,379 | Articles used for model training |
| Validation | 500 | Held-out articles for hyperparameter tuning |
| Test | 100 | Final held-out articles for evaluation |
| **Total** | **13,979** | **Full dataset** |

**Fields used:**
- `text` — Full article body (encoder input)
- `title` — Article title (decoder target)

The validation set is sampled from the training data using a fixed random seed (`seed=42`) for reproducibility.

---

## Data Preprocessing Pipeline

Two separate preprocessing pipelines are applied — one for article bodies (encoder input) and one for titles (decoder target). This separation is critical: applying stopword removal to titles would distort the generation target.

### Article Body Preprocessing (Full Pipeline)

```
Raw Text
    ↓ Lowercase
    ↓ Remove newlines
    ↓ Remove non-ASCII characters
    ↓ Remove punctuation
    ↓ Word tokenisation (NLTK)
    ↓ Stopword removal (NLTK English stopwords)
    ↓ Lemmatisation (WordNetLemmatizer)
    ↓ Processed Text
```

### Title Preprocessing (Light Pipeline)

```
Raw Title
    ↓ Lowercase
    ↓ Remove newlines
    ↓ Remove non-ASCII characters
    ↓ Remove punctuation
    ↓ Processed Title
```

> **Design Decision:** Titles are not stopword-filtered or lemmatised. Removing stopwords from titles distorts the target sequence — for example, "The Battle of Waterloo" would become "Battle Waterloo", which the model could never correctly predict. Surface-form preservation is essential for accurate ROUGE evaluation.

**Preprocessing Time:** 210.34 seconds (on Google Colab T4 GPU)

---

## Vocabulary Construction

The vocabulary is built from training article bodies using a **document-frequency threshold**:

> A token is included only if it appears in **≥ 1%** of training documents.

```
Total training documents : 13,379
Minimum document count   : 134 documents (1% of 13,379)
Final vocabulary size    : 5,842 tokens
Special tokens           : <pad>=0, <bos>=1, <eos>=2, <unk>=3
GloVe coverage           : 5,790 / 5,842 tokens (99.1%)
```

**Design Decision:** Document frequency is used instead of raw token count. A token appearing in 3 documents has no reliable statistical signal — document frequency captures which tokens are genuinely informative across the corpus. The 1% threshold yields a compact vocabulary of ~5,800 tokens where every token has meaningful training signal.

---

## Model Architectures

### B1 — Basic Seq2Seq (Bidirectional GRU)

The baseline encoder-decoder architecture:

```
Encoder:
  Input tokens → Embedding(5842, 300) → Dropout(0.3)
               → BiGRU(300, hidden=300) → Linear(600→300) + tanh
               → Context vector (300d)

Decoder:
  Input token  → Embedding(5842, 300) → Dropout(0.3)
               → GRU(300, hidden=300)
               → Linear(300→5842) → Log-softmax
               → Token probability distribution
```

**Key design choices:**
- Bidirectional GRU captures both left and right context — essential for long articles
- GRU over LSTM: fewer parameters, faster training, comparable performance on medium-length sequences
- Linear projection combines forward and backward hidden states into a single context vector

**Trainable parameters:** 7,069,342

---

### B2 — GloVe-Enhanced Seq2Seq

Identical architecture to B1 with pretrained GloVe 6B 300d word vectors loaded into the encoder embedding layer.

```
Encoder embedding initialised with GloVe 6B 300d vectors
Coverage: 5,790 / 5,842 vocabulary tokens (99.1%)
Embeddings: Fine-tuned during training (not frozen)
```

**Design Decision:** GloVe embeddings are fine-tuned rather than frozen. Freezing would prevent the model from adapting representations to the Wikipedia title generation domain — a small dataset benefits from domain adaptation even when starting from pretrained vectors.

---

### B2 — Hierarchical Encoder (HierEncoderRNN)

A two-level encoder that processes text at both word and sentence levels:

```
Level 1 — Word GRU (Bidirectional):
  Token embeddings → BiGRU → word-level hidden states
  → Mean-pool every 20 tokens → sentence embedding

Level 2 — Sentence GRU (Unidirectional):
  Sentence embeddings → GRU (init from word GRU hidden)
  → Final context vector
```

**Design Decision:** Wikipedia articles are long — a flat GRU can lose early context. The hierarchical encoder first summarises each approximate sentence, then summarises the sequence of sentence summaries, preserving structure at both levels. A chunk size of 20 tokens is used as a proxy for sentence boundaries.

**Trainable parameters:** 7,611,142

---

### B2 — Stacked Decoder (Decoder2RNN)

A deeper decoder with two stacked GRU layers:

```
Input token → Embedding → Dropout
           → GRU₁ (init: encoder hidden) → output₁
           → Dropout
           → GRU₂ (init: encoder hidden) → output₂
           → Linear(300→5842) → Log-softmax
```

**Design Decision:** Both GRUs are initialised with the same encoder hidden state, as specified. GRU₁ handles local token transitions; GRU₂ integrates those into higher-level patterns. Dropout between layers prevents over-reliance on GRU₁.

---

### B2 — Beam Search Decoding

All previous variants use greedy decoding (argmax at each step). Beam search maintains the top-k candidate sequences and returns the globally best one:

```
Beam width     : 3
Scoring        : Cumulative log-probability
Normalisation  : Length-normalised score (÷ sequence length)
Stop criterion : All beams generate <eos> or reach max_new_tokens
Fallback       : Best active beam if no beam completes
```

**Design Decision:** Length normalisation prevents beam search from always preferring short titles — a known issue with unnormalised beam search on generation tasks.

---

### B2 — All Improvements Combined

The final RNN variant combines all four improvements:

```
Encoder  : HierEncoderRNN + GloVe embeddings
Decoder  : Decoder2RNN
Decoding : Beam search (width=3)
Parameters: 8,152,942
```

---

### C1 — Fine-tuned T5-small

```
Model     : google-t5/t5-small (60M parameters)
Approach  : Full fine-tuning on the Wikipedia title generation task
Prefix    : "summarize: " prepended to all inputs
Input     : max 512 tokens (truncated)
Target    : max 30 tokens
```

**Training:**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Gradient accumulation | 4 steps (effective batch = 32) |
| Epochs | 1 |
| Mixed precision | fp16 (GPU) |
| Label masking | -100 for padding tokens |

**Design Decision:** Raw (unpreprocessed) text is used for transformer models. T5's SentencePiece tokenizer handles punctuation and casing internally — preprocessing would remove information the model was pretrained to use.

---

### C2 — Zero-shot Flan-T5 (Base and Large)

Two instruction-tuned models evaluated in zero-shot mode with two distinct prompt strategies:

**Variant 1 — Direct instruction (task-prefix style):**
```
summarize: {article text}
```

**Variant 2 — Few-shot with one example:**
```
Given a Wikipedia article, write a short title for it.
Article: Mount Everest is the Earth's highest mountain above sea level,
         located in the Himalayas on the border between Nepal and Tibet.
Title: Mount Everest
Article: {article text}
Title:
```

---

## Training Configuration

### RNN Models

| Hyperparameter | Value |
|---|---|
| Hidden dimension | 300 |
| Embedding dimension | 300 |
| Dropout | 0.3 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |
| Teacher forcing ratio | 0.5 |
| Max new tokens | 40 |
| Beam width | 3 |
| B1 epochs | 10 |
| B2 epochs | 5 |

### Transformer Models

| Hyperparameter | Value |
|---|---|
| T5-small optimizer | AdamW |
| T5-small learning rate | 5e-5 |
| T5-small batch size | 8 |
| T5-small epochs | 1 |
| Gradient accumulation | 4 steps |
| Weight decay | 0.01 |
| T5 beam width | 5 |
| Max input length | 512 |
| Max target length | 30 |

---

## Evaluation Metrics

All models are evaluated using **ROUGE F1 scores** on the 100-article test set:

| Metric | What it measures |
|---|---|
| **ROUGE-1** | Unigram overlap between generated and reference title |
| **ROUGE-2** | Bigram overlap — measures phrase-level accuracy |
| **ROUGE-L** | Longest Common Subsequence — measures sequence-level structure |

All scores use Porter stemming (`use_stemmer=True`) to reduce inflection mismatches. Scores are macro-averaged across all 100 test samples.

---

## Results

### Part B — RNN Seq2Seq Models

| Model Variant | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time |
|---|---|---|---|---|---|
| B1 Basic Seq2Seq | 1.6647 | 0.3306 | 0.0755 | 0.3306 | 417s |
| B2 + GloVe | 1.7048 | 0.3660 | 0.0884 | 0.3060 | 204s |
| B2 + Hierarchical Encoder | 1.9428 | 0.3358 | 0.0499 | 0.2858 | 233s |
| B2 + Decoder2RNN | 1.6731 | 0.3660 | 0.0822 | 0.3435 | 224s |
| B2 + Beam Search | 1.6804 | **0.3805** | 0.0479 | 0.3405 | 213s |
| B2 All Combined | 2.1432 | 0.2811 | 0.0497 | 0.2811 | 257s |

### Part C — Transformer Models

| Model | Approach | Prompt | Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| T5-small | Fine-tuned | summarize: | Greedy | 0.8769 | 0.6630 | 0.8752 |
| T5-small | Fine-tuned | summarize: | Beam-5 | 0.8810 | 0.6630 | 0.8810 |
| Flan-T5-base | Zero-shot | Variant 1 | Greedy | 0.1617 | 0.0600 | 0.1581 |
| Flan-T5-base | Zero-shot | Variant 1 | Beam-5 | 0.1627 | 0.0663 | 0.1618 |
| Flan-T5-base | Zero-shot | Variant 2 | Greedy | 0.7538 | 0.5055 | 0.7483 |
| Flan-T5-base | Zero-shot | Variant 2 | Beam-5 | 0.7784 | 0.5359 | 0.7784 |
| Flan-T5-large | Zero-shot | Variant 1 | Greedy | 0.2364 | 0.0915 | 0.2346 |
| Flan-T5-large | Zero-shot | Variant 1 | Beam-5 | 0.2341 | 0.0896 | 0.2315 |
| **Flan-T5-large** | **Zero-shot** | **Variant 2** | **Greedy** | **0.8963** | **0.6932** | **0.8947** |
| Flan-T5-large | Zero-shot | Variant 2 | Beam-5 | 0.8836 | 0.6671 | 0.8836 |

### Sample Qualitative Predictions

| Reference Title | B1 Basic | T5-small | Flan-T5-large (V2) |
|---|---|---|---|
| catholic high school singapore | catholic high school school | Catholic High School | Catholic High School |
| minnesota golden | minnesota minnesota | Minnesota Golden Gophers | Minnesota Golden Gophers |
| list of people louisiana | list of people | List of people from Louisiana | List of people from Louisiana |

---

## Comparative Analysis

### 1. Honest RNN Performance
The basic Seq2Seq model achieves ROUGE-1 of 0.33 — consistent with published results for attention-free GRU seq2seq models on short-text generation tasks. The gap between RNN and transformer performance (0.33 vs 0.88) demonstrates the impact of attention mechanisms and large-scale pretraining.

### 2. Individual Improvement Analysis

**GloVe embeddings** provide the most reliable ROUGE-1 improvement (+0.035) with significantly reduced training time (417s → 204s). Pretrained embeddings provide a strong initialisation that requires fewer epochs to achieve comparable performance.

**Decoder2RNN** shows the most consistent validation loss improvement across all epochs — it was still improving at epoch 5, suggesting it has the highest headroom for further gains with additional training.

**Beam Search** achieves the highest ROUGE-1 (0.3805) but the lowest ROUGE-2 (0.0479) among RNN variants. This reflects a known trade-off: beam search finds better individual words but can break local bigram coherence by exploring diverse alternatives.

**Hierarchical Encoder** underperforms relative to expectations. The chunk-based sentence approximation (20 tokens per chunk) is a crude proxy for true sentence boundaries — real sentence-level splitting would likely improve results substantially.

**All Combined** performs worst (ROUGE-1: 0.2811) — below the basic baseline. This counterintuitive result demonstrates that architectural complexity requires proportionally more training. With only 5 epochs, the 8.15M parameter model does not have sufficient time to learn the interaction between all components.

### 3. Prompt Engineering is More Impactful Than Model Size

```
Flan-T5-large + wrong prompt (V1) : ROUGE-1 = 0.2364
Flan-T5-base  + right prompt  (V2) : ROUGE-1 = 0.7538
```

The smaller model with the correct prompt outperforms the larger model with the wrong prompt by 3x. Variant 1 ("summarize:") causes both Flan-T5 models to generate full sentences instead of titles — the model follows the summarisation instruction literally. Variant 2 (few-shot) teaches the model the expected short-title format through a single example.

### 4. Fine-tuned Small vs Zero-shot Large

```
T5-small  (60M, fine-tuned, 1 epoch) : ROUGE-1 = 0.8810
Flan-T5-large (780M, zero-shot, V2)  : ROUGE-1 = 0.8963
```

The zero-shot large model only marginally outperforms the fine-tuned small model despite having 13x more parameters. This demonstrates the power of instruction tuning — Flan-T5 was trained to follow task instructions, making it effective without any task-specific fine-tuning.

### 5. Beam Search Does Not Universally Help

| Model | Greedy | Beam | Direction |
|---|---|---|---|
| T5-small | 0.8769 | 0.8810 | ↑ helped |
| Flan-T5-base V2 | 0.7538 | 0.7784 | ↑ helped |
| Flan-T5-large V2 | **0.8963** | 0.8836 | ↓ hurt |

Beam search hurts Flan-T5-large on Variant 2 because the model already generates near-optimal short titles greedily. Beam search explores longer, more complex alternatives that score lower against the short reference titles.

---

## Repository Structure

```
wikipedia-title-generator/
│
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── .gitignore                   ← Files excluded from version control
│
├── config.py                    ← All hyperparameters and paths
├── utils.py                     ← Shared utilities: preprocessing, vocab, timing
│
├── preprocess.py                ← Entry point: data preprocessing + vocab building
├── train_rnn.py                 ← Entry point: RNN Seq2Seq training + evaluation
├── train_transformer.py         ← Entry point: T5 fine-tuning + Flan-T5 evaluation
│
├── models/
│   ├── __init__.py
│   ├── encoder.py               ← EncoderRNN, HierEncoderRNN
│   ├── decoder.py               ← DecoderRNN, Decoder2RNN
│   └── seq2seq.py               ← Seq2seqRNN (teacher forcing, greedy, beam search)
│
├── data/
│   ├── __init__.py
│   └── dataset.py               ← WikiDataset, collate_fn, DataLoader factory
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py               ← ROUGE computation, results table, experiment loop
│
├── notebooks/
│   ├── PartA.ipynb              ← Preprocessing notebook
│   ├── PartB.ipynb              ← RNN training notebook
│   └── PartC.ipynb              ← Transformer notebook
│
└── report/
    └── Report.pdf               ← Full project report
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abhijeet-anand75/wikipedia-title-generator.git
cd wikipedia-title-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
```

### 4. Configure Paths

Open `config.py` and update the data paths:

```python
# For Google Colab
TRAIN_PATH = "/content/drive/MyDrive/Dataset/train.csv"
TEST_PATH  = "/content/drive/MyDrive/Dataset/test.csv"
GLOVE_PATH = "/content/drive/MyDrive/Dataset/glove.6B.300d.txt"

# For local machine
TRAIN_PATH = "path/to/your/train.csv"
TEST_PATH  = "path/to/your/test.csv"
GLOVE_PATH = "path/to/your/glove.6B.300d.txt"
```

### 5. Download GloVe Vectors

Download `glove.6B.zip` from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/) and extract `glove.6B.300d.txt`.

---

## Usage

### Running on Google Colab (Recommended)

```python
# Step 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2 — Clone repository
!git clone https://github.com/abhijeet-anand75/wikipedia-title-generator.git
%cd wikipedia-title-generator

# Step 3 — Install dependencies
!pip install -r requirements.txt

# Step 4 — Run pipeline in order
!python preprocess.py
!python train_rnn.py
!python train_transformer.py
```

### Running Locally

```bash
# Step 1 — Preprocess data and build vocabulary
python preprocess.py

# Step 2 — Train and evaluate all RNN variants
python train_rnn.py

# Step 3 — Fine-tune T5 and evaluate Flan-T5
python train_transformer.py
```

> **Important:** Always run `preprocess.py` before `train_rnn.py`.
> `train_transformer.py` uses raw text and can run independently.

### Expected Outputs

**After `preprocess.py`:**
```
/content/preprocessed_train.csv        (13,379 rows)
/content/preprocessed_validation.csv  (500 rows)
/content/preprocessed_test.csv        (100 rows)
/content/vocab.pkl                    (5,842 tokens)
```

**After `train_rnn.py`:**
```
/content/best_rnn_model_B1_basic.pt
/content/best_rnn_model_B2_glove.pt
/content/best_rnn_model_B2_hier.pt
/content/best_rnn_model_B2_decoder2.pt
/content/best_rnn_model_B2_beam.pt
/content/best_rnn_model_B2_all.pt
/content/results_all.json
```

**After `train_transformer.py`:**
```
/content/t5-title-gen/                (fine-tuned T5-small checkpoint)
/content/results_all.json             (updated with transformer results)
```

---

## Future Improvements

| Improvement | Expected Impact |
|---|---|
| Add attention mechanism to RNN decoder | Significant ROUGE improvement (+0.10 to +0.20) |
| Train B2_all for 15-20 epochs | Allow combined model to converge properly |
| True sentence boundary splitting in HierEncoder | More accurate hierarchical representation |
| Fine-tune T5-small for 3-5 epochs | Further ROUGE improvement |
| Fine-tune Flan-T5-base with Variant 2 prompt style | Combine best prompt with fine-tuning |
| Copy mechanism (pointer networks) | Better handling of rare named entities |
| Coverage mechanism | Reduce repetition in generated titles |
| Larger vocabulary (0.5% threshold) | Reduce `<unk>` tokens in RNN outputs |

---

## Execution Time Summary

| Stage | Time |
|---|---|
| Preprocessing (13,379 articles) | 222.59s |
| Vocabulary building | 4.78s |
| B1 Basic RNN (10 epochs) | 417.91s |
| B2 GloVe (5 epochs) | 204.70s |
| B2 Hierarchical Encoder (5 epochs) | 233.47s |
| B2 Decoder2 (5 epochs) | 224.05s |
| B2 Beam Search (5 epochs) | 213.82s |
| B2 All Combined (5 epochs) | 257.21s |
| T5-small fine-tuning (1 epoch) | 311.26s |
| T5-small inference (100 samples) | 26.67s |
| Flan-T5 inference (all variants) | 574.49s |
| **Total** | **~2,991s (~50 minutes)** |

*All timings recorded on Google Colab with NVIDIA T4 GPU (15GB VRAM)*

---

## Acknowledgements

- [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/) — Pre-trained word vectors (Pennington et al., 2014)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — T5 and Flan-T5 implementations
- [Google T5](https://huggingface.co/google-t5/t5-small) — `google-t5/t5-small` (Raffel et al., 2020)
- [Google Flan-T5](https://huggingface.co/google/flan-t5-large) — `flan-t5-base`, `flan-t5-large` (Chung et al., 2022)
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [NLTK](https://www.nltk.org/) — Natural language processing toolkit
- [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge) — Evaluation metric (Lin, 2004)

---

