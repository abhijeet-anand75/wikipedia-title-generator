# Project Report
## Seq2Seq and Transformer Models for Text Summarisation and Title Generation

**Project:** Wikipedia Article Title Generation  
**Dataset:** Wikipedia Article–Title Dataset (13,979 samples)  
**Models:** RNN Seq2Seq (6 variants) + T5-small + Flan-T5-base + Flan-T5-large  

---

## Table of Contents

1. [Task A — Dataset and Preprocessing](#task-a)
2. [Task B — RNN Seq2Seq Models](#task-b)
3. [Task C — Transformer Models](#task-c)
4. [Overall Comparison and Analysis](#overall-comparison)
5. [References](#references)

---

<a name="task-a"></a>
## 1. Task A — Dataset and Preprocessing

### 1.1 Dataset Details

The dataset consists of Wikipedia article body–title pairs split into training, validation, and test sets.

| Split | Samples |
|---|---|
| Training | 13,379 |
| Validation | 500 |
| Test | 100 |
| **Total** | **13,979** |

The validation set was sampled from the original training data using `random_state=42` for reproducibility. All three splits were confirmed to have zero missing values in both the `text` and `title` columns.

### 1.2 Preprocessing Pipeline

Two separate preprocessing pipelines were applied:

**Article body (encoder input) — Full pipeline:**

| Step | Operation | Rationale |
|---|---|---|
| 1 | Lowercase | Normalise case variation |
| 2 | Remove newlines | Clean formatting artifacts |
| 3 | Remove non-ASCII | Remove encoding noise |
| 4 | Remove punctuation | Reduce vocabulary noise |
| 5 | Word tokenisation (NLTK) | Split into tokens |
| 6 | Stopword removal | Remove low-information words |
| 7 | Lemmatisation (WordNet) | Reduce morphological variants |

**Article title (decoder target) — Light pipeline:**

| Step | Operation | Rationale |
|---|---|---|
| 1 | Lowercase | Normalise case |
| 2 | Remove newlines | Clean artifacts |
| 3 | Remove non-ASCII | Remove noise |
| 4 | Remove punctuation | Clean surface form |

> **Critical Design Decision:** Titles are processed separately from article bodies. Stopword removal and lemmatisation are NOT applied to titles because they distort the generation target — the model must learn to produce natural language titles, and removing stopwords would create targets that are impossible to evaluate correctly with ROUGE.

**Design choice — Lemmatisation over Stemming:**  
Lemmatisation (WordNetLemmatizer) was chosen over stemming (Porter/Snowball) because it returns dictionary-form base words rather than truncated stems. For example, "running" → "run" (lemma) vs "runn" (stem). This produces more semantically meaningful tokens that improve model generalisation.

### 1.3 Vocabulary Construction

| Parameter | Value |
|---|---|
| Method | Document-frequency threshold |
| Threshold | ≥ 1% of training documents |
| Min document count | 134 / 13,379 |
| Final vocabulary size | 5,842 tokens |
| Special tokens | `<pad>=0, <bos>=1, <eos>=2, <unk>=3` |
| GloVe coverage | 5,790 / 5,842 (99.1%) |

**Design choice — Document frequency over raw token count:**  
The assignment specification requires tokens appearing in at least 1% of training documents. This is more meaningful than a raw count threshold because document frequency measures how widely a token is used across the corpus, not just how many times it appears in a few documents.

### 1.4 Task A Execution Times

| Step | Time (seconds) |
|---|---|
| Dataset Loading | 3.23 |
| Text Preprocessing | 210.34 |
| Vocabulary Building | 4.78 |
| Saving Processed Data | 4.24 |
| **Total** | **222.59** |

---

<a name="task-b"></a>
## 2. Task B — RNN Seq2Seq Models

### 2.1 Model Architecture Overview

All RNN models share the same base components with progressive improvements:

**Special Tokens:**  
`<pad>=0` (padding), `<bos>=1` (decoder start), `<eos>=2` (decoder stop), `<unk>=3` (unknown)

**Training Strategy:**  
Teacher forcing with ratio 0.5 — the ground-truth token is fed as decoder input with 50% probability, and the model's own prediction with 50% probability. This stabilises early training while teaching the model to handle its own errors.

**Loss Function:**  
`CrossEntropyLoss(ignore_index=0)` — padding tokens contribute zero loss, preventing artificially low loss from padded sequences.

**Optimiser:** Adam (lr=0.001) with gradient clipping (norm=1.0)

---

### 2.2 B1 — Basic Seq2Seq (Bidirectional GRU)

**Architecture:**

```
EncoderRNN:
  Embedding(5842 → 300) → Dropout(0.3)
  → BiGRU(300, hidden=300)
  → Linear(600 → 300) + tanh
  → Context vector (300d)

DecoderRNN:
  Embedding(5842 → 300) → Dropout(0.3)
  → GRU(300, hidden=300)
  → Linear(300 → 5842) → Log-softmax
```

**Trainable Parameters:** 7,069,342  
**Decoding:** Greedy (argmax at each step)  
**Epochs:** 10

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.8047 | 2.2329 |
| 2 | 1.7859 | 1.8738 |
| 3 | 1.2557 | **1.6647** ← best |
| 4 | 0.8884 | 1.7001 |
| 10 | 0.2304 | 1.9706 |

The model converges at epoch 3 (val loss 1.6647) and then overfits — training loss continues to fall while validation loss rises. This is expected for a model without regularisation beyond dropout.

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.3306 | 0.0755 | 0.3306 | 417.91s | 1.01s |

---

### 2.3 B2 — GloVe-Enhanced Seq2Seq

**Enhancement:** Pre-trained GloVe 6B 300d embeddings loaded into the encoder embedding layer. Embeddings are fine-tuned during training.

**GloVe Coverage:** 5,790 / 5,842 vocabulary tokens (99.1%)  
**Epochs:** 5

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.3243 | 2.6725 |
| 2 | 2.3033 | 2.1622 |
| 3 | 1.7013 | 1.8925 |
| 4 | 1.2891 | 1.7435 |
| 5 | 0.9684 | **1.7048** ← best |

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.3660 | 0.0884 | 0.3060 | 204.70s | 0.98s |

**Key Finding:** GloVe achieves comparable ROUGE-1 to the basic model in half the training time, confirming that pretrained embeddings provide a strong initialisation head start. The model was still improving at epoch 5, suggesting further gains with more training.

---

### 2.4 B2 — Hierarchical Encoder (HierEncoderRNN)

**Enhancement:** Two-level encoder processing text at word and sentence levels.

**Architecture:**
```
Word-level BiGRU → chunk mean-pooling (20 tokens/chunk)
                 → sentence embeddings
Sentence-level GRU (init from word GRU hidden)
                 → context vector
```

**Trainable Parameters:** 7,611,142  
**Epochs:** 5

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.2677 | 2.7467 |
| 3 | 1.9687 | 2.1467 |
| 5 | 1.2316 | **1.9428** ← best |

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.3358 | 0.0499 | 0.2858 | 233.47s | 1.26s |

**Key Finding:** The hierarchical encoder underperforms relative to expectations. Two factors contribute: (1) chunk-based sentence approximation (20 tokens) is a crude proxy for true sentence boundaries; (2) 5 epochs are insufficient for the model to learn the complex two-level structure. Real sentence boundary detection would likely improve results substantially.

---

### 2.5 B2 — Stacked Decoder (Decoder2RNN)

**Enhancement:** Two stacked GRU layers in the decoder, both initialised with the encoder context vector.

**Architecture:**
```
Embedding → Dropout
→ GRU₁ (init: encoder hidden) → Dropout
→ GRU₂ (init: encoder hidden)
→ Linear(300 → 5842) → Log-softmax
```

**Trainable Parameters:** 7,611,142  
**Epochs:** 5

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.8808 | 2.2727 |
| 2 | 1.8522 | 1.8704 |
| 3 | 1.3091 | 1.7007 |
| 4 | 0.9497 | 1.6981 |
| 5 | 0.6983 | **1.6731** ← best |

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.3660 | 0.0822 | 0.3435 | 224.05s | 1.06s |

**Key Finding:** Decoder2 achieves the best ROUGE-L (0.3435) among all RNN variants and was still improving at epoch 5 — the only variant to improve every single epoch. This indicates significant headroom for improvement with more training epochs.

---

### 2.6 B2 — Beam Search Decoding

**Enhancement:** Beam search (width=3) with length normalisation replacing greedy decoding at inference time. Same architecture as B1 encoder + decoder.

**Beam Search Parameters:**
```
Beam width     : 3
Score          : Cumulative log-probability
Normalisation  : Score ÷ sequence length
Fallback       : Best active beam if no beam completes
```

**Epochs:** 5

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.8261 | 2.1835 |
| 3 | 1.2675 | 1.7374 |
| 5 | 0.6486 | **1.6804** ← best |

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.3805 | 0.0479 | 0.3405 | 213.82s | 7.21s |

**Key Finding:** Beam search achieves the highest ROUGE-1 (0.3805) among all RNN variants but the lowest ROUGE-2 (0.0479). This reflects a known beam search trade-off: maintaining multiple candidate sequences finds better individual word matches but can disrupt local bigram coherence. Inference time increases 7× compared to greedy (7.21s vs 1.01s).

---

### 2.7 B2 — All Improvements Combined

**Configuration:**
- Encoder: HierEncoderRNN + GloVe 6B 300d
- Decoder: Decoder2RNN
- Decoding: Beam Search (width=3)
- Parameters: 8,152,942

**Epochs:** 5

**Training Behaviour:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.4962 | 3.1204 |
| 3 | 2.3530 | 2.3991 |
| 5 | 1.7669 | **2.1432** ← best |

**Results:**

| ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time | Inference Time |
|---|---|---|---|---|
| 0.2811 | 0.0497 | 0.2811 | 257.21s | 10.44s |

**Key Finding:** The combined model performs below the basic baseline, demonstrating that architectural complexity requires proportionally more training. The HierEncoder component (which underperformed individually) compounds with the additional complexity, and 5 epochs prove insufficient for 8.15M parameters to converge.

---

### 2.8 Complete RNN Results Comparison

| Model | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | Params |
|---|---|---|---|---|---|
| B1 Basic | 1.6647 | 0.3306 | 0.0755 | 0.3306 | 7.07M |
| B2 GloVe | 1.7048 | 0.3660 | **0.0884** | 0.3060 | 7.07M |
| B2 HierEncoder | 1.9428 | 0.3358 | 0.0499 | 0.2858 | 7.61M |
| B2 Decoder2 | 1.6731 | 0.3660 | 0.0822 | **0.3435** | 7.61M |
| B2 Beam | 1.6804 | **0.3805** | 0.0479 | 0.3405 | 7.07M |
| B2 All | 2.1432 | 0.2811 | 0.0497 | 0.2811 | 8.15M |

### 2.9 Task B Execution Times

| Step | Time (seconds) |
|---|---|
| Loading data + DataLoaders | 2.52 |
| B1 Basic (10 epochs) | 417.91 |
| B2 GloVe loading | 8.69 |
| B2 GloVe (5 epochs) | 204.70 |
| B2 HierEncoder (5 epochs) | 233.47 |
| B2 Decoder2 (5 epochs) | 224.05 |
| B2 Beam Search (5 epochs) | 213.82 |
| B2 All Combined (5 epochs) | 257.21 |
| **Total Task B** | **1,594.17** |

---

<a name="task-c"></a>
## 3. Task C — Transformer Models

**Important design decision:** Raw unpreprocessed text is used for all transformer models. T5's SentencePiece tokenizer handles punctuation, casing, and subword segmentation internally. Preprocessing would remove information the model was pretrained to use.

### 3.1 C1 — Fine-tuned T5-small

**Model:** `google-t5/t5-small` (60M parameters)

**Tokenisation:**

| Setting | Value |
|---|---|
| Task prefix | `"summarize: "` |
| Max input length | 512 tokens |
| Max target length | 30 tokens |
| Label masking | -100 for padding |
| Padding strategy | Dynamic (DataCollatorForSeq2Seq) |

> **Critical implementation detail:** Padding tokens in labels are replaced with `-100` so `CrossEntropyLoss` ignores them. Without this, the model is penalised for correctly ignoring padding, which degrades training quality significantly.

**Fine-tuning Configuration:**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Gradient accumulation | 4 steps |
| Effective batch size | 32 |
| Epochs | 1 |
| Weight decay | 0.01 |
| Mixed precision | fp16 |
| eval_strategy | epoch |

**Training Progress:**

| Epoch | Loss | Grad Norm |
|---|---|---|
| 0.24 | 5.394 | 13.04 |
| 0.48 | 2.643 | 7.926 |
| 0.72 | 2.319 | 7.581 |
| 0.96 | 2.188 | 8.355 |
| **1.00 (eval)** | **0.459** | — |

**Evaluation during training:**
```
eval_rouge1    : 0.8445
eval_rouge2    : 0.5768
eval_rougeL    : 0.8447
eval_runtime   : 16.52 seconds
```

**Test Set Results:**

| Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference Time |
|---|---|---|---|---|
| Greedy (beam=1) | 0.8769 | 0.6630 | 0.8752 | 11.04s |
| Beam (beam=5) | 0.8810 | 0.6630 | 0.8810 | 15.61s |

**Sample Predictions (T5-small greedy):**

| Reference | Prediction |
|---|---|
| Weyburn | Weyburn, Saskatchewan |
| Catholic High School, Singapore | Catholic High School |
| Minnesota Golden Gophers | Minnesota Golden Gophers |

---

### 3.2 C2 — Zero-shot Flan-T5-base

**Model:** `google/flan-t5-base` (250M parameters)

**Prompt Variant 1 — Direct instruction:**
```
summarize: {article text}
```

**Prompt Variant 2 — Few-shot with one example:**
```
Given a Wikipedia article, write a short title for it.
Article: Mount Everest is the Earth's highest mountain...
Title: Mount Everest
Article: {article text}
Title:
```

**Results:**

| Prompt | Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference |
|---|---|---|---|---|---|
| Variant 1 | Greedy | 0.1617 | 0.0600 | 0.1581 | 42.07s |
| Variant 1 | Beam-5 | 0.1627 | 0.0663 | 0.1618 | 48.07s |
| Variant 2 | Greedy | 0.7538 | 0.5055 | 0.7483 | 19.06s |
| Variant 2 | Beam-5 | 0.7784 | 0.5359 | 0.7784 | 26.79s |

**Sample Predictions (Variant 1 vs Variant 2):**

| Reference | V1 Prediction | V2 Prediction |
|---|---|---|
| Weyburn | Weyburn is a city in Saskatchewan. | Weyburn, Saskatchewan |
| Catholic High School, Singapore | Catholic school in Singapore | Sino-English Catholic School |

**Key Finding:** Variant 1 causes the model to generate full descriptive sentences instead of short titles — it follows the "summarize:" instruction literally and produces a summary. Variant 2's few-shot example teaches the expected format (short title), resulting in 4.7× higher ROUGE-1 (0.16 → 0.75).

---

### 3.3 C2 — Zero-shot Flan-T5-large

**Model:** `google/flan-t5-large` (780M parameters)

**Results:**

| Prompt | Decoding | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference |
|---|---|---|---|---|---|
| Variant 1 | Greedy | 0.2364 | 0.0915 | 0.2346 | 124.07s |
| Variant 1 | Beam-5 | 0.2341 | 0.0896 | 0.2315 | 151.47s |
| Variant 2 | Greedy | **0.8963** | **0.6932** | **0.8947** | 33.68s |
| Variant 2 | Beam-5 | 0.8836 | 0.6671 | 0.8836 | 51.56s |

**Sample Predictions (Variant 1 vs Variant 2):**

| Reference | V1 Prediction | V2 Prediction |
|---|---|---|
| Weyburn | Weyburn is the eleventh-largest city in Saskatchewan, Canada. | Weyburn, Saskatchewan |
| Catholic High School, Singapore | Catholic High School is a government-aided autonomous Catholic boys' school in Bishan, Singapore, founded in 1935... | Catholic High School |

**Key Finding:** Flan-T5-large with Variant 1 generates even longer, more detailed descriptions than the base model. The larger model follows the summarisation instruction more faithfully but interprets it as producing a comprehensive summary rather than a title. With Variant 2, it achieves the **best performance across all models** (ROUGE-1: 0.8963).

### 3.4 Task C Execution Times

| Step | Time (seconds) |
|---|---|
| Data Loading | 2.71 |
| HF Dataset Build | 2.66 |
| Tokenisation | 172.70 |
| T5-small Fine-tuning | 311.26 |
| T5-small Inference | 26.67 |
| Flan-T5-base Inference | 136.00 |
| Flan-T5-large Inference | 360.78 |
| **Total Task C** | **1,090.49** |

---

<a name="overall-comparison"></a>
## 4. Overall Comparison and Analysis

### 4.1 Full Results Table

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| B1 Basic Seq2Seq | 0.3306 | 0.0755 | 0.3306 |
| B2 + GloVe | 0.3660 | 0.0884 | 0.3060 |
| B2 + HierEncoder | 0.3358 | 0.0499 | 0.2858 |
| B2 + Decoder2 | 0.3660 | 0.0822 | 0.3435 |
| B2 + Beam Search | 0.3805 | 0.0479 | 0.3405 |
| B2 All Combined | 0.2811 | 0.0497 | 0.2811 |
| T5-small (greedy) | 0.8769 | 0.6630 | 0.8752 |
| T5-small (beam-5) | 0.8810 | 0.6630 | 0.8810 |
| Flan-T5-base V1 greedy | 0.1617 | 0.0600 | 0.1581 |
| Flan-T5-base V1 beam | 0.1627 | 0.0663 | 0.1618 |
| Flan-T5-base V2 greedy | 0.7538 | 0.5055 | 0.7483 |
| Flan-T5-base V2 beam | 0.7784 | 0.5359 | 0.7784 |
| Flan-T5-large V1 greedy | 0.2364 | 0.0915 | 0.2346 |
| Flan-T5-large V1 beam | 0.2341 | 0.0896 | 0.2315 |
| **Flan-T5-large V2 greedy** | **0.8963** | **0.6932** | **0.8947** |
| Flan-T5-large V2 beam | 0.8836 | 0.6671 | 0.8836 |

### 4.2 Performance Progression

```
Performance progression (ROUGE-1):

Basic RNN   0.3306  ████████████░░░░░░░░░░░░░░░░░░░
Best RNN    0.3805  ██████████████░░░░░░░░░░░░░░░░░
T5-small    0.8810  ████████████████████████████████░
Flan-T5-L   0.8963  █████████████████████████████████
```

### 4.3 Key Findings and Discussion

**Finding 1 — The attention gap between RNN and Transformer**

The best RNN model (Beam Search, ROUGE-1: 0.3805) is outperformed by fine-tuned T5-small (ROUGE-1: 0.8810) by more than 0.50 ROUGE-1 points. This gap represents the combined effect of: (a) the attention mechanism allowing the decoder to selectively focus on relevant input tokens, (b) large-scale pretraining on diverse text corpora, and (c) subword tokenisation handling rare words that the RNN vocabulary maps to `<unk>`.

**Finding 2 — Prompt engineering outweighs model size**

```
Flan-T5-large + Variant 1 : ROUGE-1 = 0.2364 (wrong prompt)
Flan-T5-base  + Variant 2 : ROUGE-1 = 0.7538 (right prompt)
```

A model 3× smaller with the correct prompt outperforms the larger model with the wrong prompt by 3.2×. This is the most striking finding of the project and demonstrates that for instruction-tuned models, task specification is the primary performance lever.

**Finding 3 — Complexity requires proportional training**

The all-improvements-combined model (B2_all) performs worst among all RNN variants. Each individual improvement adds complexity that requires additional training to learn. With only 5 epochs, the 8.15M parameter combined model has not converged — all 5 epochs showed improving validation loss, meaning the model needed at least 10-15 more epochs to reach its potential.

**Finding 4 — Beam search does not universally improve results**

| Model | Greedy ROUGE-1 | Beam ROUGE-1 | Effect |
|---|---|---|---|
| T5-small | 0.8769 | 0.8810 | +0.0041 |
| Flan-T5-base V2 | 0.7538 | 0.7784 | +0.0246 |
| Flan-T5-large V2 | **0.8963** | 0.8836 | **-0.0127** |
| RNN B2 Beam | — | 0.3805 | best RNN |

Beam search benefits smaller or weaker models (more room for improvement) but can hurt stronger models (Flan-T5-large already generates near-optimal titles greedily). For title generation where targets are short, greedy decoding is often competitive or superior.

**Finding 5 — Fine-tuned small vs zero-shot large**

```
T5-small  (60M params, fine-tuned, 1 epoch) : ROUGE-1 = 0.8810
Flan-T5-large (780M params, zero-shot, V2)  : ROUGE-1 = 0.8963
```

The zero-shot 780M parameter model only marginally outperforms the fine-tuned 60M parameter model (+0.015 ROUGE-1). This demonstrates the power of instruction tuning: Flan-T5's training on diverse task instructions makes it highly effective without task-specific fine-tuning. The cost is inference time — Flan-T5-large takes 33.68s vs T5-small's 11.04s for 100 samples.

### 4.4 Qualitative Analysis

**Cases where RNN succeeds:**
```
Reference : list of people louisiana
B1 Pred   : list of people <unk>      ← structure correct, entity missing
T5 Pred   : List of people from Louisiana ← perfect
```

The RNN captures the overall title structure but fails on specific named entities — these become `<unk>` tokens due to vocabulary limitations.

**Cases where all models struggle:**
```
Reference : Weyburn
T5-small  : Weyburn, Saskatchewan   ← adds information
Flan-T5   : Weyburn, Saskatchewan   ← same addition
```

Both transformer models consistently add the province name even when not in the reference title. This reflects the models' tendency toward more descriptive titles learned from pretraining — single-word titles are underrepresented in their training data.

**Cases where prompt matters most:**
```
Reference : Weyburn
Flan-T5-base V1: Weyburn is a city in Saskatchewan.  ← sentence, not title
Flan-T5-base V2: Weyburn, Saskatchewan               ← correct format
```

The same model produces a qualitatively different output type based solely on the prompt — demonstrating that zero-shot performance is highly sensitive to prompt formulation.

