# =============================================================================
# train_transformer.py
# Transformer-based Title Generation — Fine-tuning and Zero-shot Prompting
#
# This script is the entry point for Part C of the NLP Assignment.
# It covers:
#
#   C1 — Fine-tuning google-t5/t5-small on the Wikipedia title generation task
#        using HuggingFace Seq2SeqTrainer
#        Evaluated with greedy decoding and beam search
#
#   C2 — Zero-shot prompting with instruction-tuned Flan-T5 models
#        (google/flan-t5-base and google/flan-t5-large)
#        Two meaningfully different prompt strategies per model
#        Evaluated with greedy decoding and beam search
#
# Why raw text for Part C?
# ─────────────────────────
#   Transformer tokenizers (BPE/SentencePiece) handle punctuation and
#   casing internally. Pre-removing punctuation or stopwords before
#   passing to a T5 tokenizer discards information the model was
#   pre-trained to use, and hurts performance.
#
# Usage
# -----
#   python train_transformer.py
#
# Prerequisites
# -------------
#   Run preprocess.py first (for the train/val/test split).
#   Raw train.csv and test.csv must exist at TRAIN_PATH and TEST_PATH.
# =============================================================================

import os
import time
import json
import inspect

import numpy as np
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict
import evaluate

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    DEVICE,
    RANDOM_STATE,
    TRAIN_PATH,
    TEST_PATH,
    VAL_SIZE,
    T5_MODEL_NAME,
    FLAN_T5_BASE,
    FLAN_T5_LARGE,
    T5_MAX_INPUT_LENGTH,
    T5_MAX_TARGET_LENGTH,
    T5_BATCH_SIZE,
    T5_LEARNING_RATE,
    T5_EPOCHS,
    T5_WEIGHT_DECAY,
    T5_GRAD_ACCUM_STEPS,
    T5_SAVE_TOTAL_LIMIT,
    T5_LOGGING_STEPS,
    T5_OUTPUT_DIR,
    T5_BEAM_WIDTH,
    T5_MAX_GEN_LEN,
    PROMPT_VARIANTS,
    RESULTS_SAVE_PATH,
)
from utils import (
    set_seed,
    Timer,
    print_section,
    load_and_split,
    logger,
)
from evaluation.metrics import (
    compute_rouge,
    print_rouge_table,
    print_single_result,
    save_results,
)


# =============================================================================
# STEP 1 — LOAD RAW DATA
# =============================================================================

def step_load_raw_data() -> tuple:
    """
    Load raw (unpreprocessed) train/val/test splits.

    Part C uses raw text intentionally — T5 tokenizers handle punctuation
    and stopwords internally. Preprocessing would remove information the
    model was pretrained to use, and hurts performance.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each has columns 'text' and 'title'.
    """
    print_section("STEP 1 — Loading Raw Data (unpreprocessed)")

    with Timer("Loading raw CSVs"):
        train_df, val_df, test_df = load_and_split(
            train_path   = TRAIN_PATH,
            test_path    = TEST_PATH,
            val_size     = VAL_SIZE,
            random_state = RANDOM_STATE,
        )

    print(f"\n  Train : {len(train_df):,} | Val : {len(val_df):,} | "
          f"Test  : {len(test_df):,}")
    print("\n  Note: Using RAW text (no preprocessing) for transformer models.")

    return train_df, val_df, test_df


# =============================================================================
# STEP 2 — BUILD HUGGINGFACE DATASET
# =============================================================================

def build_hf_dataset(
    train_df : pd.DataFrame,
    val_df   : pd.DataFrame,
    test_df  : pd.DataFrame,
) -> DatasetDict:
    """
    Convert pandas DataFrames to a HuggingFace DatasetDict.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame

    Returns
    -------
    DatasetDict with splits: "train", "validation", "test"
    """
    def df_to_hf(df):
        return Dataset.from_dict({
            "text" : df["text"].fillna("").tolist(),
            "title": df["title"].fillna("").tolist(),
        })

    return DatasetDict({
        "train"     : df_to_hf(train_df),
        "validation": df_to_hf(val_df),
        "test"      : df_to_hf(test_df),
    })


# =============================================================================
# STEP 3 — TOKENIZE DATASET FOR T5 FINE-TUNING
# =============================================================================

def tokenize_dataset(
    dataset   : DatasetDict,
    tokenizer : AutoTokenizer,
    prefix    : str = "summarize: ",
) -> DatasetDict:
    """
    Tokenize all splits for T5 fine-tuning.

    Key design decisions:
    - Input  : prepend task prefix "summarize: " (required by T5 architecture)
    - Input  : max_length = T5_MAX_INPUT_LENGTH (512) with truncation
    - Labels : max_length = T5_MAX_TARGET_LENGTH (30) with truncation
    - Labels : padding tokens replaced with -100 so CrossEntropyLoss
               ignores them — without this the model is penalised for
               correctly ignoring padding, which degrades training quality
    - Padding: done by DataCollatorForSeq2Seq during batching (not here)

    Parameters
    ----------
    dataset   : DatasetDict
    tokenizer : AutoTokenizer
    prefix    : str   Task prefix prepended to each input article

    Returns
    -------
    DatasetDict with tokenized splits
    """

    def preprocess_function(examples):
        # Prepend task prefix — required by T5 pre-training objective
        inputs  = [prefix + str(text) for text in examples["text"]]
        targets = [str(title) for title in examples["title"]]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length = T5_MAX_INPUT_LENGTH,
            truncation = True,
            padding    = False,    # DataCollator handles padding per batch
        )

        # Tokenize targets (labels)
        labels = tokenizer(
            text_target = targets,
            max_length  = T5_MAX_TARGET_LENGTH,
            truncation  = True,
            padding     = False,
        )

        # Replace padding token ids with -100 so loss ignores them
        label_ids = labels["input_ids"]
        label_ids = [
            [
                -100 if token == tokenizer.pad_token_id else token
                for token in label
            ]
            for label in label_ids
        ]

        model_inputs["labels"] = label_ids
        return model_inputs

    print_section("STEP 2 — Tokenizing Dataset")

    with Timer("Tokenization"):
        tokenized = dataset.map(
            preprocess_function,
            batched        = True,
            remove_columns = ["text", "title"],
            desc           = "Tokenizing",
        )

    print(f"  Train samples : {len(tokenized['train']):,}")
    print(f"  Val   samples : {len(tokenized['validation']):,}")

    return tokenized


# =============================================================================
# STEP 4 — COMPUTE METRICS FOR TRAINER
# =============================================================================

def make_compute_metrics(tokenizer: AutoTokenizer):
    """
    Create a compute_metrics function for Seq2SeqTrainer.

    Enables per-epoch ROUGE evaluation during training so we can monitor
    whether the model is improving on the generative task, not just
    reducing cross-entropy loss.

    Parameters
    ----------
    tokenizer : AutoTokenizer

    Returns
    -------
    callable : compute_metrics(eval_pred) -> dict
    """
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace -100 in labels before decoding
        labels = np.where(
            labels != -100, labels, tokenizer.pad_token_id
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result = rouge.compute(
            predictions = decoded_preds,
            references  = decoded_labels,
            use_stemmer = True,
        )

        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


# =============================================================================
# STEP 5 — FINE-TUNE T5-SMALL
# =============================================================================

def step_finetune_t5(
    tokenized_dataset : DatasetDict,
) -> tuple:
    """
    Fine-tune google-t5/t5-small on the title generation task
    using HuggingFace Seq2SeqTrainer.

    Parameters
    ----------
    tokenized_dataset : DatasetDict   Tokenized train/val splits

    Returns
    -------
    model     : fine-tuned T5 model
    tokenizer : T5 tokenizer
    """
    print_section("STEP 3 — Fine-tuning T5-small")
    print(f"  Model         : {T5_MODEL_NAME}")
    print(f"  Epochs        : {T5_EPOCHS}")
    print(f"  Batch size    : {T5_BATCH_SIZE}")
    print(f"  Learning rate : {T5_LEARNING_RATE}")
    print(f"  Grad accum    : {T5_GRAD_ACCUM_STEPS} steps "
          f"(effective batch = {T5_BATCH_SIZE * T5_GRAD_ACCUM_STEPS})")
    print(f"  Device        : {DEVICE}")

    # ── Load model and tokenizer ───────────────────────────────────────────────
    with Timer("Loading T5-small"):
        tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
        model     = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
        model     = model.to(DEVICE)

    # ── Data collator ─────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        padding            = True,
        label_pad_token_id = -100,
        pad_to_multiple_of = 8 if torch.cuda.is_available() else None,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_arg_kwargs = dict(
        output_dir                  = T5_OUTPUT_DIR,
        save_strategy               = "epoch",
        learning_rate               = T5_LEARNING_RATE,
        per_device_train_batch_size = T5_BATCH_SIZE,
        per_device_eval_batch_size  = T5_BATCH_SIZE,
        num_train_epochs            = T5_EPOCHS,
        weight_decay                = T5_WEIGHT_DECAY,
        gradient_accumulation_steps = T5_GRAD_ACCUM_STEPS,
        predict_with_generate       = True,
        generation_max_length       = T5_MAX_GEN_LEN,
        load_best_model_at_end      = True,
        metric_for_best_model       = "rouge1",
        greater_is_better           = True,
        save_total_limit            = T5_SAVE_TOTAL_LIMIT,
        logging_steps               = T5_LOGGING_STEPS,
        logging_dir                 = os.path.join(T5_OUTPUT_DIR, "logs"),
        fp16                        = torch.cuda.is_available(),
        seed                        = RANDOM_STATE,
        report_to                   = "none",
    )
    training_arg_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_arg_kwargs[training_arg_name] = "epoch"
    training_args = Seq2SeqTrainingArguments(**training_arg_kwargs)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized_dataset["train"],
        eval_dataset    = tokenized_dataset["validation"],
        data_collator   = data_collator,
        compute_metrics = make_compute_metrics(tokenizer),
    )
    trainer_tokenizer_arg = (
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[trainer_tokenizer_arg] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # ── Train ─────────────────────────────────────────────────────────────────
    with Timer("T5-small fine-tuning"):
        trainer.train()

    # ── Save best model ────────────────────────────────────────────────────────
    trainer.save_model(T5_OUTPUT_DIR)
    tokenizer.save_pretrained(T5_OUTPUT_DIR)
    print(f"\n  Best model saved to: {T5_OUTPUT_DIR}")

    return model, tokenizer


# =============================================================================
# STEP 6 — EVALUATE FINE-TUNED T5
# =============================================================================

def generate_t5_predictions(
    model     : AutoModelForSeq2SeqLM,
    tokenizer : AutoTokenizer,
    test_df   : pd.DataFrame,
    num_beams : int = 1,
    prefix    : str = "summarize: ",
) -> tuple:
    """
    Generate titles for all test articles using fine-tuned T5.

    Parameters
    ----------
    model     : fine-tuned T5 model
    tokenizer : T5 tokenizer
    test_df   : pd.DataFrame   Raw test data
    num_beams : int   1 = greedy, >1 = beam search
    prefix    : str   Task prefix

    Returns
    -------
    predictions : list[str]
    references  : list[str]
    """
    model.eval()
    predictions = []
    references  = test_df["title"].fillna("").tolist()

    with torch.no_grad():
        for idx, text in enumerate(test_df["text"].tolist()):

            inputs = tokenizer(
                prefix + str(text),
                return_tensors = "pt",
                max_length     = T5_MAX_INPUT_LENGTH,
                truncation     = True,
            ).to(DEVICE)

            output_ids = model.generate(
                **inputs,
                max_new_tokens = T5_MAX_GEN_LEN,
                num_beams      = num_beams,
                early_stopping = True if num_beams > 1 else False,
                length_penalty = 1.0,
            )

            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(pred.strip())

            if (idx + 1) % 10 == 0:
                print(f"  Generated {idx + 1} / {len(test_df)} titles")

    return predictions, references


def step_evaluate_t5(
    model     : AutoModelForSeq2SeqLM,
    tokenizer : AutoTokenizer,
    test_df   : pd.DataFrame,
) -> dict:
    """
    Evaluate fine-tuned T5-small with greedy and beam search.

    Parameters
    ----------
    model     : fine-tuned T5 model
    tokenizer : T5 tokenizer
    test_df   : pd.DataFrame

    Returns
    -------
    dict : ROUGE scores for greedy and beam variants
    """
    print_section("STEP 4 — Evaluating Fine-tuned T5-small")

    results = {}

    # ── Greedy ────────────────────────────────────────────────────────────────
    print("\n  Greedy decoding (num_beams=1)...")
    start = time.time()
    preds_greedy, refs = generate_t5_predictions(
        model, tokenizer, test_df, num_beams=1
    )
    print(f"  Inference time (greedy): {time.time() - start:.2f} seconds")

    scores_greedy = compute_rouge(preds_greedy, refs)
    print_single_result("T5-small (greedy)", scores_greedy)
    results["T5_small_greedy"] = scores_greedy

    # Sample predictions
    print("\n  Sample predictions (greedy):")
    for i in range(min(3, len(preds_greedy))):
        print(f"    [{i+1}] Ref  : {refs[i]}")
        print(f"          Pred : {preds_greedy[i]}")

    # ── Beam Search ───────────────────────────────────────────────────────────
    print(f"\n  Beam search (num_beams={T5_BEAM_WIDTH})...")
    start = time.time()
    preds_beam, refs = generate_t5_predictions(
        model, tokenizer, test_df, num_beams=T5_BEAM_WIDTH
    )
    print(f"  Inference time (beam): {time.time() - start:.2f} seconds")

    scores_beam = compute_rouge(preds_beam, refs)
    print_single_result("T5-small (beam)", scores_beam)
    results["T5_small_beam"] = scores_beam

    # Sample predictions
    print("\n  Sample predictions (beam):")
    for i in range(min(3, len(preds_beam))):
        print(f"    [{i+1}] Ref  : {refs[i]}")
        print(f"          Pred : {preds_beam[i]}")

    return results


# =============================================================================
# STEP 7 — ZERO-SHOT FLAN-T5 EVALUATION
# =============================================================================

def step_evaluate_flan_t5(test_df: pd.DataFrame) -> dict:
    """
    Evaluate Flan-T5-base and Flan-T5-large in zero-shot mode.

    Two prompt variants tested per model:

    Variant 1 — Direct instruction (task-prefix style):
        "summarize: <article>"
        Matches T5 pre-training convention. Simple and reliable.

    Variant 2 — Few-shot with one example:
        Shows the model the expected format before the actual input.
        Typically helps smaller models (base) more than larger ones.

    Both greedy and beam search evaluated for each combination.

    Parameters
    ----------
    test_df : pd.DataFrame   Raw test data

    Returns
    -------
    dict : ROUGE scores keyed by "modelkey_promptvariant_decoding"
    """
    print_section("STEP 5 — Zero-shot Flan-T5 Evaluation")

    flan_models = {
        "flan_t5_base" : FLAN_T5_BASE,
        "flan_t5_large": FLAN_T5_LARGE,
    }

    references  = test_df["title"].fillna("").tolist()
    all_results = {}

    for model_key, model_name in flan_models.items():

        print(f"\n{'─'*60}")
        print(f"  Loading: {model_name}")
        print(f"{'─'*60}")

        with Timer(f"Loading {model_key}"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model     = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            model     = model.to(DEVICE)
            model.eval()

        for prompt_key, prompt_template in PROMPT_VARIANTS.items():

            print(f"\n  Prompt variant : {prompt_key}")
            print(f"  Template       : {prompt_template[:70].strip()}...")

            for decode_label, num_beams in [("greedy", 1),
                                             ("beam",   T5_BEAM_WIDTH)]:

                result_key = f"{model_key}_{prompt_key}_{decode_label}"
                print(f"\n  Running: {result_key}")

                start       = time.time()
                predictions = []

                with torch.no_grad():
                    for idx, text in enumerate(test_df["text"].tolist()):

                        # Format the prompt with article text
                        input_text = prompt_template.format(str(text))

                        inputs = tokenizer(
                            input_text,
                            return_tensors = "pt",
                            max_length     = T5_MAX_INPUT_LENGTH,
                            truncation     = True,
                        ).to(DEVICE)

                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens = T5_MAX_GEN_LEN,
                            num_beams      = num_beams,
                            early_stopping = True if num_beams > 1 else False,
                            length_penalty = 1.0,
                        )

                        pred = tokenizer.decode(
                            output_ids[0], skip_special_tokens=True
                        )
                        predictions.append(pred.strip())

                        if (idx + 1) % 10 == 0:
                            print(f"    Generated {idx+1} / {len(test_df)}")

                elapsed = time.time() - start
                print(f"  Inference time : {elapsed:.2f} seconds")

                # Compute ROUGE
                scores = compute_rouge(predictions, references)
                print_single_result(result_key, scores)

                # Show sample predictions
                print(f"\n  Samples ({result_key}):")
                for i in range(min(2, len(predictions))):
                    print(f"    [{i+1}] Ref  : {references[i]}")
                    print(f"          Pred : {predictions[i]}")

                all_results[result_key] = scores

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()
        print(f"\n  GPU memory freed after {model_key}")

    return all_results


# =============================================================================
# STEP 8 — PRINT FULL COMPARISON TABLE
# =============================================================================

def step_print_full_table(
    t5_results   : dict,
    flan_results : dict,
) -> None:
    """
    Print a unified ROUGE comparison table for all Part C models.

    Parameters
    ----------
    t5_results   : dict   Results from fine-tuned T5-small
    flan_results : dict   Results from zero-shot Flan-T5 models
    """
    print_section("PART C — FULL RESULTS COMPARISON")
    print_rouge_table({**t5_results, **flan_results})

    print("\n  Key observations to discuss in report:")
    print("  1. Does fine-tuned T5-small outperform zero-shot Flan-T5-base?")
    print("  2. Does Flan-T5-large outperform Flan-T5-base zero-shot?")
    print("  3. Which prompt variant performs better and why?")
    print("  4. Does beam search consistently improve over greedy?")
    print("  5. How do transformer results compare to best RNN variant?")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run the full Task C pipeline.

    Steps
    -----
    1. Load raw (unpreprocessed) data
    2. Build HuggingFace DatasetDict
    3. Tokenize for T5 fine-tuning
    4. Fine-tune T5-small with Seq2SeqTrainer
    5. Evaluate fine-tuned T5 — greedy and beam
    6. Evaluate Flan-T5-base and Flan-T5-large — zero-shot, 2 prompts each
    7. Print full comparison table
    8. Save all results to JSON
    """
    overall_start = time.time()
    timings       = {}
    all_results   = {}

    # Reproducibility
    set_seed(RANDOM_STATE)

    # ── Step 1 : Load raw data ─────────────────────────────────────────────────
    t0 = time.time()
    train_df, val_df, test_df = step_load_raw_data()
    timings["Data Loading"] = time.time() - t0

    # ── Step 2 : Build HuggingFace DatasetDict ────────────────────────────────
    t0 = time.time()
    hf_dataset = build_hf_dataset(train_df, val_df, test_df)
    timings["HF Dataset Build"] = time.time() - t0

    # ── Step 3 : Tokenize ─────────────────────────────────────────────────────
    t0        = time.time()
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    tokenized = tokenize_dataset(hf_dataset, tokenizer)
    timings["Tokenization"] = time.time() - t0

    # ── Step 4 : Fine-tune T5-small ───────────────────────────────────────────
    t0 = time.time()
    t5_model, t5_tokenizer = step_finetune_t5(tokenized)
    timings["T5 Fine-tuning"] = time.time() - t0

    # ── Step 5 : Evaluate fine-tuned T5 ──────────────────────────────────────
    t0 = time.time()
    t5_results = step_evaluate_t5(t5_model, t5_tokenizer, test_df)
    timings["T5 Inference"] = time.time() - t0
    all_results.update(t5_results)

    # Free T5 GPU memory before loading Flan-T5
    del t5_model
    torch.cuda.empty_cache()

    # ── Step 6 : Evaluate Flan-T5 zero-shot ──────────────────────────────────
    t0 = time.time()
    flan_results = step_evaluate_flan_t5(test_df)
    timings["Flan-T5 Inference"] = time.time() - t0
    all_results.update(flan_results)

    # ── Step 7 : Print full table ─────────────────────────────────────────────
    t0 = time.time()
    step_print_full_table(t5_results, flan_results)
    timings["Results Table"] = time.time() - t0

    # ── Step 8 : Save results (merge with taskB results if they exist) ────────
    existing = {}
    if os.path.exists(RESULTS_SAVE_PATH):
        with open(RESULTS_SAVE_PATH, "r") as f:
            existing = json.load(f)

    t0 = time.time()
    existing.update(all_results)
    save_results(existing, RESULTS_SAVE_PATH)
    timings["Saving Results"] = time.time() - t0

    # ── Timing summary ────────────────────────────────────────────────────────
    print_section("TASK C — EXECUTION SUMMARY")
    print(f"\n  {'Step':<30} {'Time (seconds)':>15}")
    print(f"  {'-'*47}")
    total = 0.0
    for step, elapsed in timings.items():
        print(f"  {step:<30} {elapsed:>15.2f}")
        total += elapsed
    print(f"  {'-'*47}")
    print(f"  {'TOTAL':<30} {total:>15.2f}")

    print(f"\nTask C complete. Total time: {total:.2f} seconds")
    print(f"All results saved to : {RESULTS_SAVE_PATH}\n")


if __name__ == "__main__":
    main()
