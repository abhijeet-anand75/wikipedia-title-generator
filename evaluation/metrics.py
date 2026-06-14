# =============================================================================
# evaluation/metrics.py
# ROUGE evaluation utilities for the Seq2Seq title generation project
#
# Contents:
#   - compute_rouge        : Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores
#   - print_rouge_table    : Pretty-print a results table to stdout
#   - save_results         : Persist all results to a JSON file
#   - generate_rnn_titles  : Run a trained RNN model over the test set
#   - evaluate_rnn         : Full evaluation pipeline for one RNN variant
#   - evaluate_all_variants: Loop over all experiment variants and evaluate
# =============================================================================

import os
import json
import time
import torch
import pandas as pd

from rouge_score import rouge_scorer as rs

from config import (
    DEVICE,
    MAX_NEW_TOKENS,
    BEAM_WIDTH,
    RESULTS_SAVE_PATH,
    EXPERIMENT_VARIANTS,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    UNK_IDX,
)


# =============================================================================
# CORE ROUGE COMPUTATION
# =============================================================================

def compute_rouge(
    predictions : list,
    references  : list,
    use_stemmer : bool = True,
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Each score is the macro-average F1 across all prediction-reference pairs.

    Parameters
    ----------
    predictions : list[str]
        Generated titles — one string per test sample.
    references  : list[str]
        Ground-truth titles — one string per test sample.
    use_stemmer : bool
        Apply Porter stemmer before matching (reduces inflection mismatches).
        Default True — consistent with standard summarisation evaluation.

    Returns
    -------
    dict with keys:
        "rouge1" : float  ROUGE-1 F1
        "rouge2" : float  ROUGE-2 F1
        "rougeL" : float  ROUGE-L F1

    Raises
    ------
    ValueError
        If predictions and references have different lengths.
    """
    # Guard against empty prediction/reference lists
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and "
            f"references ({len(references)}) must have the same length."
        )

    scorer = rs.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer = use_stemmer,
    )

    r1_scores, r2_scores, rl_scores = [], [], []

    for ref, pred in zip(references, predictions):
        # Guard against empty strings — rouge_scorer crashes on them
        if not str(pred).strip():
            pred = "<empty>"
        if not str(ref).strip():
            ref = "<empty>"

        score = scorer.score(str(ref), str(pred))
        r1_scores.append(score["rouge1"].fmeasure)
        r2_scores.append(score["rouge2"].fmeasure)
        rl_scores.append(score["rougeL"].fmeasure)

    results = {
        "rouge1" : round(sum(r1_scores) / len(r1_scores), 4),
        "rouge2" : round(sum(r2_scores) / len(r2_scores), 4),
        "rougeL" : round(sum(rl_scores) / len(rl_scores), 4),
    }

    return results


# =============================================================================
# DISPLAY
# =============================================================================

def print_rouge_table(results_dict: dict) -> None:
    """
    Pretty-print a comparison table of ROUGE scores for all model variants.

    Parameters
    ----------
    results_dict : dict
        Keys   = model variant names  (e.g. "B1_basic", "B2_glove")
        Values = dicts with keys "rouge1", "rouge2", "rougeL"

    Example output
    --------------
    ============================================================
    ROUGE EVALUATION RESULTS
    ============================================================
    Model Variant        ROUGE-1    ROUGE-2    ROUGE-L
    ------------------------------------------------------------
    B1_basic              0.1823     0.0512     0.1801
    B2_glove              0.2341     0.0734     0.2315
    ...
    ============================================================
    """
    bar      = "=" * 60
    thin_bar = "-" * 60

    print(f"\n{bar}")
    print("  ROUGE EVALUATION RESULTS")
    print(f"{bar}")
    print(f"{'Model Variant':<22} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
    print(thin_bar)

    for variant_name, scores in results_dict.items():
        r1 = scores.get("rouge1", 0.0)
        r2 = scores.get("rouge2", 0.0)
        rl = scores.get("rougeL", 0.0)
        print(f"{variant_name:<22} {r1:>10.4f} {r2:>10.4f} {rl:>10.4f}")

    print(bar)


def print_single_result(variant_name: str, scores: dict) -> None:
    """
    Print ROUGE scores for a single model variant.

    Parameters
    ----------
    variant_name : str   Human-readable model name
    scores       : dict  Keys: rouge1, rouge2, rougeL
    """
    print(f"\n── ROUGE Scores: {variant_name} ──")
    print(f"  ROUGE-1 F1 : {scores['rouge1']:.4f}")
    print(f"  ROUGE-2 F1 : {scores['rouge2']:.4f}")
    print(f"  ROUGE-L F1 : {scores['rougeL']:.4f}")


# =============================================================================
# RESULTS PERSISTENCE
# =============================================================================

def save_results(results_dict: dict, path: str = RESULTS_SAVE_PATH) -> None:
    """
    Save all evaluation results to a JSON file.

    Parameters
    ----------
    results_dict : dict   All results keyed by variant name
    path         : str    Output file path
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved → {path}")


def load_results(path: str = RESULTS_SAVE_PATH) -> dict:
    """
    Load previously saved results from a JSON file.

    Parameters
    ----------
    path : str   Path to the saved results JSON

    Returns
    -------
    dict : results keyed by variant name
    """
    with open(path, "r") as f:
        results = json.load(f)

    print(f"Results loaded ← {path}")
    return results


# =============================================================================
# RNN EVALUATION HELPERS
# =============================================================================

def generate_rnn_titles(
    model         : torch.nn.Module,
    test_loader   : torch.utils.data.DataLoader,
    vocab_itos    : dict,
    decode_method : str = "greedy",
    beam_width    : int = BEAM_WIDTH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device        : torch.device = DEVICE,
) -> tuple:
    """
    Run a trained Seq2seqRNN model over the test set and collect
    generated titles alongside ground-truth references.

    Parameters
    ----------
    model         : Seq2seqRNN   Trained model (already on device)
    test_loader   : DataLoader   Test set loader (batch_size=1)
    vocab_itos    : dict         index → token mapping
    decode_method : str          "greedy" or "beam"
    beam_width    : int          Beam width (beam search only)
    max_new_tokens: int          Max tokens to generate
    device        : torch.device

    Returns
    -------
    predictions : list[str]   Generated titles
    references  : list[str]   Ground-truth titles
    """
    model.eval()

    predictions = []
    references  = []

    with torch.no_grad():
        for batch_idx, (text, title, text_len) in enumerate(test_loader):

            text     = text.to(device)
            text_len = text_len.to(device)
            title    = title.to(device)

            # Generate title
            generated = model.generate(
                text           = text,
                text_len       = text_len,
                decode_method  = decode_method,
                beam_width     = beam_width,
                max_new_tokens = max_new_tokens,
                vocab_itos     = vocab_itos,
            )

            # Decode reference title (skip special tokens)
            ref_indices = title[0].tolist()
            ref_tokens  = [
                vocab_itos.get(idx, "<unk>")
                for idx in ref_indices
                if idx not in (PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX)
            ]
            reference = " ".join(ref_tokens)

            predictions.append(generated if isinstance(generated, str) else "")
            references.append(reference)

            # Progress indicator every 10 samples
            if (batch_idx + 1) % 10 == 0:
                print(f"  Generated {batch_idx + 1} / {len(test_loader)} titles")

    return predictions, references


def evaluate_rnn(
    model         : torch.nn.Module,
    test_loader   : torch.utils.data.DataLoader,
    vocab_itos    : dict,
    variant_name  : str,
    decode_method : str = "greedy",
    beam_width    : int = BEAM_WIDTH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device        : torch.device = DEVICE,
    show_examples : int = 5,
) -> dict:
    """
    Full evaluation pipeline for one RNN model variant.

    Generates titles for all test samples, computes ROUGE scores,
    prints a results summary, and optionally shows example predictions.

    Parameters
    ----------
    model         : Seq2seqRNN
    test_loader   : DataLoader    batch_size=1
    vocab_itos    : dict          index → token
    variant_name  : str           Label for this experiment
    decode_method : str           "greedy" or "beam"
    beam_width    : int
    max_new_tokens: int
    device        : torch.device
    show_examples : int           Number of prediction examples to print

    Returns
    -------
    dict : ROUGE scores with keys rouge1, rouge2, rougeL
    """
    print(f"\nEvaluating: {variant_name} | decode: {decode_method}")
    start = time.time()

    # Generate all predictions
    predictions, references = generate_rnn_titles(
        model          = model,
        test_loader    = test_loader,
        vocab_itos     = vocab_itos,
        decode_method  = decode_method,
        beam_width     = beam_width,
        max_new_tokens = max_new_tokens,
        device         = device,
    )

    elapsed = time.time() - start
    print(f"Inference time: {elapsed:.2f} seconds")

    # Compute ROUGE
    scores = compute_rouge(predictions, references)
    print_single_result(variant_name, scores)

    # Show example predictions
    if show_examples > 0:
        print(f"\n── Example Predictions ({variant_name}) ──")
        for i in range(min(show_examples, len(predictions))):
            print(f"  [{i+1}] Ref  : {references[i]}")
            print(f"      Pred : {predictions[i]}")
            print()

    return scores


# =============================================================================
# FULL EXPERIMENT LOOP
# =============================================================================

def evaluate_all_variants(
    trained_models : dict,
    test_loader    : torch.utils.data.DataLoader,
    vocab_itos     : dict,
    device         : torch.device = DEVICE,
) -> dict:
    """
    Evaluate all trained RNN variants and compile results into one dict.

    Parameters
    ----------
    trained_models : dict
        Keys   = variant names matching EXPERIMENT_VARIANTS in config.py
        Values = trained Seq2seqRNN model instances

    test_loader    : DataLoader   batch_size=1
    vocab_itos     : dict         index → token
    device         : torch.device

    Returns
    -------
    all_results : dict
        Keys   = variant names
        Values = dicts with rouge1, rouge2, rougeL
    """
    all_results = {}

    for variant_name, (enc_type, dec_type, use_glove, decode_method) \
            in EXPERIMENT_VARIANTS.items():

        if variant_name not in trained_models:
            print(f"  Skipping {variant_name} — model not found in trained_models dict")
            continue

        model  = trained_models[variant_name]
        scores = evaluate_rnn(
            model         = model,
            test_loader   = test_loader,
            vocab_itos    = vocab_itos,
            variant_name  = variant_name,
            decode_method = decode_method,
            device        = device,
        )
        all_results[variant_name] = scores

    # Print comparison table
    print_rouge_table(all_results)

    return all_results
