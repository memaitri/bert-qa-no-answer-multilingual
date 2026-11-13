"""
train_qa.py
-----------
Fine-tune a multilingual (or any HuggingFace) model on SQuAD 2.0 for
extractive QA with no-answer detection.

WHY XLM-R?
RoBERTa was trained only on English text, so it cannot process Hindi.
XLM-RoBERTa (xlm-roberta-base) was trained on 100 languages and uses a shared
vocabulary, enabling zero-shot cross-lingual transfer: fine-tuned on English
SQuAD 2.0, it can answer questions in Hindi without ever seeing Hindi QA data.

UNANSWERABLE EXAMPLES — why we point them at CLS (index 0)
When a question is unanswerable the gold start/end positions are both 0.
In SQuAD 2.0 "is_impossible=True" examples the answer text is empty, so there
is no token to point at.  During training the model learns: "if start=0 and
end=0, the [CLS] token is the answer" i.e. there is no answer.  At inference
we compare start_logit[0]+end_logit[0] (the null score) against the best real
span score, and abstain if the null score wins.

Run:
    # Quick pipeline smoke test (200 examples, 1 epoch, ~2 min on CPU):
    python train_qa.py --smoke

    # Colab T4 — full run (30k examples, 2 epochs, ~1-2 hr):
    python train_qa.py --batch_size 16 --grad_accum 2 --epochs 2

    # Smaller model for faster Colab run:
    python train_qa.py --model google/rembert --epochs 2

    # Resume or evaluate a saved checkpoint:
    QA_MODEL=./checkpoints/xlmr_squad2 python evaluate_squad.py 200
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",       default="xlm-roberta-base",
                   help="HuggingFace model name or local path (default: xlm-roberta-base)")
    p.add_argument("--train_samples", type=int, default=30_000,
                   help="number of SQuAD 2.0 train examples to use (default: 30000)")
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--max_len",     type=int,   default=384)
    p.add_argument("--stride",      type=int,   default=128)
    p.add_argument("--batch_size",  type=int,   default=8,
                   help="per-device train batch size (default 8; use 16 on T4)")
    p.add_argument("--grad_accum",  type=int,   default=4,
                   help="gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--output_dir",  default=os.path.join(HERE, "checkpoints", "xlmr_squad2"))
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--smoke",       action="store_true",
                   help="tiny smoke test: 200 train + 50 eval examples, 1 epoch")
    return p.parse_args()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_squad(examples, tokenizer, max_len, stride):
    """
    Tokenize SQuAD 2.0 examples and map character-level answer positions to
    token-level start/end indices.

    For UNANSWERABLE questions (is_impossible=True or empty answers):
        start_positions = end_positions = 0  (the [CLS] / <s> token).
        Rationale: the model is trained to point at token 0 when there is no
        answer; at inference we detect this by comparing null_score (sum of
        logits at token 0) against the best real span score.

    For ANSWERABLE questions:
        We use tokenizer's offset_mapping to find which tokens cover the gold
        answer character span, clamped to the context window.  If the gold
        span is completely outside the current window (can happen with stride
        windowing), we fall back to CLS (0, 0) for that window.
    """
    questions = [q.strip() for q in examples["question"]]
    contexts  = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_len,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map    = tokenized.pop("overflow_to_sample_mapping")
    offset_map    = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_map):
        sample_idx = sample_map[i]
        answers    = examples["answers"][sample_idx]
        seq_ids    = tokenized.sequence_ids(i)

        # CLS fallback for unanswerable or when the answer window doesn't cover gold
        cls_idx = 0

        if not answers["answer_start"] or answers["text"] == []:
            # Unanswerable: target the CLS token (index 0)
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        ans_start_char = answers["answer_start"][0]
        ans_end_char   = ans_start_char + len(answers["text"][0])

        # Find the token range that belongs to the context (sequence_id == 1)
        ctx_start = next((j for j, s in enumerate(seq_ids) if s == 1), None)
        ctx_end   = next((j for j in range(len(seq_ids)-1, -1, -1) if seq_ids[j] == 1), None)

        if ctx_start is None or ctx_end is None:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        # If the answer is outside this window → CLS fallback
        window_start_char = offsets[ctx_start][0]
        window_end_char   = offsets[ctx_end][1]
        if ans_start_char < window_start_char or ans_end_char > window_end_char:
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
            continue

        # Map character positions to token positions
        tok_start = ctx_start
        for j in range(ctx_start, ctx_end + 1):
            if offsets[j][0] <= ans_start_char:
                tok_start = j
        tok_end = ctx_end
        for j in range(ctx_end, ctx_start - 1, -1):
            if offsets[j][1] >= ans_end_char:
                tok_end = j

        start_positions.append(tok_start)
        end_positions.append(tok_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized


# ── Loss curve ────────────────────────────────────────────────────────────────

def save_loss_curve(log_history, out_path):
    """Save a training loss PNG from the Trainer log_history."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps  = [e["step"]       for e in log_history if "loss" in e]
        losses = [e["loss"]       for e in log_history if "loss" in e]
        if not steps:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, losses, linewidth=1.5, color="#2196F3")
        ax.set_xlabel("Step"); ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Curve"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Loss curve saved → {out_path}")
    except Exception as e:
        print(f"  (could not save loss curve: {e})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Smoke-test overrides
    if args.smoke:
        args.train_samples = 200
        args.epochs        = 1
        args.batch_size    = 4
        args.grad_accum    = 1
        print("── SMOKE TEST MODE (200 train, 1 epoch) ──")

    # Suppress TensorFlow import in transformers — we only use PyTorch.
    # On Windows, transformers' TF integration triggers Keras 3 compatibility
    # issues even when USE_TF=0 is set. Setting both env vars prevents the crash.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import torch
    from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                               TrainingArguments, Trainer, default_data_collator)
    from datasets import load_dataset

    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model     = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading SQuAD 2.0 ...")
    raw = load_dataset("rajpurkar/squad_v2")

    n_train = min(args.train_samples, len(raw["train"]))
    n_eval  = min(500 if not args.smoke else 50, len(raw["validation"]))

    rng = raw["train"].shuffle(seed=args.seed)
    train_ds = rng.select(range(n_train))
    eval_ds  = raw["validation"].shuffle(seed=args.seed).select(range(n_eval))

    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    # ── Tokenize ──────────────────────────────────────────────────────────────
    print("Tokenizing ...")
    tok_fn = lambda ex: preprocess_squad(ex, tokenizer, args.max_len, args.stride)
    remove_cols = [c for c in train_ds.column_names
                   if c not in ("input_ids", "attention_mask", "token_type_ids",
                                "start_positions", "end_positions")]

    train_tok = train_ds.map(tok_fn, batched=True,
                              remove_columns=train_ds.column_names,
                              desc="Tokenizing train")
    eval_tok  = eval_ds.map(tok_fn,  batched=True,
                              remove_columns=eval_ds.column_names,
                              desc="Tokenizing eval")

    # ── Training arguments ────────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        fp16                        = use_fp16,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        logging_steps               = 50,
        seed                        = args.seed,
        report_to                   = "none",   # no W&B / MLflow
        dataloader_num_workers      = 0,        # safer on Windows / Colab
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_tok,
        eval_dataset    = eval_tok,
        data_collator   = default_data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training on {len(train_tok)} windows "
          f"(fp16={'on' if use_fp16 else 'off (CPU)'}) ...")
    trainer.train()

    # ── Save model + tokenizer ────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved → {args.output_dir}")

    # ── Save loss curve ───────────────────────────────────────────────────────
    save_loss_curve(
        trainer.state.log_history,
        os.path.join(args.output_dir, "loss_curve.png")
    )

    # ── Save config record ────────────────────────────────────────────────────
    config_record = {
        "model":          args.model,
        "train_samples":  n_train,
        "eval_samples":   n_eval,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "max_len":        args.max_len,
        "stride":         args.stride,
        "batch_size":     args.batch_size,
        "grad_accum":     args.grad_accum,
        "fp16":           use_fp16,
        "seed":           args.seed,
        "smoke":          args.smoke,
    }
    cfg_path = os.path.join(args.output_dir, "train_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_record, f, indent=2)
    print(f"Config saved → {cfg_path}")
    print("\nDone! To evaluate:")
    print(f"  QA_MODEL={args.output_dir} python evaluate_squad.py 200")
    print(f"  python evaluate_multilingual.py --models {args.output_dir}")


if __name__ == "__main__":
    main()
