"""
run_cl_experiment.py
───────────────────────────────────────────────────────────────────────────────
Short but complete Continual Learning comparison:
  1. Baseline   — frozen data2vec, no adaptation
  2. Fine-tune  — gradient updates on Task B (upper layers only)
  3. RandOpt    — gradient-free perturbation on Task B (upper layers only)

All three are evaluated on both Task A and Task B before and after adaptation.
The key metric is Backward Transfer (BWT) — catastrophic forgetting on Task A.

Logging: wandb (project = "CL_experiment")
RunAI:   submit via `runai submit --image ... -- python run_cl_experiment.py`

Quick run (smoke-test scale):
    python run_cl_experiment.py --n_samples 10 --n_steps 5 --n_candidates 10

Full short run (recommended):
    python run_cl_experiment.py --n_samples 30 --n_steps 20 --n_candidates 20

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW
───────────────────────────────────────────────────────────────────────────────

  Audio waveform  [raw float32, 16 kHz]
       ↓
  CNN feature extractor  (7 conv layers, frozen — never touched)
       ↓  (B, T', 512)
  Transformer encoder    (12 layers, 768-dim hidden)
       ↓  (B, T', 768)
  CTC head               (Linear 768 → vocab_size=32, random init)
       ↓  (B, T', 32)
  Greedy CTC decode      → predicted text → WER

What a CTC head does:
  - Maps each encoder frame (768-dim) to a probability distribution over
    characters (a-z, space, blank).
  - CTC loss handles misalignment between audio frames (~150) and text (~10
    chars) by marginalizing over all valid alignments.
  - Here it is RANDOMLY INITIALIZED — we only care that WER *changes* between
    encoder states, not about absolute accuracy.

───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import copy
import time
import torch
import torch.nn as nn
import wandb
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer as compute_wer

# ── Inline RandOpt (no import dependency, self-contained) ────────────────────
import random

ENCODER_NAME   = "facebook/data2vec-audio-base"
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"
HIDDEN_DIM     = 768
WANDB_PROJECT  = "CL_experiment"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CL experiment: baseline vs fine-tune vs RandOpt")
    p.add_argument("--n_samples",      type=int,   default=20,
                   help="Utterances per task")
    p.add_argument("--n_steps",        type=int,   default=10,
                   help="RandOpt adaptation steps")
    p.add_argument("--n_candidates",   type=int,   default=20,
                   help="Noise candidates per RandOpt step")
    p.add_argument("--sigma",          type=float, default=1e-3,
                   help="Initial noise std for RandOpt")
    p.add_argument("--layers_from",    type=int,   default=8,
                   help="First encoder layer to perturb (0-based)")
    p.add_argument("--finetune_lr",    type=float, default=1e-5,
                   help="LR for naive gradient fine-tuning")
    p.add_argument("--finetune_steps", type=int,   default=30,
                   help="Gradient update steps for fine-tuning")
    p.add_argument("--run_name",       type=str,   default=None,
                   help="wandb run name (auto-generated if omitted)")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_samples(config: str, n: int, split: str = "validation"):
    print(f"  Loading librispeech_asr [{config}] {split} …")
    ds = load_dataset(
        "librispeech_asr", config,
        split=split,
        streaming=False,
        trust_remote_code=True,
    )
    return list(ds.take(n))


def to_inputs(samples, processor, device):
    audio = [s["audio"]["array"] for s in samples]
    refs  = [s["text"].lower().strip() for s in samples]
    iv    = processor(
        audio, sampling_rate=16_000,
        return_tensors="pt", padding=True,
    ).input_values.to(device)
    return iv, refs


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_wer(state_dict, input_values, refs, processor, ctc_head, device):
    """Load state_dict into a temp encoder and return WER. No gradients."""
    tmp = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    with torch.no_grad():
        hidden   = tmp(input_values).last_hidden_state        # (B, T, 768)
        logits   = ctc_head(hidden)                            # (B, T, vocab)
        pred_ids = torch.argmax(logits, dim=-1)
        preds    = [p.lower().strip()
                    for p in processor.batch_decode(pred_ids)]
    return compute_wer(refs, preds)


# ─────────────────────────────────────────────────────────────────────────────
# RandOpt (gradient-free adaptation)
# ─────────────────────────────────────────────────────────────────────────────
def get_perturbable(encoder, layers_from):
    return [
        n for n, _ in encoder.named_parameters()
        if "encoder.layers" in n
        and int(n.split("encoder.layers.")[1].split(".")[0]) >= layers_from
    ]


def perturb(state, names, sigma, seed):
    torch.manual_seed(seed)
    noisy = copy.deepcopy(state)
    for n in names:
        noisy[n] = noisy[n] + torch.randn_like(noisy[n]) * sigma
    return noisy


def run_randopt(base_state, perturbable_names, score_fn,
                n_steps, n_candidates, sigma, sigma_decay=0.5, min_sigma=1e-5):
    """
    Gradient-free adaptation loop.
    Returns (final_state, history_list)
    Each history entry: {step, sigma, base_score, best_score, accepted, delta}
    """
    state   = copy.deepcopy(base_state)
    history = []

    for step in range(n_steps):
        base_score  = score_fn(state)
        best_score  = base_score
        best_state  = None

        for k in range(n_candidates):
            seed      = step * n_candidates + k
            candidate = perturb(state, perturbable_names, sigma, seed)
            s         = score_fn(candidate)
            if s < best_score:
                best_score = s
                best_state = candidate

        accepted = best_state is not None
        if accepted:
            state = best_state
        elif sigma > min_sigma:
            sigma *= sigma_decay

        entry = {
            "step":       step + 1,
            "sigma":      sigma,
            "base_score": base_score,
            "best_score": best_score,
            "delta":      base_score - best_score,   # positive = improvement
            "accepted":   accepted,
        }
        history.append(entry)

        status = "✓" if accepted else "✗"
        print(f"  [RandOpt] step {step+1:3d}/{n_steps} {status} "
              f"σ={sigma:.2e}  "
              f"WER {base_score:.4f} → {best_score:.4f}  "
              f"Δ={base_score - best_score:+.4f}")

    return state, history


# ─────────────────────────────────────────────────────────────────────────────
# Naive fine-tuning (gradient-based, upper layers only)
# ─────────────────────────────────────────────────────────────────────────────
def run_finetune(base_state, encoder_ref, ctc_head, iv_b, refs_b,
                 processor, device, lr, steps, layers_from):
    """
    Gradient fine-tuning on Task B.
    Only upper layers (>= layers_from) are unfrozen — same scope as RandOpt.
    Uses CTC loss.
    Returns (final_state, loss_history)
    """
    enc = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    enc.load_state_dict(copy.deepcopy(base_state))

    for name, param in enc.named_parameters():
        if "encoder.layers" in name:
            layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
            param.requires_grad_(layer_idx >= layers_from)
        else:
            param.requires_grad_(False)

    ctc_ft    = copy.deepcopy(ctc_head)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, enc.parameters()), lr=lr
    )
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    enc.train()
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        hidden    = enc(iv_b).last_hidden_state              # (B, T, 768)
        logits    = ctc_ft(hidden)                            # (B, T, V)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        target_ids = [
            torch.tensor(processor.tokenizer.encode(r), dtype=torch.long)
            for r in refs_b
        ]
        targets        = torch.cat(target_ids)
        input_lengths  = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in target_ids], dtype=torch.long)

        loss = ctc_loss_fn(
            log_probs.cpu(), targets.cpu(),
            input_lengths.cpu(), target_lengths.cpu()
        )
        loss.backward()
        optimizer.step()

        l = loss.item()
        loss_history.append(l)
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  [Finetune] step {step+1:3d}/{steps}  CTC loss={l:.4f}")

    enc.eval()
    return enc.state_dict(), loss_history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = args.device
    t0     = time.time()

    run_name = args.run_name or (
        f"n{args.n_samples}_steps{args.n_steps}_σ{args.sigma}_L{args.layers_from}"
    )

    # ── wandb init ──────────────────────────────────────────────────────────
    wandb.init(
        project = WANDB_PROJECT,
        name    = run_name,
        config  = vars(args),
    )
    print(f"\n[wandb] Run: {wandb.run.get_url()}\n")

    # ── Model setup ──────────────────────────────────────────────────────────
    print("Loading processor …")
    processor  = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    vocab_size = processor.tokenizer.vocab_size

    # Shared random CTC head (same weights for all three methods → fair comparison)
    torch.manual_seed(42)
    ctc_head = nn.Linear(HIDDEN_DIM, vocab_size).to(device)
    nn.init.xavier_uniform_(ctc_head.weight)
    ctc_head.eval()

    print("Loading pretrained data2vec encoder …")
    encoder   = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    encoder.eval()
    base_state = copy.deepcopy(encoder.state_dict())

    perturbable = get_perturbable(encoder, args.layers_from)
    print(f"Perturbable tensors: {len(perturbable)} "
          f"(layers {args.layers_from}–11)\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    samples_a = load_samples("clean", args.n_samples)   # Task A: clean speech
    samples_b = load_samples("other", args.n_samples)   # Task B: noisy/other

    iv_a, refs_a = to_inputs(samples_a, processor, device)
    iv_b, refs_b = to_inputs(samples_b, processor, device)

    # ── Pre-adaptation WER (all methods share same starting point) ───────────
    print("\n── PRE-ADAPTATION ───────────────────────────────────────────────")
    wer_a_pre = score_wer(base_state, iv_a, refs_a, processor, ctc_head, device)
    wer_b_pre = score_wer(base_state, iv_b, refs_b, processor, ctc_head, device)
    print(f"  Task A (clean) WER  = {wer_a_pre:.4f}")
    print(f"  Task B (other) WER  = {wer_b_pre:.4f}")
    wandb.log({"pre/task_a_wer": wer_a_pre, "pre/task_b_wer": wer_b_pre})

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 1: RandOpt
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── RANDOPT adaptation on Task B ─────────────────────────────────")

    def score_b_randopt(state):
        return score_wer(state, iv_b, refs_b, processor, ctc_head, device)

    t_ro = time.time()
    randopt_state, ro_history = run_randopt(
        base_state,
        perturbable,
        score_b_randopt,
        n_steps      = args.n_steps,
        n_candidates = args.n_candidates,
        sigma        = args.sigma,
    )

    # Log per-step RandOpt metrics to wandb
    for h in ro_history:
        wandb.log({
            "randopt/step":       h["step"],
            "randopt/sigma":      h["sigma"],
            "randopt/taskB_wer":  h["best_score"],
            "randopt/delta_wer":  h["delta"],
            "randopt/accepted":   int(h["accepted"]),
        })

    wer_a_ro = score_wer(randopt_state, iv_a, refs_a, processor, ctc_head, device)
    wer_b_ro = score_wer(randopt_state, iv_b, refs_b, processor, ctc_head, device)
    bwt_ro   = wer_a_ro - wer_a_pre
    fwt_ro   = wer_b_pre - wer_b_ro     # positive = improvement on B
    acceptance_density = sum(h["accepted"] for h in ro_history) / len(ro_history)

    wandb.log({
        "randopt/final_task_a_wer":      wer_a_ro,
        "randopt/final_task_b_wer":      wer_b_ro,
        "randopt/bwt":                   bwt_ro,
        "randopt/fwt":                   fwt_ro,
        "randopt/acceptance_density":    acceptance_density,
        "randopt/time_s":                time.time() - t_ro,
    })
    print(f"\n  [RandOpt] Task A WER after = {wer_a_ro:.4f}  (BWT={bwt_ro:+.4f})")
    print(f"  [RandOpt] Task B WER after = {wer_b_ro:.4f}  (FWT={fwt_ro:+.4f})")
    print(f"  [RandOpt] Acceptance density δ(m) = {acceptance_density:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 2: Naive fine-tuning
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── FINETUNE adaptation on Task B ────────────────────────────────")
    t_ft = time.time()
    ft_state, ft_losses = run_finetune(
        base_state, encoder, ctc_head,
        iv_b, refs_b,
        processor, device,
        lr=args.finetune_lr,
        steps=args.finetune_steps,
        layers_from=args.layers_from,
    )

    for step_i, loss_val in enumerate(ft_losses):
        wandb.log({"finetune/step": step_i + 1, "finetune/ctc_loss": loss_val})

    wer_a_ft = score_wer(ft_state, iv_a, refs_a, processor, ctc_head, device)
    wer_b_ft = score_wer(ft_state, iv_b, refs_b, processor, ctc_head, device)
    bwt_ft   = wer_a_ft - wer_a_pre
    fwt_ft   = wer_b_pre - wer_b_ft

    wandb.log({
        "finetune/final_task_a_wer": wer_a_ft,
        "finetune/final_task_b_wer": wer_b_ft,
        "finetune/bwt":              bwt_ft,
        "finetune/fwt":              fwt_ft,
        "finetune/time_s":           time.time() - t_ft,
    })
    print(f"\n  [Finetune] Task A WER after = {wer_a_ft:.4f}  (BWT={bwt_ft:+.4f})")
    print(f"  [Finetune] Task B WER after = {wer_b_ft:.4f}  (FWT={fwt_ft:+.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD 3: Baseline (frozen, no adaptation)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── FROZEN BASELINE ──────────────────────────────────────────────")
    # Task A / B WER are unchanged — same as pre-adaptation
    wandb.log({
        "baseline/task_a_wer": wer_a_pre,
        "baseline/task_b_wer": wer_b_pre,
        "baseline/bwt":        0.0,
        "baseline/fwt":        0.0,
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Final comparison table
    # ─────────────────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"{'Method':<16} {'TaskA WER':>10} {'TaskB WER':>10} "
          f"{'BWT(↓=bad)':>12} {'FWT(↑=good)':>12}")
    print("-" * 65)
    print(f"{'Frozen (no adapt)':<16} {wer_a_pre:>10.4f} {wer_b_pre:>10.4f} "
          f"{'0.0000':>12} {'0.0000':>12}")
    print(f"{'Fine-tune':<16} {wer_a_ft:>10.4f} {wer_b_ft:>10.4f} "
          f"{bwt_ft:>12.4f} {fwt_ft:>12.4f}")
    print(f"{'RandOpt':<16} {wer_a_ro:>10.4f} {wer_b_ro:>10.4f} "
          f"{bwt_ro:>12.4f} {fwt_ro:>12.4f}")
    print(f"{'='*65}")
    print(f"Total time: {total_time/60:.1f} min")

    # Log the comparison summary table as a wandb Table
    comparison_table = wandb.Table(
        columns=["Method", "TaskA_WER", "TaskB_WER", "BWT", "FWT"],
        data=[
            ["Frozen",    wer_a_pre, wer_b_pre, 0.0,    0.0],
            ["Fine-tune", wer_a_ft,  wer_b_ft,  bwt_ft, fwt_ft],
            ["RandOpt",   wer_a_ro,  wer_b_ro,  bwt_ro, fwt_ro],
        ]
    )
    wandb.log({"comparison/summary": comparison_table})

    # ─────────────────────────────────────────────────────────────────────────
    # Research direction verdict
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── RESEARCH DIRECTION VERDICT ───────────────────────────────────")
    forgetting_avoided = bwt_ro < bwt_ft
    adapted_task_b     = fwt_ro > 0 or fwt_ft > 0

    if forgetting_avoided and adapted_task_b:
        verdict = "STRONG: RandOpt adapts Task B with less forgetting than fine-tuning."
    elif forgetting_avoided:
        verdict = "PARTIAL: RandOpt avoids forgetting but Task B WER did not improve. Scale up n_steps/n_candidates."
    elif adapted_task_b:
        verdict = "MIXED: Task B improved but forgetting happened. Try smaller sigma or fewer steps."
    else:
        verdict = "NEGATIVE: Neither method improved Task B. CTC head needs supervised training first."

    print(f"  → {verdict}")
    wandb.log({"verdict": verdict,
               "randopt_forgetting_avoided": forgetting_avoided,
               "any_taskB_improvement": adapted_task_b})

    wandb.finish()
    print(f"\n[wandb] Run complete: {wandb.run.get_url() if wandb.run else 'see wandb dashboard'}")


if __name__ == "__main__":
    main()
