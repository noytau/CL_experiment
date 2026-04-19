"""
cl_train.py
───────────────────────────────────────────────────────────────────────────────
Continual Learning experiment: data2vec-audio + RandOpt vs. naive fine-tuning.

Experiment design (two-task sequential protocol):
  • Task A  : LibriSpeech clean   (validation split, first N_SAMPLES utterances)
  • Task B  : LibriSpeech other   (validation split, first N_SAMPLES utterances)

Two adaptation strategies are compared:
  1. Baseline (frozen encoder)  — no adaptation at all.
  2. Naive fine-tuning          — gradient update on Task B.
  3. RandOpt                    — gradient-free weight perturbation on Task B.

Catastrophic forgetting is measured as the *degradation on Task A after
adaptation on Task B* (backward transfer / BWT):
  BWT = WER_A_after - WER_A_before    (positive = forgetting)

Usage:
    python cl_train.py [--n_samples 20] [--n_steps 10] [--sigma 1e-3]
"""

import argparse
import copy
import torch
import torch.nn as nn
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer
from randopt import RandOpt, RandOptConfig

# ── Config ───────────────────────────────────────────────────────────────────
ENCODER_NAME   = "facebook/data2vec-audio-base"
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"
HIDDEN_DIM     = 768


# ── CLI args ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples",    type=int,   default=20)
    p.add_argument("--n_steps",      type=int,   default=10,
                   help="RandOpt adaptation steps")
    p.add_argument("--n_candidates", type=int,   default=20,
                   help="Perturbation candidates per step")
    p.add_argument("--sigma",        type=float, default=1e-3)
    p.add_argument("--layers_from",  type=int,   default=8)
    p.add_argument("--finetune_lr",  type=float, default=1e-5,
                   help="LR for naive gradient fine-tuning baseline")
    p.add_argument("--finetune_steps", type=int, default=20,
                   help="Gradient update steps for naive fine-tuning baseline")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────
def load_samples(config: str, n: int):
    print(f"  Loading librispeech_asr [{config}] validation …")
    ds = load_dataset(
        "librispeech_asr", config,
        split="validation",
        streaming=False,
        trust_remote_code=True,
    )
    return list(ds.take(n))


def batch_to_inputs(samples, processor, device):
    audio = [s["audio"]["array"] for s in samples]
    refs  = [s["text"].lower().strip() for s in samples]
    iv    = processor(audio, sampling_rate=16_000,
                      return_tensors="pt", padding=True).input_values.to(device)
    return iv, refs


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_wer(state_dict, input_values, refs, processor, ctc_head, device):
    """
    Load state_dict into a temporary encoder copy and return WER.
    Called by RandOpt — must accept (state_dict, *args) signature.
    """
    tmp = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    with torch.no_grad():
        hidden   = tmp(input_values).last_hidden_state
        logits   = ctc_head(hidden)
        pred_ids = torch.argmax(logits, dim=-1)
        preds    = [p.lower().strip()
                    for p in processor.batch_decode(pred_ids)]
    return wer(refs, preds)


# ── Naive fine-tuning baseline ────────────────────────────────────────────────
def finetune_encoder(encoder, ctc_head, input_values, refs,
                     processor, device, lr, steps, layers_from):
    """
    Simple gradient fine-tuning: only upper layers are updated (same scope
    as RandOpt) to make the comparison fair.
    """
    encoder_ft = copy.deepcopy(encoder)

    # Freeze lower layers, unfreeze upper layers
    for name, param in encoder_ft.named_parameters():
        if "encoder.layers" in name:
            layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
            param.requires_grad_(layer_idx >= layers_from)
        else:
            param.requires_grad_(False)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, encoder_ft.parameters()), lr=lr
    )
    ctc_head_ft = copy.deepcopy(ctc_head)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    encoder_ft.train()
    for step in range(steps):
        optimizer.zero_grad()
        hidden  = encoder_ft(input_values).last_hidden_state   # (B, T, 768)
        logits  = ctc_head_ft(hidden)                           # (B, T, V)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        # Build CTC targets from processor tokenizer
        target_ids = [
            torch.tensor(processor.tokenizer.encode(r), dtype=torch.long)
            for r in refs
        ]
        targets      = torch.cat(target_ids)
        input_lengths  = torch.full((logits.size(0),), logits.size(1),
                                    dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in target_ids],
                                      dtype=torch.long)

        loss = ctc_loss_fn(
            log_probs.cpu(),
            targets.cpu(),
            input_lengths.cpu(),
            target_lengths.cpu(),
        )
        loss.backward()
        optimizer.step()

        if (step + 1) % 5 == 0:
            print(f"    [finetune] step {step+1}/{steps} loss={loss.item():.4f}")

    encoder_ft.eval()
    return encoder_ft


# ── Evaluation helper ─────────────────────────────────────────────────────────
def eval_wer(encoder_state, ctc_head, processor, samples, device, label):
    iv, refs = batch_to_inputs(samples, processor, device)

    def _score(state, iv, refs):
        return score_wer(state, iv, refs, processor, ctc_head, device)

    w = _score(encoder_state, iv, refs)
    print(f"  {label:35s} WER = {w:.4f}")
    return w


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = args.device
    print(f"Device: {device}\n")

    # Load processor and build shared frozen CTC head
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    vocab_size = processor.tokenizer.vocab_size
    ctc_head   = torch.nn.Linear(HIDDEN_DIM, vocab_size).to(device)
    torch.nn.init.xavier_uniform_(ctc_head.weight)
    ctc_head.eval()

    # Load data
    samples_a = load_samples("clean", args.n_samples)
    samples_b = load_samples("other", args.n_samples)
    iv_b, refs_b = batch_to_inputs(samples_b, processor, device)

    # Load pretrained encoder once
    print("\nLoading pretrained data2vec-audio encoder …")
    encoder_pretrained = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    encoder_pretrained.eval()
    base_state = copy.deepcopy(encoder_pretrained.state_dict())

    # ── Pre-adaptation baseline ───────────────────────────────────────────────
    print("\n── PRE-ADAPTATION (baseline) ─────────────────────────────────────")
    wer_a_pre = eval_wer(base_state, ctc_head, processor, samples_a,
                         device, "Task A before adaptation")
    wer_b_pre = eval_wer(base_state, ctc_head, processor, samples_b,
                         device, "Task B before adaptation")

    # ── RandOpt adaptation on Task B ─────────────────────────────────────────
    print("\n── RANDOPT ADAPTATION on Task B ─────────────────────────────────")
    cfg = RandOptConfig()
    cfg.sigma        = args.sigma
    cfg.n_candidates = args.n_candidates
    cfg.n_steps      = args.n_steps
    cfg.layers_from  = args.layers_from

    def _score_fn(state_dict, iv, refs):
        return score_wer(state_dict, iv, refs, processor, ctc_head, device)

    randopt = RandOpt(encoder_pretrained, _score_fn, cfg)
    randopt.adapt(score_args=(iv_b, refs_b))
    randopt.summary()

    randopt_state = randopt.get_state()
    print("\n── POST-RANDOPT ─────────────────────────────────────────────────")
    wer_a_randopt = eval_wer(randopt_state, ctc_head, processor, samples_a,
                             device, "Task A after RandOpt (Task B)")
    wer_b_randopt = eval_wer(randopt_state, ctc_head, processor, samples_b,
                             device, "Task B after RandOpt")

    # ── Naive fine-tuning adaptation on Task B ────────────────────────────────
    print("\n── NAIVE FINE-TUNING on Task B ──────────────────────────────────")
    encoder_ft = finetune_encoder(
        encoder_pretrained, ctc_head, iv_b, refs_b,
        processor, device,
        lr=args.finetune_lr,
        steps=args.finetune_steps,
        layers_from=args.layers_from,
    )
    ft_state = encoder_ft.state_dict()

    print("\n── POST-FINETUNE ─────────────────────────────────────────────────")
    wer_a_ft = eval_wer(ft_state, ctc_head, processor, samples_a,
                        device, "Task A after fine-tuning (Task B)")
    wer_b_ft = eval_wer(ft_state, ctc_head, processor, samples_b,
                        device, "Task B after fine-tuning")

    # ── Catastrophic Forgetting Summary ──────────────────────────────────────
    bwt_randopt  = wer_a_randopt - wer_a_pre
    bwt_finetune = wer_a_ft      - wer_a_pre

    print("\n" + "=" * 60)
    print("CATASTROPHIC FORGETTING ANALYSIS")
    print(f"  Task A WER before adaptation        : {wer_a_pre:.4f}")
    print()
    print(f"  [RandOpt]   Task A WER after          : {wer_a_randopt:.4f}")
    print(f"  [RandOpt]   Task B WER after          : {wer_b_randopt:.4f}")
    print(f"  [RandOpt]   BWT (forgetting on A)     : {bwt_randopt:+.4f}")
    print()
    print(f"  [Finetune]  Task A WER after          : {wer_a_ft:.4f}")
    print(f"  [Finetune]  Task B WER after          : {wer_b_ft:.4f}")
    print(f"  [Finetune]  BWT (forgetting on A)     : {bwt_finetune:+.4f}")
    print()
    if bwt_randopt < bwt_finetune:
        print("✓ RandOpt shows LESS catastrophic forgetting than fine-tuning.")
    else:
        print("⚠ Fine-tuning showed no more forgetting — try more fine-tune steps.")
    print("=" * 60)


if __name__ == "__main__":
    main()
