"""
baseline_asr.py
───────────────────────────────────────────────────────────────────────────────
Baseline: data2vec-audio-base encoder + randomly-initialised CTC head,
evaluated on LibriSpeech (HuggingFace datasets).

This script establishes the *un-adapted* WER on two data splits:
  • Task A  – librispeech_asr / clean  / validation   (100h domain)
  • Task B  – librispeech_asr / other  / validation   (noisy/other domain)

No fine-tuning happens here — the encoder stays frozen in its pretrained state.
The purpose is to record the WER floor before any continual-learning adaptation,
so later experiments can measure both forward transfer (Task B improves) and
backward interference / catastrophic forgetting (Task A degrades).

Usage:
    python baseline_asr.py [--n_samples 20] [--device cuda]
"""

import argparse
import copy
import torch
import numpy as np
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer

# ── Defaults ────────────────────────────────────────────────────────────────
ENCODER_NAME  = "facebook/data2vec-audio-base"
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"   # shares tokenizer with d2v-audio
HIDDEN_DIM    = 768
N_SAMPLES     = 20     # utterances per task — keep small for quick baseline

# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(processor, device):
    """Frozen data2vec encoder + random CTC head."""
    encoder = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    vocab_size = processor.tokenizer.vocab_size
    ctc_head = torch.nn.Linear(HIDDEN_DIM, vocab_size).to(device)
    torch.nn.init.xavier_uniform_(ctc_head.weight)
    ctc_head.eval()
    return encoder, ctc_head


def load_split(split_name: str, config: str, n: int):
    """
    Load n samples from a LibriSpeech split.

    split_name : "validation" | "test"
    config     : "clean" | "other"
    """
    print(f"  Loading librispeech_asr [{config}] {split_name} …")
    ds = load_dataset(
        "librispeech_asr", config,
        split=split_name,
        streaming=False,
        trust_remote_code=True,
    )
    return list(ds.take(n))


def score(encoder, ctc_head, processor, samples, device):
    """Return WER for a list of HuggingFace dataset samples."""
    audio  = [s["audio"]["array"] for s in samples]
    refs   = [s["text"].lower().strip() for s in samples]

    inputs = processor(
        audio, sampling_rate=16_000,
        return_tensors="pt", padding=True,
    ).input_values.to(device)

    with torch.no_grad():
        hidden  = encoder(inputs).last_hidden_state      # (B, T, 768)
        logits  = ctc_head(hidden)                        # (B, T, vocab)
        pred_ids = torch.argmax(logits, dim=-1)
        preds   = [p.lower().strip()
                   for p in processor.batch_decode(pred_ids)]

    return wer(refs, preds), preds, refs


def main():
    args   = parse_args()
    device = args.device
    print(f"Device: {device}")

    print("Loading processor …")
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)

    print("Building model (frozen encoder + random CTC head) …")
    encoder, ctc_head = build_model(processor, device)

    # ── Task A: LibriSpeech clean ────────────────────────────────────────────
    samples_a = load_split("validation", "clean", args.n_samples)
    wer_a, preds_a, refs_a = score(encoder, ctc_head, processor,
                                   samples_a, device)
    print(f"\nTask A (clean)  WER = {wer_a:.4f}")
    for ref, pred in list(zip(refs_a, preds_a))[:3]:
        print(f"  REF : {ref}")
        print(f"  PRED: {pred}")
        print()

    # ── Task B: LibriSpeech other ────────────────────────────────────────────
    samples_b = load_split("validation", "other", args.n_samples)
    wer_b, preds_b, refs_b = score(encoder, ctc_head, processor,
                                   samples_b, device)
    print(f"Task B (other)  WER = {wer_b:.4f}")
    for ref, pred in list(zip(refs_b, preds_b))[:3]:
        print(f"  REF : {ref}")
        print(f"  PRED: {pred}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"BASELINE (no fine-tuning, random CTC head)")
    print(f"  Task A WER : {wer_a:.4f}")
    print(f"  Task B WER : {wer_b:.4f}")
    print("=" * 50)
    print(
        "\nNote: WER will be near 1.0 with a random CTC head.\n"
        "The goal is not low absolute WER — it is to see WER *change*\n"
        "across perturbations, revealing that the encoder representations\n"
        "are non-trivially different between perturbed and unperturbed states."
    )

    return {
        "task_a_wer": wer_a,
        "task_b_wer": wer_b,
        "encoder_state": copy.deepcopy(encoder.state_dict()),
    }


if __name__ == "__main__":
    main()
