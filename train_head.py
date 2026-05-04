"""
train_head.py — Experiment 1
Train a Linear CTC head on a frozen data2vec-audio-base encoder (LibriSpeech clean).
Target: Task A WER from ~1.0 (random head) to ~5–15% (trained head).
Checkpoint is saved to --save_dir and tracked via wandb.
"""

import argparse
import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer as compute_wer
import wandb

ENCODER_NAME   = "facebook/data2vec-audio-base"
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"
HIDDEN_DIM     = 768
WANDB_PROJECT  = "CL_experiment"


def parse_args():
    p = argparse.ArgumentParser(description="Exp 1: train CTC head on frozen data2vec encoder")
    p.add_argument("--n_train",    type=int,   default=500,
                   help="Training samples from train.100 (0 = full split ~28k)")
    p.add_argument("--n_val",      type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--n_epochs",   type=int,   default=20)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--save_dir",   type=str,   default="checkpoints/exp1")
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--hf_home",    type=str,   default=None,
                   help="HuggingFace cache dir — set to a PVC path on cluster")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["audio"]["array"], s["text"].lower().strip()


def collate_fn(batch, processor):
    audio, refs = zip(*batch)
    iv = processor(
        list(audio), sampling_rate=16_000,
        return_tensors="pt", padding=True,
    ).input_values
    return iv, list(refs)


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_wer(encoder, ctc_head, processor, samples, device, batch_size):
    encoder.eval()
    ctc_head.eval()
    all_preds, all_refs = [], []

    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        audio = [s["audio"]["array"] for s in chunk]
        refs  = [s["text"].lower().strip() for s in chunk]
        iv = processor(audio, sampling_rate=16_000,
                       return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            hidden   = encoder(iv).last_hidden_state
            logits   = ctc_head(hidden)
            pred_ids = torch.argmax(logits, dim=-1)
            preds    = [p.lower().strip() for p in processor.batch_decode(pred_ids)]
        all_preds.extend(preds)
        all_refs.extend(refs)

    return compute_wer(all_refs, all_preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    os.makedirs(args.save_dir, exist_ok=True)

    run_name = args.run_name or f"exp1_head_n{args.n_train}_ep{args.n_epochs}_lr{args.lr}"
    wandb.init(project=WANDB_PROJECT, name=run_name, config=vars(args))
    print(f"[wandb] {wandb.run.get_url()}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading processor …")
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)

    print("Loading frozen encoder …")
    encoder = Data2VecAudioModel.from_pretrained(ENCODER_NAME).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    vocab_size = processor.tokenizer.vocab_size
    torch.manual_seed(42)
    ctc_head = nn.Linear(HIDDEN_DIM, vocab_size).to(device)
    nn.init.xavier_uniform_(ctc_head.weight)

    optimizer   = torch.optim.Adam(ctc_head.parameters(), lr=args.lr)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    n_params = sum(p.numel() for p in ctc_head.parameters())
    print(f"CTC head params: {n_params:,}  (vocab_size={vocab_size})")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading LibriSpeech clean train.100 …")
    raw_train = load_dataset("librispeech_asr", "clean", split="train.100",
                             trust_remote_code=True)
    raw_val   = load_dataset("librispeech_asr", "clean", split="validation",
                             trust_remote_code=True)

    train_samples = list(raw_train.take(args.n_train)) if args.n_train > 0 else list(raw_train)
    val_samples   = list(raw_val.take(args.n_val))
    print(f"Train: {len(train_samples)} samples  |  Val: {len(val_samples)} samples\n")

    loader = DataLoader(
        AudioDataset(train_samples),
        batch_size = args.batch_size,
        shuffle    = True,
        collate_fn = partial(collate_fn, processor=processor),
        num_workers = 0,
    )

    # ── Initial WER (random head baseline) ────────────────────────────────────
    wer_init = eval_wer(encoder, ctc_head, processor, val_samples, device, args.batch_size)
    print(f"Initial val WER (random head) = {wer_init:.4f}")
    wandb.log({"val/wer": wer_init, "epoch": 0})

    best_wer  = wer_init
    ckpt_path = os.path.join(args.save_dir, "best_ctc_head.pt")

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, args.n_epochs + 1):
        ctc_head.train()
        epoch_loss = 0.0
        t0 = time.time()

        for iv, refs in loader:
            iv = iv.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                hidden = encoder(iv).last_hidden_state           # (B, T, 768)

            logits    = ctc_head(hidden)                          # (B, T, V)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, V)

            target_ids     = [torch.tensor(processor.tokenizer.encode(r), dtype=torch.long)
                              for r in refs]
            targets        = torch.cat(target_ids)
            input_lengths  = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in target_ids], dtype=torch.long)

            loss = ctc_loss_fn(
                log_probs.cpu(), targets.cpu(),
                input_lengths.cpu(), target_lengths.cpu(),
            )
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "step": global_step})

        avg_loss = epoch_loss / len(loader)
        val_wer  = eval_wer(encoder, ctc_head, processor, val_samples, device, args.batch_size)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.n_epochs}  "
              f"loss={avg_loss:.4f}  val_wer={val_wer:.4f}  ({elapsed:.0f}s)")
        wandb.log({"train/epoch_loss": avg_loss, "val/wer": val_wer, "epoch": epoch})

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(ctc_head.state_dict(), ckpt_path)
            print(f"  ✓ New best WER = {best_wer:.4f}  → {ckpt_path}")
            wandb.log({"val/best_wer": best_wer, "epoch": epoch})

    print(f"\nTraining complete. Best val WER = {best_wer:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
