"""
smoke_test.py
Minimal RandOpt POC on data2vec-audio.
Goal: verify that weight perturbation changes WER.
Expected runtime: ~5 min on GPU, ~30 min on CPU.
"""
import torch
import copy
import numpy as np
from transformers import Data2VecAudioModel, Wav2Vec2Processor
from datasets import load_dataset
from jiwer import wer

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "facebook/data2vec-audio-base"
N_SAMPLES    = 10       # utterances for scoring (keep tiny for smoke test)
N_PERTURB    = 20       # number of random perturbations to try
SIGMA        = 1e-3     # noise intensity — start here, tune later
LAYERS_FROM  = 8        # only perturb layers 8-11
# ────────────────────────────────────────────────────────────────────────────

print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
encoder   = Data2VecAudioModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()

# Simple CTC head — random init is fine for smoke test
# (we only care that WER *changes* between perturbations, not the absolute value)
VOCAB_SIZE = processor.tokenizer.vocab_size   # typically 32
ctc_head   = torch.nn.Linear(768, VOCAB_SIZE).to(DEVICE)
torch.nn.init.xavier_uniform_(ctc_head.weight)
ctc_head.eval()

print("Loading 10 samples from LibriSpeech dev-clean...")
ds = load_dataset(
    "librispeech_asr", "clean", split="validation",
    streaming=False, trust_remote_code=True
)
samples = list(ds.take(N_SAMPLES))

def get_inputs(samples):
    """Preprocess audio samples into model inputs."""
    audio_arrays = [s["audio"]["array"] for s in samples]
    refs         = [s["text"].lower() for s in samples]
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    return inputs.input_values.to(DEVICE), refs

def score_wer(state_dict, input_values, refs):
    """Score WER for a given encoder state dict."""
    # Load state into a temp copy (no grad needed)
    tmp = Data2VecAudioModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tmp.load_state_dict(state_dict)
    tmp.eval()
    with torch.no_grad():
        hidden = tmp(input_values).last_hidden_state   # (B, T, 768)
        logits = ctc_head(hidden)                       # (B, T, vocab)
        pred_ids = torch.argmax(logits, dim=-1)
        preds = processor.batch_decode(pred_ids)
    # lower-case and strip whitespace
    preds = [p.lower().strip() for p in preds]
    refs  = [r.lower().strip() for r in refs]
    return wer(refs, preds)

# ── Get perturbable parameter names ─────────────────────────────────────────
perturbable = [
    name for name, _ in encoder.named_parameters()
    if "encoder.layers" in name
    and int(name.split("encoder.layers.")[1].split(".")[0]) >= LAYERS_FROM
]
print(f"Perturbable params: {len(perturbable)} tensors "
      f"(layers {LAYERS_FROM}–11 only)")

# ── Baseline ─────────────────────────────────────────────────────────────────
input_values, refs = get_inputs(samples)
base_state = copy.deepcopy(encoder.state_dict())
base_wer   = score_wer(base_state, input_values, refs)
print(f"\nBaseline WER: {base_wer:.4f}")

# ── Perturbation loop ────────────────────────────────────────────────────────
results = []
for i in range(N_PERTURB):
    torch.manual_seed(i)
    noisy_state = copy.deepcopy(base_state)
    for name in perturbable:
        noise = torch.randn_like(noisy_state[name]) * SIGMA
        noisy_state[name] = noisy_state[name] + noise

    w = score_wer(noisy_state, input_values, refs)
    delta = base_wer - w      # positive = improvement
    results.append((i, w, delta))
    print(f"  Seed {i:3d} | WER={w:.4f} | ΔWER={delta:+.4f}"
          f"  {'✓ better' if delta > 0 else '✗ worse'}")

# ── Summary ──────────────────────────────────────────────────────────────────
improving = [r for r in results if r[2] > 0]
density   = len(improving) / N_PERTURB
best      = min(results, key=lambda r: r[1])

print(f"\n{'='*50}")
print(f"Baseline WER:        {base_wer:.4f}")
print(f"Best perturbed WER:  {best[1]:.4f}  (seed {best[0]})")
print(f"Perturbation density δ(m) = {density:.2f}  "
      f"({len(improving)}/{N_PERTURB} improved)")
print(f"{'='*50}")

if density > 0:
    print("\n✓ POC succeeded: perturbations change WER and some improve it.")
    print("  The model is showing early thicket behavior.")
else:
    print("\n⚠ No perturbations improved WER.")
    print("  Try: smaller sigma (1e-4), more samples (N_PERTURB=50),")
    print("       or train the CTC head first (see next step).")
