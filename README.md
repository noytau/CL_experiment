# CL_experiment — Continual Learning with data2vec-audio + RandOpt

Gradient-free continual learning for speech models using random weight
perturbation (**RandOpt**) applied to HuggingFace's `data2vec-audio-base`.

---

## 1. Motivation

Standard fine-tuning on a new audio domain (Task B) causes **catastrophic
forgetting**: the model loses its representations for the original domain
(Task A). RandOpt avoids gradient updates entirely — it finds beneficial weight
perturbations through random search, staying in the "thicket" of solutions near
the pre-trained weights.

---

## 2. Repository Structure

```
CL_experiment/
├── requirements.txt       # pip dependencies
├── baseline_asr.py        # WER of frozen encoder (no adaptation)
├── randopt.py             # RandOpt algorithm (reusable module)
├── cl_train.py            # Full CL experiment: baseline vs finetune vs RandOpt
└── smoke_test.py          # Minimal POC: does perturbation change WER?
```

---

## 3. Installation

```bash
pip install -r requirements.txt
```

### Key dependencies
| Package | Role |
|---------|------|
| `transformers` | `Data2VecAudioModel`, `Wav2Vec2Processor` |
| `datasets` | LibriSpeech via HuggingFace Hub |
| `jiwer` | Word Error Rate computation |
| `torch` + `torchaudio` | Model inference and audio I/O |

---

## 4. Quick Start

### Smoke test (verify perturbation changes WER, ~5 min GPU)
```bash
python smoke_test.py
```

### Frozen-encoder baseline
```bash
python baseline_asr.py --n_samples 20
```

### Full CL experiment
```bash
python cl_train.py \
    --n_samples 20 \
    --n_steps 10 \
    --n_candidates 20 \
    --sigma 1e-3 \
    --layers_from 8
```

---

## 5. RandOpt Algorithm — Design and Implementation Steps

### 5.1 High-level idea

RandOpt is a **zeroth-order** (gradient-free) optimiser. At each adaptation
step it:

1. Samples `K` noise vectors `δ_k ~ N(0, σ²I)`.
2. Adds each δ_k only to **upper encoder layers** (layers 8–11 by default).
3. Evaluates each perturbed model on the new-task mini-batch (WER or CTC loss).
4. **Accepts** the δ* that yields the best improvement; rejects all others.
5. If no δ improves, σ is halved (σ-annealing) — the search radius shrinks.

The process is repeated for `n_steps` steps.

### 5.2 Implementation steps (in `randopt.py`)

| Step | Code location | Description |
|------|--------------|-------------|
| 1 | `get_perturbable_params()` | Identify which parameters belong to layers >= `LAYERS_FROM` |
| 2 | `apply_noise()` | Deep-copy state dict, add N(0,σ²) noise to selected params |
| 3 | `RandOpt.step()` | Score all K candidates with the user-supplied `score_fn` |
| 4 | `RandOpt.step()` | Accept best candidate; if none wins, decay σ |
| 5 | `RandOpt.adapt()` | Outer loop over `n_steps` |
| 6 | `RandOpt.summary()` | Print acceptance density δ(m) and score trajectory |

### 5.3 Integration with data2vec

`data2vec-audio-base` has:
- 7-layer CNN feature extractor (frozen — acoustic features, do not touch)
- 12 Transformer encoder layers (layers 0–11)

**RandOpt targets layers 8–11** (upper half), because:
- Layers 0–7 encode domain-agnostic phonetics — leaving them intact limits forgetting.
- Layers 8–11 encode higher-level contextual representations — most task-specific, most benefit from domain adaptation.

### 5.4 Hyperparameter guidance

| Param | Default | When to change |
|-------|---------|---------------|
| `sigma` | 1e-3 | Decrease to 1e-4 if no candidates win; increase to 5e-3 for very different domains |
| `n_candidates` | 20 | Increase to 50–100 for better search at the cost of time |
| `n_steps` | 10 | Increase to 50–200 for real adaptation runs |
| `layers_from` | 8 | Decrease to 6 for more aggressive adaptation; 10 for minimal forgetting |
| `sigma_decay` | 0.5 | Lower (0.7) to decay more slowly |

---

## 6. Baseline: data2vec + LibriSpeech ASR

### 6.1 What the baseline measures

The **frozen encoder + random CTC head** baseline answers:

> *Does the pre-trained encoder already contain enough information to
> differentiate between utterances, even with a randomly initialised
> output layer?*

WER with a random CTC head will be near 1.0 — that is expected.
What matters is whether perturbations **change** WER (confirmed by `smoke_test.py`),
and whether some perturbations **improve** it.

### 6.2 Do we need to fine-tune data2vec?

**Short answer: No for the RandOpt POC. Yes for production ASR.**

| Setting | Fine-tune encoder? | Reason |
|---------|--------------------|--------|
| RandOpt POC / smoke test | **No** | We only care that WER changes with perturbations; random CTC head is sufficient |
| Catastrophic forgetting measurement | **No** (encoder) | We compare encoder states, not absolute WER values |
| Real ASR deployment | **Yes** | CTC head must be trained; encoder can be partially unfrozen |

**Why not fine-tune the encoder for the CL experiment?**

Fine-tuning the full encoder with gradient descent is exactly what we are
trying to *replace* with RandOpt. Fine-tuning first would:
- Conflate the two methods.
- Risk catastrophic forgetting before RandOpt even runs.

The correct experimental setup is:
```
Pre-trained data2vec → [RandOpt on Task B]         → evaluate on Task A + B
Pre-trained data2vec → [Gradient fine-tune on Task B] → evaluate on Task A + B
```

---

## 7. Catastrophic Forgetting: Explanation and Measurement

### 7.1 What is catastrophic forgetting?

When a neural network is gradient-trained on Task B, weights shift to minimise
Task B loss. Because the loss landscape is not shared between tasks, these
weight updates often **overwrite** features learned for Task A — particularly in
upper layers closest to the output.

In ASR terms: after fine-tuning on LibriSpeech *other* (noisier, different
speaker distribution), the model forgets how to decode LibriSpeech *clean*.

### 7.2 Why RandOpt resists forgetting

1. **No gradient descent** — the model never follows a gradient pointing away
   from Task A's basin. Each accepted perturbation is small (σ ≈ 1e-3), so the
   model stays in a neighbourhood of the pre-trained weights.

2. **Partial parameter scope** — only layers 8–11 are perturbed, leaving
   lower-layer acoustic representations intact.

3. **Acceptance gating** — a perturbation is only accepted if it strictly
   improves Task B score. This prevents random drift that could hurt Task A.

### 7.3 Measurement: Backward Transfer (BWT)

```
BWT = WER_A_after_adapting_on_B - WER_A_before_adapting_on_B
```

- BWT > 0  →  forgetting (Task A WER increased).
- BWT ≈ 0  →  no forgetting.
- BWT < 0  →  positive backward transfer (rare).

### 7.4 Expected results

| Method | Task B WER | Task A BWT |
|--------|-----------|-----------|
| Frozen baseline | High (no adaptation) | ~0 |
| Naive fine-tuning (many steps) | Lower | Positive (forgetting) |
| RandOpt (σ=1e-3, 10 steps) | Slightly lower | Near-zero or negative |

---

## 8. Next Steps

1. **Train the CTC head** on LibriSpeech clean (supervised fine-tuning) so
   Task A WER starts at a meaningful level (~5–15%) before the CL experiment.
2. **Scale RandOpt** — increase `n_steps` to 100–500 and `n_candidates` to 50.
3. **Add EWC baseline** — Elastic Weight Consolidation is the standard CL
   comparison point.
4. **Evaluate on your domain** — replace Task B with your custom audio dataset
   (SpectralFM / NOVA domain).
5. **σ schedule** — implement cosine σ annealing instead of step decay.
