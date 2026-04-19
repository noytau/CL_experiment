"""
randopt.py
───────────────────────────────────────────────────────────────────────────────
RandOpt: Randomised Weight Perturbation for Continual Learning.

Algorithm (per adaptation step):
  1. Sample K random perturbation directions δ_k ~ N(0, σ²I)
     applied only to the upper encoder layers (LAYERS_FROM … 11).
  2. Score each perturbed model on the current-task mini-batch (WER proxy).
  3. Accept the best δ* that strictly improves upon the current model.
  4. If no δ improves, halve σ and retry (optional σ-annealing).
  5. Repeat for `n_steps` adaptation steps.

Key properties:
  • No gradient computation → no risk of exploding / vanishing gradients.
  • Only upper layers are perturbed → lower-layer acoustic representations
    (which capture domain-agnostic phonetics) are left untouched, naturally
    limiting catastrophic forgetting.
  • Perturbations are small (σ ≈ 1e-3) → the model stays in the "thicket"
    of equivalent solutions rather than jumping to a new basin.

Reference: loosely based on evolution-strategy / zeroth-order optimisation
ideas applied to continual learning (ES-MAML, SMAC, etc.).
"""

import copy
import torch
from typing import List, Tuple, Optional, Callable


# ── Configuration ─────────────────────────────────────────────────────────────

class RandOptConfig:
    sigma: float       = 1e-3    # initial noise std
    n_candidates: int  = 20      # K perturbations per step
    n_steps: int       = 10      # adaptation steps per new task
    layers_from: int   = 8       # first layer index to perturb (0-based)
    n_layers: int      = 12      # total encoder layers (data2vec-base has 12)
    sigma_decay: float = 0.5     # σ is multiplied by this when no candidate wins
    min_sigma: float   = 1e-5    # stop decaying below this
    seed: Optional[int] = None   # reproducibility


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_perturbable_params(encoder, layers_from: int) -> List[str]:
    """Return parameter names belonging to encoder layers >= layers_from."""
    return [
        name for name, _ in encoder.named_parameters()
        if "encoder.layers" in name
        and int(name.split("encoder.layers.")[1].split(".")[0]) >= layers_from
    ]


def apply_noise(state_dict: dict, param_names: List[str],
                sigma: float, seed: int) -> dict:
    """Return a new state_dict with Gaussian noise added to param_names."""
    torch.manual_seed(seed)
    noisy = copy.deepcopy(state_dict)
    for name in param_names:
        noise = torch.randn_like(noisy[name]) * sigma
        noisy[name] = noisy[name] + noise
    return noisy


# ── Core Algorithm ────────────────────────────────────────────────────────────

class RandOpt:
    """
    Gradient-free continual-learning adapter based on random weight perturbation.

    Parameters
    ----------
    encoder : Data2VecAudioModel
        Pre-trained (and possibly already adapted) encoder. **Not modified
        in-place** — call `.accept()` explicitly to commit the best delta.
    score_fn : Callable[[dict, ...], float]
        A function (state_dict, *score_args) → scalar loss/WER.
        Lower is better.
    config : RandOptConfig
    """

    def __init__(self, encoder, score_fn: Callable, config: RandOptConfig = None):
        if config is None:
            config = RandOptConfig()
        self.cfg       = config
        self.score_fn  = score_fn
        self.state     = copy.deepcopy(encoder.state_dict())
        self.param_names = get_perturbable_params(encoder, config.layers_from)
        self.sigma     = config.sigma
        self.history: List[dict] = []   # log of each step

        print(f"[RandOpt] Perturbable tensors: {len(self.param_names)} "
              f"(layers {config.layers_from}–{config.n_layers - 1})")

    def step(self, score_args: tuple = (), global_seed_offset: int = 0) -> dict:
        """
        Run one adaptation step: try cfg.n_candidates perturbations,
        accept the best one if it improves the current score.

        Returns a log dict with keys: sigma, base_score, best_score,
        best_seed, accepted.
        """
        base_score = self.score_fn(self.state, *score_args)

        best_score = base_score
        best_state = None
        best_seed  = -1

        for k in range(self.cfg.n_candidates):
            seed = global_seed_offset * self.cfg.n_candidates + k
            candidate = apply_noise(self.state, self.param_names,
                                    self.sigma, seed)
            s = self.score_fn(candidate, *score_args)
            if s < best_score:
                best_score = s
                best_state = candidate
                best_seed  = seed

        accepted = best_state is not None
        if accepted:
            self.state = best_state
        elif self.sigma > self.cfg.min_sigma:
            self.sigma *= self.cfg.sigma_decay   # shrink search radius

        log = {
            "sigma":      self.sigma,
            "base_score": base_score,
            "best_score": best_score,
            "best_seed":  best_seed,
            "accepted":   accepted,
        }
        self.history.append(log)
        return log

    def adapt(self, score_args: tuple = ()) -> List[dict]:
        """
        Run cfg.n_steps adaptation steps.
        Returns the full history list.
        """
        print(f"[RandOpt] Starting adaptation: "
              f"{self.cfg.n_steps} steps × {self.cfg.n_candidates} candidates")
        for i in range(self.cfg.n_steps):
            log = self.step(score_args, global_seed_offset=i)
            status = "✓ accepted" if log["accepted"] else "✗ rejected"
            print(f"  Step {i+1:3d}/{self.cfg.n_steps} | "
                  f"σ={log['sigma']:.2e} | "
                  f"score {log['base_score']:.4f} → {log['best_score']:.4f} "
                  f"[{status}]")
        return self.history

    def get_state(self) -> dict:
        """Return the current (best) model state dict."""
        return copy.deepcopy(self.state)

    def summary(self) -> None:
        if not self.history:
            print("[RandOpt] No steps recorded yet.")
            return
        n_accepted = sum(h["accepted"] for h in self.history)
        density    = n_accepted / len(self.history)
        scores     = [h["base_score"] for h in self.history]
        print("\n" + "=" * 55)
        print(f"[RandOpt] Adaptation summary")
        print(f"  Steps        : {len(self.history)}")
        print(f"  Accepted     : {n_accepted} / {len(self.history)} "
              f"(density δ = {density:.2f})")
        print(f"  Initial score: {scores[0]:.4f}")
        print(f"  Final score  : {scores[-1]:.4f}")
        print(f"  Final σ      : {self.history[-1]['sigma']:.2e}")
        print("=" * 55)
