#!/usr/bin/env python3
"""Generate paper figures from reproducible experiments (no hard-coded curves)."""

from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_CACHE_ROOT = Path(__file__).parent / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alfm_bem.adapters import AdapterConfig, BoundedAdapter
from alfm_bem.bem import BidirectionalExperienceMemory, CoverageMode
from alfm_bem.synthetic import generate_clustered_ood, generate_modes, generate_overlapping_experiences, normalize


def _set_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 10


@dataclass(frozen=True)
class OODMetrics:
    auc: float
    id_coverage_mean: float
    ood_coverage_mean: float


@dataclass(frozen=True)
class DriftMetrics:
    bounded_final_norm: float
    unbounded_final_norm: float
    bound: float


def _compute_roc(coverage_id: List[float], coverage_ood: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    # Label convention: 1 = ID, 0 = OOD; higher coverage should indicate ID.
    all_scores = np.asarray(coverage_id + coverage_ood, dtype=float)
    all_labels = np.asarray([1] * len(coverage_id) + [0] * len(coverage_ood), dtype=int)

    thresholds = np.linspace(0.0, 1.0, 201)
    tpr: List[float] = []
    fpr: List[float] = []

    for th in thresholds:
        tp = np.sum((all_scores > th) & (all_labels == 1))
        fn = np.sum((all_scores <= th) & (all_labels == 1))
        fp = np.sum((all_scores > th) & (all_labels == 0))
        tn = np.sum((all_scores <= th) & (all_labels == 0))

        tpr.append(tp / (tp + fn + 1e-10))
        fpr.append(fp / (fp + tn + 1e-10))

    # Sort by increasing FPR for trapezoidal AUC.
    fpr_arr = np.asarray(fpr)
    tpr_arr = np.asarray(tpr)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]
    auc = float(np.trapezoid(tpr_arr, fpr_arr))

    return fpr_arr, tpr_arr, auc


def compute_ood_metrics(
    *,
    seed: int = 42,
    dim: int = 64,
    n_train_fail: int = 500,
    n_train_succ: int = 500,
    n_id: int = 200,
    n_ood: int = 200,
    kde_bandwidth: float = 0.3,
) -> Tuple[OODMetrics, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Match the ablation setup (partially overlapping failure/success distributions).
    failure_modes = generate_modes(10, dim, rng=rng)
    success_modes = generate_modes(5, dim, rng=rng)
    train_emb, train_out, _, _ = generate_overlapping_experiences(
        n_train_fail,
        n_train_succ,
        dim,
        overlap=0.3,
        failure_modes=failure_modes,
        success_modes=success_modes,
        rng=rng,
    )

    bem = BidirectionalExperienceMemory(
        dim=dim,
        similarity_threshold=0.5,
        coverage_mode=CoverageMode.KDE,
        kde_bandwidth=kde_bandwidth,
    )

    for i, (vec, out) in enumerate(zip(train_emb, train_out)):
        bem.add_experience(vec, float(out), f"train_{i}")

    # ID samples: same mixture as failure modes.
    id_samples = []
    for _ in range(n_id):
        mode = failure_modes[rng.integers(len(failure_modes))]
        vec = mode + rng.standard_normal(dim) * 0.05
        id_samples.append(normalize(vec))

    ood_samples = generate_clustered_ood(n_ood, dim, shift_magnitude=3.0, rng=rng)

    id_cov = [bem.coverage_signal(z) for z in id_samples]
    ood_cov = [bem.coverage_signal(z) for z in ood_samples]

    fpr, tpr, auc = _compute_roc(id_cov, ood_cov)

    return (
        OODMetrics(
            auc=auc,
            id_coverage_mean=float(np.mean(id_cov)),
            ood_coverage_mean=float(np.mean(ood_cov)),
        ),
        fpr,
        tpr,
    )


def compute_drift_metrics(
    *,
    seed: int = 42,
    dim: int = 64,
    n_steps: int = 1000,
    batch_size: int = 32,
    bound: float = 2.0,
) -> Tuple[DriftMetrics, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Populate a BEM to supply experience replay batches.
    failure_modes = normalize(rng.standard_normal((10, dim)))
    failures = []
    for _ in range(500):
        mode = failure_modes[rng.integers(len(failure_modes))]
        vec = mode + rng.standard_normal(dim) * 0.05
        failures.append(normalize(vec))
    successes = normalize(rng.standard_normal((500, dim)))

    bem = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.7)
    for i, vec in enumerate(failures):
        bem.add_experience(vec, float(rng.uniform(-1.0, -0.3)), f"train_fail_{i}")
    for i, vec in enumerate(successes):
        bem.add_experience(vec, float(rng.uniform(0.3, 1.0)), f"train_succ_{i}")

    cfg_bounded = AdapterConfig(
        input_dim=dim,
        hidden_dim=64,
        output_dim=dim,
        max_grad_norm=0.5,
        max_param_norm=bound,
        learning_rate=1e-2,
        l2_weight=0.0,
    )
    bounded = BoundedAdapter(cfg_bounded)

    cfg_unbounded = AdapterConfig(
        input_dim=dim,
        hidden_dim=64,
        output_dim=dim,
        max_grad_norm=1e9,
        max_param_norm=1e9,
        learning_rate=1e-2,
        l2_weight=0.0,
    )
    unbounded = BoundedAdapter(cfg_unbounded)

    steps = np.arange(n_steps)
    bounded_norm = np.zeros(n_steps, dtype=float)
    unbounded_norm = np.zeros(n_steps, dtype=float)

    for step in range(n_steps):
        batch = bem.sample_for_training(batch_size)
        bounded.train_step(batch, None)
        unbounded.train_step(batch, None)
        bounded_norm[step] = bounded._compute_param_norm()
        unbounded_norm[step] = unbounded._compute_param_norm()

    return (
        DriftMetrics(
            bounded_final_norm=float(bounded_norm[-1]),
            unbounded_final_norm=float(unbounded_norm[-1]),
            bound=float(bound),
        ),
        steps,
        bounded_norm,
        unbounded_norm,
    )


def main() -> Dict[str, float]:
    _set_style()

    root = Path(__file__).parent
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1) OOD ROC
    ood_metrics, fpr, tpr = compute_ood_metrics()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"KDE coverage (AUC={ood_metrics.auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random (AUC=0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="blue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figures_dir / "ood_roc.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # 2) Adapter norm drift
    drift_metrics, steps, bounded_norm, unbounded_norm = compute_drift_metrics()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(steps, bounded_norm, "b-", linewidth=2, label="Bounded")
    ax.plot(steps, unbounded_norm, "r-", linewidth=2, label="Unbounded")
    ax.axhline(y=drift_metrics.bound, color="blue", linestyle="--", alpha=0.5, label=f"Bound ($c_\\theta={drift_metrics.bound:.1f}$)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Parameter Norm $\\|\\theta\\|_F$")
    ax.set_xlim([0, steps[-1]])
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "drift.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    metrics_path = figures_dir / "paper_metrics.json"
    metrics_payload = {
        "ood": asdict(ood_metrics),
        "drift": asdict(drift_metrics),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {figures_dir / 'ood_roc.pdf'}")
    print(f"Saved {figures_dir / 'drift.pdf'}")
    print(f"Saved {metrics_path}")

    return {
        "ood_auc": ood_metrics.auc,
        "ood_id_mean": ood_metrics.id_coverage_mean,
        "ood_ood_mean": ood_metrics.ood_coverage_mean,
        "bounded_final_norm": drift_metrics.bounded_final_norm,
        "unbounded_final_norm": drift_metrics.unbounded_final_norm,
    }


if __name__ == "__main__":
    main()
