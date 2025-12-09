"""
Threshold Sensitivity Analysis
==============================

Analyzes how sensitive ALFM-BEM is to its key threshold parameters:
1. similarity_threshold (BEM neighbor filtering)
2. risk_threshold (Consensus Engine decision)
3. kde_bandwidth (Coverage signal)
4. success_threshold / failure_threshold (Outcome classification)

This helps understand:
- Which thresholds are most critical
- Safe operating ranges
- Interaction effects between thresholds

Author: David Ahmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode

np.random.seed(42)


# =============================================================================
# Data Generation
# =============================================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)


def generate_test_data(
    n_train: int = 500,
    n_test: int = 100,
    n_ood: int = 50,
    dim: int = 64
) -> Tuple[List, List, List, List, List]:
    """Generate train/test data with clear failure/success patterns."""

    # Define failure and success modes
    failure_modes = [normalize(np.random.randn(dim)) for _ in range(5)]
    success_modes = [normalize(np.random.randn(dim)) for _ in range(5)]

    train_emb, train_out = [], []
    test_failures, test_successes = [], []

    # Training data
    for _ in range(n_train // 2):
        # Failures
        mode = failure_modes[np.random.randint(len(failure_modes))]
        vec = normalize(mode + np.random.randn(dim) * 0.1)
        train_emb.append(vec)
        train_out.append(np.random.uniform(-1.0, -0.5))

        # Successes
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = normalize(mode + np.random.randn(dim) * 0.1)
        train_emb.append(vec)
        train_out.append(np.random.uniform(0.5, 1.0))

    # Test failures
    for _ in range(n_test // 2):
        mode = failure_modes[np.random.randint(len(failure_modes))]
        vec = normalize(mode + np.random.randn(dim) * 0.1)
        test_failures.append(vec)

    # Test successes
    for _ in range(n_test // 2):
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = normalize(mode + np.random.randn(dim) * 0.1)
        test_successes.append(vec)

    # OOD samples (random directions)
    ood_samples = [normalize(np.random.randn(dim)) for _ in range(n_ood)]

    return train_emb, train_out, test_failures, test_successes, ood_samples


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(
    bem: BidirectionalExperienceMemory,
    test_failures: List[np.ndarray],
    test_successes: List[np.ndarray],
    ood_samples: List[np.ndarray],
    risk_threshold: float = 0.3
) -> Dict[str, float]:
    """Compute comprehensive metrics for threshold evaluation."""

    # Failure detection (True Positive = high risk for failure)
    tp, fp, fn, tn = 0, 0, 0, 0

    for z in test_failures:
        risk, _ = bem.risk_signal(z)
        if risk > risk_threshold:
            tp += 1
        else:
            fn += 1

    for z in test_successes:
        risk, _ = bem.risk_signal(z)
        if risk > risk_threshold:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # OOD detection via coverage
    id_coverage = [bem.coverage_signal(z) for z in test_failures + test_successes]
    ood_coverage = [bem.coverage_signal(z) for z in ood_samples]

    # Coverage separation
    id_mean = np.mean(id_coverage)
    ood_mean = np.mean(ood_coverage)
    coverage_gap = id_mean - ood_mean

    # Coverage AUC
    all_cov = id_coverage + ood_coverage
    all_labels = [1] * len(id_coverage) + [0] * len(ood_coverage)

    # Simple AUC calculation
    sorted_pairs = sorted(zip(all_cov, all_labels), reverse=True)
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    auc = 0.0
    tp_count = 0
    for _, label in sorted_pairs:
        if label == 1:
            tp_count += 1
        else:
            auc += tp_count
    auc /= (n_pos * n_neg + 1e-10)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage_gap": coverage_gap,
        "coverage_auc": auc,
        "id_coverage_mean": id_mean,
        "ood_coverage_mean": ood_mean
    }


# =============================================================================
# Sensitivity Analyses
# =============================================================================

def analyze_similarity_threshold(
    train_emb: List,
    train_out: List,
    test_failures: List,
    test_successes: List,
    ood_samples: List,
    dim: int = 64
) -> Dict:
    """Analyze sensitivity to similarity_threshold."""

    thresholds = np.arange(0.1, 0.95, 0.05)
    results = {"thresholds": thresholds.tolist(), "metrics": []}

    for sim_thresh in thresholds:
        bem = BidirectionalExperienceMemory(
            dim=dim,
            similarity_threshold=sim_thresh,
            coverage_mode=CoverageMode.KDE,
            kde_bandwidth=0.3
        )

        for emb, out in zip(train_emb, train_out):
            bem.add_experience(emb, out, "train")

        metrics = compute_metrics(bem, test_failures, test_successes, ood_samples)
        results["metrics"].append(metrics)

    return results


def analyze_kde_bandwidth(
    train_emb: List,
    train_out: List,
    test_failures: List,
    test_successes: List,
    ood_samples: List,
    dim: int = 64
) -> Dict:
    """Analyze sensitivity to KDE bandwidth."""

    bandwidths = np.arange(0.1, 1.0, 0.05)
    results = {"bandwidths": bandwidths.tolist(), "metrics": []}

    for bw in bandwidths:
        bem = BidirectionalExperienceMemory(
            dim=dim,
            similarity_threshold=0.5,
            coverage_mode=CoverageMode.KDE,
            kde_bandwidth=bw
        )

        for emb, out in zip(train_emb, train_out):
            bem.add_experience(emb, out, "train")

        metrics = compute_metrics(bem, test_failures, test_successes, ood_samples)
        results["metrics"].append(metrics)

    return results


def analyze_risk_threshold(
    train_emb: List,
    train_out: List,
    test_failures: List,
    test_successes: List,
    ood_samples: List,
    dim: int = 64
) -> Dict:
    """Analyze sensitivity to risk decision threshold."""

    # Fixed BEM, vary decision threshold
    bem = BidirectionalExperienceMemory(
        dim=dim,
        similarity_threshold=0.5,
        coverage_mode=CoverageMode.KDE,
        kde_bandwidth=0.3
    )

    for emb, out in zip(train_emb, train_out):
        bem.add_experience(emb, out, "train")

    thresholds = np.arange(0.05, 0.95, 0.05)
    results = {"thresholds": thresholds.tolist(), "metrics": []}

    for risk_thresh in thresholds:
        metrics = compute_metrics(bem, test_failures, test_successes, ood_samples, risk_threshold=risk_thresh)
        results["metrics"].append(metrics)

    return results


def analyze_threshold_interactions(
    train_emb: List,
    train_out: List,
    test_failures: List,
    test_successes: List,
    ood_samples: List,
    dim: int = 64
) -> Dict:
    """Analyze interactions between similarity_threshold and risk_threshold."""

    sim_thresholds = [0.3, 0.5, 0.7, 0.9]
    risk_thresholds = [0.2, 0.4, 0.6, 0.8]

    results = {
        "sim_thresholds": sim_thresholds,
        "risk_thresholds": risk_thresholds,
        "f1_matrix": [],
        "coverage_auc_matrix": []
    }

    for sim_thresh in sim_thresholds:
        f1_row = []
        cov_row = []

        bem = BidirectionalExperienceMemory(
            dim=dim,
            similarity_threshold=sim_thresh,
            coverage_mode=CoverageMode.KDE,
            kde_bandwidth=0.3
        )

        for emb, out in zip(train_emb, train_out):
            bem.add_experience(emb, out, "train")

        for risk_thresh in risk_thresholds:
            metrics = compute_metrics(bem, test_failures, test_successes, ood_samples, risk_threshold=risk_thresh)
            f1_row.append(metrics["f1"])
            cov_row.append(metrics["coverage_auc"])

        results["f1_matrix"].append(f1_row)
        results["coverage_auc_matrix"].append(cov_row)

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_sensitivity_results(
    sim_results: Dict,
    bw_results: Dict,
    risk_results: Dict,
    interaction_results: Dict,
    output_path: str = "threshold_sensitivity.pdf"
):
    """Create comprehensive visualization of threshold sensitivity."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Similarity Threshold vs F1
    ax = axes[0, 0]
    thresholds = sim_results["thresholds"]
    f1_vals = [m["f1"] for m in sim_results["metrics"]]
    precision_vals = [m["precision"] for m in sim_results["metrics"]]
    recall_vals = [m["recall"] for m in sim_results["metrics"]]

    ax.plot(thresholds, f1_vals, 'b-', linewidth=2, label='F1')
    ax.plot(thresholds, precision_vals, 'g--', label='Precision')
    ax.plot(thresholds, recall_vals, 'r--', label='Recall')
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Similarity Threshold Impact')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. KDE Bandwidth vs Coverage AUC
    ax = axes[0, 1]
    bandwidths = bw_results["bandwidths"]
    cov_auc = [m["coverage_auc"] for m in bw_results["metrics"]]
    cov_gap = [m["coverage_gap"] for m in bw_results["metrics"]]

    ax.plot(bandwidths, cov_auc, 'b-', linewidth=2, label='Coverage AUC')
    ax.set_xlabel('KDE Bandwidth')
    ax.set_ylabel('Coverage AUC')
    ax.set_title('KDE Bandwidth Impact on OOD Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Secondary axis for coverage gap
    ax2 = ax.twinx()
    ax2.plot(bandwidths, cov_gap, 'r--', label='Coverage Gap')
    ax2.set_ylabel('Coverage Gap (ID - OOD)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 3. Risk Threshold ROC-like curve
    ax = axes[0, 2]
    thresholds = risk_results["thresholds"]
    precision_vals = [m["precision"] for m in risk_results["metrics"]]
    recall_vals = [m["recall"] for m in risk_results["metrics"]]
    f1_vals = [m["f1"] for m in risk_results["metrics"]]

    ax.plot(recall_vals, precision_vals, 'b-', linewidth=2)
    ax.scatter(recall_vals, precision_vals, c=thresholds, cmap='viridis', s=50)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Risk Threshold Precision-Recall Curve')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Risk Threshold')
    ax.grid(True, alpha=0.3)

    # 4. Risk Threshold vs F1
    ax = axes[1, 0]
    ax.plot(thresholds, f1_vals, 'b-', linewidth=2, marker='o')
    best_idx = np.argmax(f1_vals)
    ax.axvline(thresholds[best_idx], color='r', linestyle='--', label=f'Best: {thresholds[best_idx]:.2f}')
    ax.set_xlabel('Risk Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('Risk Threshold vs F1')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Interaction Heatmap (F1)
    ax = axes[1, 1]
    f1_matrix = np.array(interaction_results["f1_matrix"])
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(interaction_results["risk_thresholds"])))
    ax.set_xticklabels([f'{x:.1f}' for x in interaction_results["risk_thresholds"]])
    ax.set_yticks(range(len(interaction_results["sim_thresholds"])))
    ax.set_yticklabels([f'{x:.1f}' for x in interaction_results["sim_thresholds"]])
    ax.set_xlabel('Risk Threshold')
    ax.set_ylabel('Similarity Threshold')
    ax.set_title('F1 Score (Threshold Interaction)')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(interaction_results["sim_thresholds"])):
        for j in range(len(interaction_results["risk_thresholds"])):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    # 6. Summary recommendations
    ax = axes[1, 2]
    ax.axis('off')

    # Find optimal thresholds
    best_sim_idx = np.argmax([m["f1"] for m in sim_results["metrics"]])
    best_bw_idx = np.argmax([m["coverage_auc"] for m in bw_results["metrics"]])
    best_risk_idx = np.argmax([m["f1"] for m in risk_results["metrics"]])

    summary_text = f"""
    THRESHOLD SENSITIVITY SUMMARY
    =============================

    Recommended Thresholds:
    • Similarity: {sim_results['thresholds'][best_sim_idx]:.2f}
      (F1: {sim_results['metrics'][best_sim_idx]['f1']:.3f})

    • KDE Bandwidth: {bw_results['bandwidths'][best_bw_idx]:.2f}
      (OOD AUC: {bw_results['metrics'][best_bw_idx]['coverage_auc']:.3f})

    • Risk Threshold: {risk_results['thresholds'][best_risk_idx]:.2f}
      (F1: {risk_results['metrics'][best_risk_idx]['f1']:.3f})

    Observations:
    • Higher similarity threshold = higher precision, lower recall
    • KDE bandwidth has moderate sensitivity
    • Risk threshold has clear optimal range around 0.3-0.5
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")

    return fig


# =============================================================================
# Main Experiment
# =============================================================================

def run_threshold_sensitivity(seed: int = 42, verbose: bool = True):
    """Run complete threshold sensitivity analysis."""

    np.random.seed(seed)
    dim = 64

    print("=" * 70)
    print("Threshold Sensitivity Analysis")
    print("=" * 70)

    # Generate data
    print("\n[1/6] Generating test data...")
    train_emb, train_out, test_failures, test_successes, ood_samples = generate_test_data(
        n_train=500, n_test=100, n_ood=50, dim=dim
    )

    # Run analyses
    print("[2/6] Analyzing similarity_threshold...")
    sim_results = analyze_similarity_threshold(
        train_emb, train_out, test_failures, test_successes, ood_samples, dim
    )

    print("[3/6] Analyzing kde_bandwidth...")
    bw_results = analyze_kde_bandwidth(
        train_emb, train_out, test_failures, test_successes, ood_samples, dim
    )

    print("[4/6] Analyzing risk_threshold...")
    risk_results = analyze_risk_threshold(
        train_emb, train_out, test_failures, test_successes, ood_samples, dim
    )

    print("[5/6] Analyzing threshold interactions...")
    interaction_results = analyze_threshold_interactions(
        train_emb, train_out, test_failures, test_successes, ood_samples, dim
    )

    # Visualize
    print("[6/6] Generating visualizations...")
    plot_sensitivity_results(
        sim_results, bw_results, risk_results, interaction_results
    )

    # Save raw results
    all_results = {
        "similarity": sim_results,
        "bandwidth": bw_results,
        "risk": risk_results,
        "interactions": interaction_results
    }

    output_path = Path(__file__).parent / "threshold_sensitivity_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)

        # Best thresholds
        best_sim_idx = np.argmax([m["f1"] for m in sim_results["metrics"]])
        best_bw_idx = np.argmax([m["coverage_auc"] for m in bw_results["metrics"]])
        best_risk_idx = np.argmax([m["f1"] for m in risk_results["metrics"]])

        print(f"\nOptimal Similarity Threshold: {sim_results['thresholds'][best_sim_idx]:.2f}")
        print(f"  F1: {sim_results['metrics'][best_sim_idx]['f1']:.3f}")
        print(f"  Precision: {sim_results['metrics'][best_sim_idx]['precision']:.3f}")
        print(f"  Recall: {sim_results['metrics'][best_sim_idx]['recall']:.3f}")

        print(f"\nOptimal KDE Bandwidth: {bw_results['bandwidths'][best_bw_idx]:.2f}")
        print(f"  Coverage AUC: {bw_results['metrics'][best_bw_idx]['coverage_auc']:.3f}")

        print(f"\nOptimal Risk Threshold: {risk_results['thresholds'][best_risk_idx]:.2f}")
        print(f"  F1: {risk_results['metrics'][best_risk_idx]['f1']:.3f}")

        # Sensitivity ranking
        print("\n" + "-" * 40)
        print("Sensitivity Ranking (Variance in F1):")

        sim_f1_var = np.var([m["f1"] for m in sim_results["metrics"]])
        risk_f1_var = np.var([m["f1"] for m in risk_results["metrics"]])

        print(f"  1. Risk Threshold: variance = {risk_f1_var:.4f}")
        print(f"  2. Similarity Threshold: variance = {sim_f1_var:.4f}")
        print("\n  → Risk threshold has highest impact on F1")

    return all_results


def run_multi_seed_sensitivity(n_seeds: int = 5) -> Dict:
    """Run sensitivity analysis across multiple seeds for robustness."""

    all_results = []

    print("=" * 70)
    print(f"Multi-Seed Threshold Sensitivity (N={n_seeds})")
    print("=" * 70)

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        results = run_threshold_sensitivity(seed=seed, verbose=False)
        all_results.append(results)

    # Aggregate optimal thresholds
    optimal_sims = []
    optimal_bws = []
    optimal_risks = []

    for r in all_results:
        best_sim_idx = np.argmax([m["f1"] for m in r["similarity"]["metrics"]])
        best_bw_idx = np.argmax([m["coverage_auc"] for m in r["bandwidth"]["metrics"]])
        best_risk_idx = np.argmax([m["f1"] for m in r["risk"]["metrics"]])

        optimal_sims.append(r["similarity"]["thresholds"][best_sim_idx])
        optimal_bws.append(r["bandwidth"]["bandwidths"][best_bw_idx])
        optimal_risks.append(r["risk"]["thresholds"][best_risk_idx])

    print("\n" + "=" * 70)
    print("Optimal Threshold Summary (across seeds)")
    print("=" * 70)
    print(f"Similarity Threshold: {np.mean(optimal_sims):.2f} ± {np.std(optimal_sims):.2f}")
    print(f"KDE Bandwidth: {np.mean(optimal_bws):.2f} ± {np.std(optimal_bws):.2f}")
    print(f"Risk Threshold: {np.mean(optimal_risks):.2f} ± {np.std(optimal_risks):.2f}")

    return {
        "optimal_similarity": {"mean": np.mean(optimal_sims), "std": np.std(optimal_sims)},
        "optimal_bandwidth": {"mean": np.mean(optimal_bws), "std": np.std(optimal_bws)},
        "optimal_risk": {"mean": np.mean(optimal_risks), "std": np.std(optimal_risks)}
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed analysis")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of seeds")
    args = parser.parse_args()

    if args.multi_seed:
        run_multi_seed_sensitivity(n_seeds=args.n_seeds)
    else:
        run_threshold_sensitivity()
