"""
ALFM-BEM Experiments
====================

Comprehensive experimental evaluation of the ALFM-BEM architecture:
1. BEM Retrieval: Precision-recall for failure/success retrieval
2. OOD Detection: Coverage signal as distribution shift detector
3. Adapter Stability: Bounded drift under experience replay
4. End-to-End: Complete system evaluation on synthetic deployment

Author: David Ahmann
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, Experience, CoverageMode
from projection import ContrastiveProjection
from adapters import BoundedAdapter, AdapterConfig
from consensus import ConsensusEngine, Action

np.random.seed(42)


# =============================================================================
# Synthetic Data Generation
# =============================================================================

# Global shared failure modes (initialized once per run)
_SHARED_FAILURE_MODES = None


def generate_failure_modes(n_modes: int = 10, dim: int = 64) -> np.ndarray:
    """Generate cluster centroids representing distinct failure modes."""
    modes = np.random.randn(n_modes, dim)
    modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    return modes


def get_shared_failure_modes(n_modes: int, dim: int) -> np.ndarray:
    """Get or create shared failure modes for consistent train/test split."""
    global _SHARED_FAILURE_MODES
    if _SHARED_FAILURE_MODES is None or _SHARED_FAILURE_MODES.shape != (n_modes, dim):
        _SHARED_FAILURE_MODES = generate_failure_modes(n_modes, dim)
    return _SHARED_FAILURE_MODES


def generate_experiences(
    n_failures: int = 500,
    n_successes: int = 500,
    dim: int = 64,
    noise_level: float = 0.05,
    n_failure_modes: int = 10,
    failure_modes: np.ndarray = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate synthetic experiences with clustered failures and uniform successes.
    
    Failures cluster around failure modes (simulating recurring patterns).
    Successes are more uniformly distributed (diverse successful approaches).
    
    Args:
        failure_modes: If provided, use these modes. Otherwise generate new ones.
    """
    if failure_modes is None:
        failure_modes = get_shared_failure_modes(n_failure_modes, dim)
    
    embeddings = []
    outcomes = []
    
    # Generate failures clustered around modes
    for _ in range(n_failures):
        mode_idx = np.random.randint(len(failure_modes))
        noise = np.random.randn(dim) * noise_level
        vec = failure_modes[mode_idx] + noise
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)
        outcomes.append(np.random.uniform(-1.0, -0.3))  # Failure outcome
    
    # Generate successes (more uniform)
    for _ in range(n_successes):
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)
        outcomes.append(np.random.uniform(0.3, 1.0))  # Success outcome
    
    return embeddings, outcomes


def generate_ood_samples(
    n_samples: int = 200,
    dim: int = 64,
    shift_magnitude: float = 3.0
) -> List[np.ndarray]:
    """
    Generate out-of-distribution samples.
    
    OOD samples cluster around a novel centroid, representing inputs
    from a deployment domain not seen during training.
    """
    # Create a strong directional shift (tight cluster in a new region)
    shift_direction = np.random.randn(dim)
    shift_direction = shift_magnitude * shift_direction / np.linalg.norm(shift_direction)
    
    samples = []
    for _ in range(n_samples):
        # OOD samples are tightly clustered around the shifted direction
        noise = np.random.randn(dim) * 0.1
        vec = shift_direction + noise
        vec = vec / np.linalg.norm(vec)
        samples.append(vec)
    
    return samples


# =============================================================================
# Experiment 1: BEM Retrieval Precision-Recall
# =============================================================================

def experiment_bem_retrieval(
    bem: BidirectionalExperienceMemory,
    test_failures: List[np.ndarray],
    test_successes: List[np.ndarray],
    thresholds: np.ndarray = np.linspace(0.0, 1.0, 50)
) -> Dict:
    """
    Evaluate BEM's ability to retrieve relevant failures and successes.
    
    Returns precision-recall curves for both.
    """
    # Test failure retrieval
    failure_precisions = []
    failure_recalls = []
    
    for thresh in thresholds:
        bem.similarity_threshold = thresh
        tp, fp, fn = 0, 0, 0
        
        for z in test_failures:
            risk, retrieved = bem.risk_signal(z)
            if len(retrieved) > 0:
                tp += 1
            else:
                fn += 1
        
        for z in test_successes:
            risk, retrieved = bem.risk_signal(z)
            if len(retrieved) > 0:
                fp += 1
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        failure_precisions.append(precision)
        failure_recalls.append(recall)
    
    # Test success retrieval
    success_precisions = []
    success_recalls = []
    
    for thresh in thresholds:
        bem.similarity_threshold = thresh
        tp, fp, fn = 0, 0, 0
        
        for z in test_successes:
            success, retrieved = bem.success_signal(z)
            if len(retrieved) > 0:
                tp += 1
            else:
                fn += 1
        
        for z in test_failures:
            success, retrieved = bem.success_signal(z)
            if len(retrieved) > 0:
                fp += 1
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        success_precisions.append(precision)
        success_recalls.append(recall)
    
    return {
        "failure_precision": failure_precisions,
        "failure_recall": failure_recalls,
        "success_precision": success_precisions,
        "success_recall": success_recalls,
        "thresholds": thresholds.tolist()
    }


# =============================================================================
# Experiment 2: OOD Detection via Coverage Signal
# =============================================================================

def experiment_ood_detection(
    bem: BidirectionalExperienceMemory,
    in_distribution: List[np.ndarray],
    out_of_distribution: List[np.ndarray]
) -> Dict:
    """
    Evaluate coverage signal as OOD detector.
    
    Returns ROC curve data and AUC.
    """
    # Compute coverage for in-distribution
    id_coverage = [bem.coverage_signal(z) for z in in_distribution]
    
    # Compute coverage for OOD
    ood_coverage = [bem.coverage_signal(z) for z in out_of_distribution]
    
    # ROC curve: coverage as detector (low coverage = OOD)
    all_coverage = id_coverage + ood_coverage
    all_labels = [1] * len(id_coverage) + [0] * len(ood_coverage)  # 1 = ID, 0 = OOD
    
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    
    for thresh in thresholds:
        # Predict ID if coverage > thresh
        tp = sum(1 for c, l in zip(all_coverage, all_labels) if c > thresh and l == 1)
        fn = sum(1 for c, l in zip(all_coverage, all_labels) if c <= thresh and l == 1)
        fp = sum(1 for c, l in zip(all_coverage, all_labels) if c > thresh and l == 0)
        tn = sum(1 for c, l in zip(all_coverage, all_labels) if c <= thresh and l == 0)
        
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        tprs.append(tpr)
        fprs.append(fpr)
    
    # Compute AUC (trapezoidal)
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += 0.5 * (fprs[i-1] - fprs[i]) * (tprs[i-1] + tprs[i])
    
    return {
        "tpr": tprs,
        "fpr": fprs,
        "auc": auc,
        "id_coverage_mean": np.mean(id_coverage),
        "ood_coverage_mean": np.mean(ood_coverage)
    }


# =============================================================================
# Experiment 3: Adapter Stability Under Experience Replay
# =============================================================================

def experiment_adapter_stability(
    n_steps: int = 500,
    batch_size: int = 32,
    dim: int = 64
) -> Dict:
    """
    Compare bounded vs unbounded adapter updates.
    
    Measures parameter drift over training steps.
    """
    # Generate experiences
    embeddings, outcomes = generate_experiences(n_failures=500, n_successes=500, dim=dim, noise_level=0.05, n_failure_modes=10)
    
    # Create BEM and populate
    bem = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.7)
    for i, (emb, out) in enumerate(zip(embeddings, outcomes)):
        bem.add_experience(emb, out, f"context_{i}")
    
    # Bounded adapter - strict constraints
    config_bounded = AdapterConfig(
        input_dim=dim,
        hidden_dim=64,
        output_dim=dim,
        max_grad_norm=0.5,      # Tight gradient clipping
        max_param_norm=5.0,     # Tight parameter norm bound
        learning_rate=1e-2      # Higher LR to show effect of constraints
    )
    adapter_bounded = BoundedAdapter(config_bounded)
    
    # Unbounded adapter (loose limits to show unbounded behavior)
    config_unbounded = AdapterConfig(
        input_dim=dim,
        hidden_dim=64,
        output_dim=dim,
        max_grad_norm=1000.0,   # Effectively unbounded
        max_param_norm=10000.0, # Effectively unbounded
        learning_rate=1e-2      # Same LR for fair comparison
    )
    adapter_unbounded = BoundedAdapter(config_unbounded)
    
    bounded_drift = []
    unbounded_drift = []
    
    for step in range(n_steps):
        experiences = bem.sample_for_training(batch_size)
        
        # Train bounded
        adapter_bounded.train_step(experiences, None)
        bounded_drift.append(adapter_bounded.total_drift)
        
        # Train unbounded
        adapter_unbounded.train_step(experiences, None)
        unbounded_drift.append(adapter_unbounded.total_drift)
    
    return {
        "steps": list(range(n_steps)),
        "bounded_drift": bounded_drift,
        "unbounded_drift": unbounded_drift,
        "bounded_final_norm": adapter_bounded._compute_param_norm(),
        "unbounded_final_norm": adapter_unbounded._compute_param_norm()
    }


# =============================================================================
# Experiment 4: End-to-End Consensus Engine Behavior
# =============================================================================

def experiment_consensus_behavior(
    bem: BidirectionalExperienceMemory,
    test_samples: List[np.ndarray],
    test_labels: List[str]  # "failure", "success", "ood"
) -> Dict:
    """
    Evaluate Consensus Engine action distribution.
    """
    consensus = ConsensusEngine()
    
    action_counts = {action: 0 for action in Action}
    correct_actions = 0
    
    results = []
    for z, label in zip(test_samples, test_labels):
        risk, _ = bem.risk_signal(z)
        success, _ = bem.success_signal(z)
        coverage = bem.coverage_signal(z)
        
        decision = consensus.decide(risk, success, coverage, {})
        action_counts[decision.action] += 1
        
        # Check if action is "correct"
        if label == "failure" and decision.action in [Action.ABSTAIN, Action.ESCALATE]:
            correct_actions += 1
        elif label == "success" and decision.action == Action.TRUST:
            correct_actions += 1
        elif label == "ood" and decision.action == Action.QUERY:
            correct_actions += 1
        
        results.append({
            "label": label,
            "action": decision.action.value,
            "risk": risk,
            "success": success,
            "coverage": coverage
        })
    
    return {
        "action_distribution": {k.value: v for k, v in action_counts.items()},
        "accuracy": correct_actions / len(test_samples),
        "results": results
    }


# =============================================================================
# Run All Experiments
# =============================================================================

def run_all_experiments(save_dir: str = "../data"):
    """Run all experiments and save results."""
    import json
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("ALFM-BEM Experimental Evaluation")
    print("=" * 60)
    
    dim = 64  # Use 64D for effective similarity-based retrieval
    
    # Generate data - use 64D for effective cosine similarity retrieval
    print("\n[1/5] Generating synthetic data...")
    train_emb, train_out = generate_experiences(n_failures=500, n_successes=500, dim=dim, noise_level=0.05, n_failure_modes=10)
    test_emb, test_out = generate_experiences(n_failures=100, n_successes=100, dim=dim, noise_level=0.05, n_failure_modes=10)
    ood_samples = generate_ood_samples(n_samples=200, dim=dim)
    
    # Create and populate BEM with KDE coverage mode for better OOD detection
    print("[2/5] Populating BEM...")
    bem = BidirectionalExperienceMemory(
        dim=dim, 
        similarity_threshold=0.7,
        coverage_mode=CoverageMode.KDE,
        kde_bandwidth=0.3
    )
    for i, (emb, out) in enumerate(zip(train_emb, train_out)):
        bem.add_experience(emb, out, f"train_context_{i}")
    
    print(f"      BEM stats: {bem.get_statistics()}")
    
    # Experiment 1: Retrieval
    print("\n[3/5] Running BEM retrieval experiment...")
    test_failures = [emb for emb, out in zip(test_emb, test_out) if out < 0]
    test_successes = [emb for emb, out in zip(test_emb, test_out) if out > 0]
    retrieval_results = experiment_bem_retrieval(bem, test_failures, test_successes)
    
    # Compute F1 at optimal threshold
    f1_scores = [
        2 * p * r / (p + r + 1e-10) 
        for p, r in zip(retrieval_results["failure_precision"], retrieval_results["failure_recall"])
    ]
    best_f1 = max(f1_scores)
    print(f"      Best Failure Retrieval F1: {best_f1:.3f}")
    
    # Experiment 2: OOD Detection
    # For OOD detection, use only clustered failures as ID (high coverage expected)
    # This tests the core question: can coverage detect novel failure patterns?
    print("[4/5] Running OOD detection experiment...")
    # Generate fresh ID samples from same failure modes (to avoid data leakage)
    shared_modes = get_shared_failure_modes(10, dim)
    id_failures = []
    for _ in range(200):
        mode_idx = np.random.randint(len(shared_modes))
        noise = np.random.randn(dim) * 0.05
        vec = shared_modes[mode_idx] + noise
        vec = vec / np.linalg.norm(vec)
        id_failures.append(vec)
    ood_results = experiment_ood_detection(bem, id_failures, ood_samples)
    print(f"      OOD Detection AUC: {ood_results['auc']:.3f}")
    print(f"      ID Coverage Mean: {ood_results['id_coverage_mean']:.3f}")
    print(f"      OOD Coverage Mean: {ood_results['ood_coverage_mean']:.3f}")
    
    # Experiment 3: Adapter Stability
    print("[5/5] Running adapter stability experiment...")
    stability_results = experiment_adapter_stability(n_steps=500, dim=dim)
    print(f"      Bounded Final Drift: {stability_results['bounded_drift'][-1]:.3f}")
    print(f"      Unbounded Final Drift: {stability_results['unbounded_drift'][-1]:.3f}")
    
    # Save results
    results = {
        "bem_stats": bem.get_statistics(),
        "retrieval": retrieval_results,
        "ood_detection": ood_results,
        "adapter_stability": stability_results
    }
    
    with open(f"{save_dir}/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {save_dir}/experiment_results.json")
    
    # Generate plots
    generate_plots(results, save_dir)
    
    return results


def generate_plots(results: Dict, save_dir: str):
    """Generate publication-quality plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Precision-Recall for failure retrieval
    ax1 = axes[0, 0]
    ax1.plot(results["retrieval"]["failure_recall"], 
             results["retrieval"]["failure_precision"], 
             'b-', linewidth=2, label='Failure Retrieval')
    ax1.plot(results["retrieval"]["success_recall"],
             results["retrieval"]["success_precision"],
             'g--', linewidth=2, label='Success Retrieval')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('BEM Retrieval Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ROC for OOD detection
    ax2 = axes[0, 1]
    ax2.plot(results["ood_detection"]["fpr"], 
             results["ood_detection"]["tpr"],
             'r-', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'OOD Detection (AUC = {results["ood_detection"]["auc"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Adapter drift
    ax3 = axes[1, 0]
    ax3.plot(results["adapter_stability"]["steps"],
             results["adapter_stability"]["bounded_drift"],
             'b-', linewidth=2, label='Bounded (ALFM-BEM)')
    ax3.plot(results["adapter_stability"]["steps"],
             results["adapter_stability"]["unbounded_drift"],
             'r--', linewidth=2, label='Unbounded')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Cumulative Parameter Drift')
    ax3.set_title('Adapter Stability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: BEM statistics
    ax4 = axes[1, 1]
    stats = results["bem_stats"]
    labels = ['Failures', 'Successes', 'Neutral']
    sizes = [stats.get('failures', 0), stats.get('successes', 0), stats.get('neutral', 0)]
    colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('BEM Experience Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/alfm_bem_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/alfm_bem_results.png")


if __name__ == "__main__":
    results = run_all_experiments()
