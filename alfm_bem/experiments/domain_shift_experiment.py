"""
Domain Shift Experiment with Bounded Adapters
==============================================

A rigorous evaluation of BoundedAdapter's ability to handle domain shift.

Scenarios:
1. Gradual Drift: Slow rotation of success/failure regions
2. Sudden Shift: Abrupt change in decision boundary
3. Multi-Domain: Different domains with shared structure
4. Few-Shot Adaptation: Minimal data in new domain

Comparison:
- Fixed BEM (no adapter)
- BEM + Adapter (online learning)
- BEM + Adapter + Replay (experience replay from old domain)

Metrics:
- Adaptation Delay: Steps to recover 90% of original performance
- Worst-Case Accuracy: Minimum rolling accuracy during shift
- Final Accuracy: Accuracy after stabilization
- Forgetting: Performance drop on original domain

Author: David Ahmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode, Experience
from adapters import BoundedAdapter, AdapterConfig

np.random.seed(42)


# =============================================================================
# Data Generation
# =============================================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return vec / (np.linalg.norm(vec) + 1e-10)


@dataclass
class DomainSpec:
    """Specification for a domain's decision boundary."""
    name: str
    success_direction: np.ndarray  # Direction where y > 0
    noise_scale: float = 0.1


def generate_domain_data(
    domain: DomainSpec,
    n_samples: int,
    dim: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for a specific domain."""
    embeddings = np.random.randn(n_samples, dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Project onto success direction
    projections = embeddings @ domain.success_direction
    noise = np.random.randn(n_samples) * domain.noise_scale

    # Outcome based on projection + noise
    outcomes = np.sign(projections + noise)

    return embeddings, outcomes


def create_rotating_domains(
    n_domains: int,
    dim: int = 64,
    rotation_angle: float = np.pi / 4
) -> List[DomainSpec]:
    """Create domains with progressively rotated decision boundaries."""
    base_direction = np.zeros(dim)
    base_direction[0] = 1.0  # Start aligned with first axis

    domains = []
    for i in range(n_domains):
        # Create rotation in 0-1 plane
        angle = i * rotation_angle
        direction = base_direction.copy()
        direction[0] = np.cos(angle)
        direction[1] = np.sin(angle)
        direction = normalize(direction)

        domains.append(DomainSpec(
            name=f"Domain_{i}",
            success_direction=direction,
            noise_scale=0.1
        ))

    return domains


def create_sudden_shift_domains(dim: int = 64) -> Tuple[DomainSpec, DomainSpec]:
    """Create two domains with orthogonal decision boundaries (sudden shift)."""
    d1 = np.zeros(dim)
    d1[0] = 1.0

    d2 = np.zeros(dim)
    d2[1] = 1.0  # Orthogonal

    return (
        DomainSpec(name="Domain_A", success_direction=d1, noise_scale=0.1),
        DomainSpec(name="Domain_B", success_direction=d2, noise_scale=0.1)
    )


# =============================================================================
# Evaluation Helpers
# =============================================================================

def predict_outcome(
    bem: BidirectionalExperienceMemory,
    z: np.ndarray,
    adapter: Optional[BoundedAdapter] = None
) -> float:
    """Predict outcome using BEM (optionally with adapter)."""
    if adapter is not None:
        z = adapter(z)

    risk, _ = bem.risk_signal(z)
    success, _ = bem.success_signal(z)

    return 1.0 if success > risk else -1.0


def compute_accuracy(
    predictions: List[float],
    targets: List[float],
    window: int = 50
) -> List[float]:
    """Compute rolling accuracy."""
    accuracies = []
    for i in range(len(predictions)):
        start = max(0, i - window + 1)
        preds = predictions[start:i+1]
        targs = targets[start:i+1]
        acc = sum(1 for p, t in zip(preds, targs) if np.sign(p) == np.sign(t)) / len(preds)
        accuracies.append(acc)
    return accuracies


# =============================================================================
# Experiment 1: Gradual Domain Drift
# =============================================================================

def run_gradual_drift_experiment(
    dim: int = 64,
    n_steps: int = 2000,
    n_rotations: int = 4,
    verbose: bool = True
) -> Dict:
    """
    Gradual drift: decision boundary slowly rotates.

    Compare:
    - Fixed: No adaptation
    - Adaptive: Online adapter training
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Gradual Domain Drift")
    print("=" * 70)

    # Create rotating domains
    domains = create_rotating_domains(n_rotations, dim, rotation_angle=np.pi / 6)
    steps_per_domain = n_steps // n_rotations

    # Initialize systems
    bem_fixed = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)
    bem_adaptive = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)

    adapter_config = AdapterConfig(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=32,
        learning_rate=0.01,
        max_grad_norm=1.0
    )
    adapter = BoundedAdapter(adapter_config, tenant_id="drift_test")

    # Track predictions
    fixed_preds, adaptive_preds = [], []
    targets = []
    domain_boundaries = []

    current_domain_idx = 0

    for step in range(n_steps):
        # Determine current domain (with gradual transition)
        domain_idx = min(step // steps_per_domain, n_rotations - 1)

        if domain_idx != current_domain_idx:
            domain_boundaries.append(step)
            current_domain_idx = domain_idx
            if verbose:
                print(f"Step {step}: Transitioning to {domains[domain_idx].name}")

        domain = domains[domain_idx]

        # Generate sample
        z, y = generate_domain_data(domain, n_samples=1, dim=dim)
        z, y = z[0], y[0]

        # Fixed system prediction
        pred_fixed = predict_outcome(bem_fixed, z)
        fixed_preds.append(pred_fixed)

        # Adaptive system prediction
        pred_adaptive = predict_outcome(bem_adaptive, z, adapter)
        adaptive_preds.append(pred_adaptive)

        targets.append(y)

        # Learning
        bem_fixed.add_experience(z, y, f"gradual_{step}")
        bem_adaptive.add_experience(z, y, f"gradual_{step}")

        # Adapter training (every 10 steps)
        if step % 10 == 0 and len(bem_adaptive.experiences) > 5:
            # Sample recent experiences
            recent = bem_adaptive.experiences[-min(20, len(bem_adaptive.experiences)):]
            adapter.train_step(recent)

    # Compute accuracies
    fixed_acc = compute_accuracy(fixed_preds, targets, window=50)
    adaptive_acc = compute_accuracy(adaptive_preds, targets, window=50)

    # Metrics
    results = {
        "fixed_mean_acc": np.mean(fixed_acc[100:]),  # Skip warmup
        "adaptive_mean_acc": np.mean(adaptive_acc[100:]),
        "fixed_worst_acc": np.min(fixed_acc[100:]),
        "adaptive_worst_acc": np.min(adaptive_acc[100:]),
        "domain_boundaries": domain_boundaries,
        "fixed_acc": fixed_acc,
        "adaptive_acc": adaptive_acc
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Fixed Mean Accuracy: {results['fixed_mean_acc']:.3f}")
        print(f"  Adaptive Mean Accuracy: {results['adaptive_mean_acc']:.3f}")
        print(f"  Improvement: {(results['adaptive_mean_acc'] - results['fixed_mean_acc']) * 100:.1f}%")

    return results


# =============================================================================
# Experiment 2: Sudden Domain Shift
# =============================================================================

def run_sudden_shift_experiment(
    dim: int = 64,
    n_pretrain: int = 500,
    n_postshift: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Sudden shift: abrupt change from Domain A to Domain B.

    Measures adaptation delay after the shift.
    """
    print("\n" + "=" * 70)
    print("Experiment 2: Sudden Domain Shift")
    print("=" * 70)

    domain_a, domain_b = create_sudden_shift_domains(dim)

    # Pre-training phase on Domain A
    if verbose:
        print("Phase 1: Pre-training on Domain A...")

    bem_fixed = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)
    bem_adaptive = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)

    adapter_config = AdapterConfig(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=32,
        learning_rate=0.05,  # Higher LR for faster adaptation
        max_grad_norm=5.0
    )
    adapter = BoundedAdapter(adapter_config, tenant_id="sudden_shift")

    # Pre-train on Domain A
    pretrain_emb, pretrain_out = generate_domain_data(domain_a, n_pretrain, dim)
    for i, (z, y) in enumerate(zip(pretrain_emb, pretrain_out)):
        bem_fixed.add_experience(z, y, f"pretrain_{i}")
        bem_adaptive.add_experience(z, y, f"pretrain_{i}")

    # Evaluate on Domain A (baseline)
    test_a_emb, test_a_out = generate_domain_data(domain_a, 100, dim)
    baseline_acc = sum(
        1 for z, y in zip(test_a_emb, test_a_out)
        if np.sign(predict_outcome(bem_fixed, z)) == np.sign(y)
    ) / 100

    if verbose:
        print(f"  Baseline accuracy on Domain A: {baseline_acc:.3f}")

    # Shift to Domain B
    if verbose:
        print("\nPhase 2: Sudden shift to Domain B...")

    fixed_preds, adaptive_preds = [], []
    targets = []

    for step in range(n_postshift):
        z, y = generate_domain_data(domain_b, n_samples=1, dim=dim)
        z, y = z[0], y[0]

        # Predictions
        pred_fixed = predict_outcome(bem_fixed, z)
        pred_adaptive = predict_outcome(bem_adaptive, z, adapter)

        fixed_preds.append(pred_fixed)
        adaptive_preds.append(pred_adaptive)
        targets.append(y)

        # Learning (sparse: every 20 steps to simulate few-shot)
        if step % 20 == 0:
            bem_fixed.add_experience(z, y, f"postshift_{step}")
            bem_adaptive.add_experience(z, y, f"postshift_{step}")

            # Train adapter on new data
            exp = Experience(
                embedding=z,
                outcome=y,
                context_hash=f"postshift_{step}",
                timestamp=datetime.now(),
                tenant_id="test",
                domain_id="domain_b"
            )
            adapter.train_step([exp])

    # Compute rolling accuracy
    fixed_acc = compute_accuracy(fixed_preds, targets, window=50)
    adaptive_acc = compute_accuracy(adaptive_preds, targets, window=50)

    # Find adaptation delay (steps to reach 0.9 * baseline)
    target_acc = 0.9 * baseline_acc

    def find_recovery_step(acc_list, target):
        for i, acc in enumerate(acc_list):
            if acc >= target:
                return i
        return len(acc_list)  # Never recovered

    fixed_recovery = find_recovery_step(fixed_acc, target_acc)
    adaptive_recovery = find_recovery_step(adaptive_acc, target_acc)

    results = {
        "baseline_acc": baseline_acc,
        "fixed_final_acc": fixed_acc[-1] if fixed_acc else 0,
        "adaptive_final_acc": adaptive_acc[-1] if adaptive_acc else 0,
        "fixed_recovery_steps": fixed_recovery,
        "adaptive_recovery_steps": adaptive_recovery,
        "fixed_worst_acc": min(fixed_acc) if fixed_acc else 0,
        "adaptive_worst_acc": min(adaptive_acc) if adaptive_acc else 0,
        "fixed_acc": fixed_acc,
        "adaptive_acc": adaptive_acc
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Fixed Recovery Steps: {fixed_recovery} (Final: {results['fixed_final_acc']:.3f})")
        print(f"  Adaptive Recovery Steps: {adaptive_recovery} (Final: {results['adaptive_final_acc']:.3f})")
        print(f"  Adaptation Speedup: {fixed_recovery / (adaptive_recovery + 1):.1f}x")

    return results


# =============================================================================
# Experiment 3: Few-Shot Domain Adaptation
# =============================================================================

def run_few_shot_experiment(
    dim: int = 64,
    n_source: int = 500,
    n_target_shots: List[int] = [1, 5, 10, 20, 50],
    verbose: bool = True
) -> Dict:
    """
    Few-shot adaptation: how well can adapter adapt with minimal target data?
    """
    print("\n" + "=" * 70)
    print("Experiment 3: Few-Shot Domain Adaptation")
    print("=" * 70)

    domain_a, domain_b = create_sudden_shift_domains(dim)

    results = {"n_shots": n_target_shots, "fixed_acc": [], "adaptive_acc": []}

    for n_shots in n_target_shots:
        if verbose:
            print(f"\n  Testing with {n_shots} target shots...")

        # Initialize fresh systems
        bem_fixed = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)
        bem_adaptive = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)

        adapter_config = AdapterConfig(
            input_dim=dim,
            output_dim=dim,
            hidden_dim=32,
            learning_rate=0.1,  # High LR for few-shot
            max_grad_norm=5.0
        )
        adapter = BoundedAdapter(adapter_config, tenant_id="few_shot")

        # Pre-train on source domain
        source_emb, source_out = generate_domain_data(domain_a, n_source, dim)
        for i, (z, y) in enumerate(zip(source_emb, source_out)):
            bem_fixed.add_experience(z, y, f"source_{i}")
            bem_adaptive.add_experience(z, y, f"source_{i}")

        # Few-shot adaptation on target domain
        target_emb, target_out = generate_domain_data(domain_b, n_shots, dim)
        for i, (z, y) in enumerate(zip(target_emb, target_out)):
            bem_fixed.add_experience(z, y, f"target_{i}")
            bem_adaptive.add_experience(z, y, f"target_{i}")

            # Train adapter
            exp = Experience(
                embedding=z,
                outcome=y,
                context_hash=f"target_{i}",
                timestamp=datetime.now(),
                tenant_id="test",
                domain_id="target"
            )
            adapter.train_step([exp])

        # Evaluate on target domain
        test_emb, test_out = generate_domain_data(domain_b, 100, dim)

        fixed_correct = sum(
            1 for z, y in zip(test_emb, test_out)
            if np.sign(predict_outcome(bem_fixed, z)) == np.sign(y)
        )
        adaptive_correct = sum(
            1 for z, y in zip(test_emb, test_out)
            if np.sign(predict_outcome(bem_adaptive, z, adapter)) == np.sign(y)
        )

        results["fixed_acc"].append(fixed_correct / 100)
        results["adaptive_acc"].append(adaptive_correct / 100)

        if verbose:
            print(f"    Fixed: {fixed_correct / 100:.3f}, Adaptive: {adaptive_correct / 100:.3f}")

    return results


# =============================================================================
# Experiment 4: Forgetting Analysis
# =============================================================================

def run_forgetting_experiment(
    dim: int = 64,
    n_per_domain: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Test if adapter causes forgetting on original domain.
    """
    print("\n" + "=" * 70)
    print("Experiment 4: Catastrophic Forgetting Analysis")
    print("=" * 70)

    domain_a, domain_b = create_sudden_shift_domains(dim)

    # Initialize
    bem = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.3)

    adapter_config = AdapterConfig(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=32,
        learning_rate=0.05,
        max_grad_norm=1.0
    )
    adapter = BoundedAdapter(adapter_config, tenant_id="forgetting_test")

    # Phase 1: Train on Domain A
    if verbose:
        print("Phase 1: Training on Domain A...")

    domain_a_emb, domain_a_out = generate_domain_data(domain_a, n_per_domain, dim)
    for i, (z, y) in enumerate(zip(domain_a_emb, domain_a_out)):
        bem.add_experience(z, y, f"domain_a_{i}")

    # Evaluate on Domain A (pre-adaptation baseline)
    test_a_emb, test_a_out = generate_domain_data(domain_a, 100, dim)
    pre_adapt_acc_a = sum(
        1 for z, y in zip(test_a_emb, test_a_out)
        if np.sign(predict_outcome(bem, z, adapter)) == np.sign(y)
    ) / 100

    if verbose:
        print(f"  Pre-adaptation accuracy on Domain A: {pre_adapt_acc_a:.3f}")

    # Phase 2: Adapt to Domain B
    if verbose:
        print("\nPhase 2: Adapting to Domain B...")

    domain_b_emb, domain_b_out = generate_domain_data(domain_b, n_per_domain, dim)
    for i, (z, y) in enumerate(zip(domain_b_emb, domain_b_out)):
        bem.add_experience(z, y, f"domain_b_{i}")

        # Train adapter
        exp = Experience(
            embedding=z,
            outcome=y,
            context_hash=f"domain_b_{i}",
            timestamp=datetime.now(),
            tenant_id="test",
            domain_id="domain_b"
        )
        adapter.train_step([exp])

    # Evaluate on both domains after adaptation
    post_adapt_acc_a = sum(
        1 for z, y in zip(test_a_emb, test_a_out)
        if np.sign(predict_outcome(bem, z, adapter)) == np.sign(y)
    ) / 100

    test_b_emb, test_b_out = generate_domain_data(domain_b, 100, dim)
    post_adapt_acc_b = sum(
        1 for z, y in zip(test_b_emb, test_b_out)
        if np.sign(predict_outcome(bem, z, adapter)) == np.sign(y)
    ) / 100

    forgetting = pre_adapt_acc_a - post_adapt_acc_a

    results = {
        "pre_adapt_acc_a": pre_adapt_acc_a,
        "post_adapt_acc_a": post_adapt_acc_a,
        "post_adapt_acc_b": post_adapt_acc_b,
        "forgetting": forgetting,
        "adapter_stats": adapter.get_statistics()
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Pre-adaptation on A: {pre_adapt_acc_a:.3f}")
        print(f"  Post-adaptation on A: {post_adapt_acc_a:.3f}")
        print(f"  Post-adaptation on B: {post_adapt_acc_b:.3f}")
        print(f"  Forgetting (A): {forgetting:.3f}")
        print(f"  Adapter total drift: {results['adapter_stats']['total_drift']:.4f}")

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_all_experiments(
    gradual_results: Dict,
    sudden_results: Dict,
    fewshot_results: Dict,
    forgetting_results: Dict,
    output_path: str = "domain_shift_results.pdf"
):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gradual Drift
    ax = axes[0, 0]
    steps = np.arange(len(gradual_results["fixed_acc"]))
    ax.plot(steps, gradual_results["fixed_acc"], 'b--', alpha=0.7, label='Fixed BEM')
    ax.plot(steps, gradual_results["adaptive_acc"], 'r-', linewidth=2, label='BEM + Adapter')

    for boundary in gradual_results.get("domain_boundaries", []):
        ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Step')
    ax.set_ylabel('Rolling Accuracy')
    ax.set_title('Gradual Domain Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sudden Shift
    ax = axes[0, 1]
    steps = np.arange(len(sudden_results["fixed_acc"]))
    ax.plot(steps, sudden_results["fixed_acc"], 'b--', alpha=0.7, label='Fixed BEM')
    ax.plot(steps, sudden_results["adaptive_acc"], 'r-', linewidth=2, label='BEM + Adapter')
    ax.axhline(sudden_results["baseline_acc"], color='green', linestyle=':', label='Baseline (pre-shift)')

    ax.set_xlabel('Steps After Shift')
    ax.set_ylabel('Rolling Accuracy')
    ax.set_title('Sudden Domain Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Few-Shot Adaptation
    ax = axes[1, 0]
    n_shots = fewshot_results["n_shots"]
    ax.plot(n_shots, fewshot_results["fixed_acc"], 'b--o', label='Fixed BEM')
    ax.plot(n_shots, fewshot_results["adaptive_acc"], 'r-o', linewidth=2, label='BEM + Adapter')
    ax.set_xlabel('Number of Target Shots')
    ax.set_ylabel('Target Domain Accuracy')
    ax.set_title('Few-Shot Domain Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 4. Forgetting Analysis
    ax = axes[1, 1]
    categories = ['Pre-Adapt\n(Domain A)', 'Post-Adapt\n(Domain A)', 'Post-Adapt\n(Domain B)']
    values = [
        forgetting_results["pre_adapt_acc_a"],
        forgetting_results["post_adapt_acc_a"],
        forgetting_results["post_adapt_acc_b"]
    ]
    colors = ['green', 'blue', 'orange']
    bars = ax.bar(categories, values, color=colors, alpha=0.7)

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Catastrophic Forgetting Analysis\n(Forgetting: {forgetting_results["forgetting"]:.3f})')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {output_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

def run_all_experiments(verbose: bool = True) -> Dict:
    """Run all domain shift experiments."""

    print("=" * 70)
    print("Domain Shift Experiments with Bounded Adapters")
    print("=" * 70)

    gradual = run_gradual_drift_experiment(verbose=verbose)
    sudden = run_sudden_shift_experiment(verbose=verbose)
    fewshot = run_few_shot_experiment(verbose=verbose)
    forgetting = run_forgetting_experiment(verbose=verbose)

    # Generate plots
    plot_all_experiments(gradual, sudden, fewshot, forgetting)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n1. Gradual Drift:")
    print(f"   Adaptive improvement: {(gradual['adaptive_mean_acc'] - gradual['fixed_mean_acc']) * 100:.1f}%")

    print(f"\n2. Sudden Shift:")
    print(f"   Recovery speedup: {gradual.get('fixed_recovery_steps', 1) / (sudden.get('adaptive_recovery_steps', 1) + 1):.1f}x")

    print(f"\n3. Few-Shot (5 shots):")
    idx = fewshot["n_shots"].index(5) if 5 in fewshot["n_shots"] else 1
    print(f"   Fixed: {fewshot['fixed_acc'][idx]:.3f}, Adaptive: {fewshot['adaptive_acc'][idx]:.3f}")

    print(f"\n4. Forgetting:")
    print(f"   Catastrophic forgetting: {forgetting['forgetting']:.3f}")

    # Save results
    all_results = {
        "gradual": {k: v for k, v in gradual.items() if k not in ['fixed_acc', 'adaptive_acc']},
        "sudden": {k: v for k, v in sudden.items() if k not in ['fixed_acc', 'adaptive_acc']},
        "fewshot": fewshot,
        "forgetting": {k: v for k, v in forgetting.items() if k != 'adapter_stats'}
    }

    output_path = Path(__file__).parent / "domain_shift_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "gradual", "sudden", "fewshot", "forgetting"])
    args = parser.parse_args()

    if args.experiment == "all":
        run_all_experiments()
    elif args.experiment == "gradual":
        run_gradual_drift_experiment()
    elif args.experiment == "sudden":
        run_sudden_shift_experiment()
    elif args.experiment == "fewshot":
        run_few_shot_experiment()
    elif args.experiment == "forgetting":
        run_forgetting_experiment()
