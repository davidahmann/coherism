"""
Ablation Study: BEM vs RAG vs NEP
=================================

Compares BEM against simpler baselines to demonstrate the value of each component:

1. Plain RAG: Retrieves similar experiences without outcome weighting
   - No distinction between failures and successes
   - No OOD detection

2. NEP (Negative Experience Prediction): Stores only failures
   - Binary failure memory (no continuous outcomes)
   - No success patterns
   - Max-similarity for coverage (not KDE)

3. BEM (Full): Unified bidirectional memory with KDE coverage
   - Continuous outcome spectrum
   - Success pattern retrieval
   - KDE-based OOD detection

Author: David Ahmann
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode

np.random.seed(42)


# =============================================================================
# Baseline Implementations
# =============================================================================

class PlainRAG:
    """
    Simple retrieval-augmented memory: retrieves similar items regardless of outcome.
    
    This is what you'd get with vanilla vector similarity search.
    """
    
    def __init__(self, dim: int = 64, similarity_threshold: float = 0.7):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.embeddings: List[np.ndarray] = []
        self.outcomes: List[float] = []
        self._matrix = None
    
    def add_experience(self, embedding: np.ndarray, outcome: float, context: str):
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        self.embeddings.append(embedding)
        self.outcomes.append(outcome)
        self._matrix = None  # Invalidate cache
    
    def _ensure_matrix(self):
        if self._matrix is None and self.embeddings:
            self._matrix = np.vstack(self.embeddings)
    
    def retrieve(self, z: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve k most similar experiences (no outcome filtering)."""
        self._ensure_matrix()
        if self._matrix is None:
            return []
        
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        sims = self._matrix @ z_norm
        
        # Return top-k regardless of outcome
        indices = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in indices if sims[i] > self.similarity_threshold]
    
    def risk_signal(self, z: np.ndarray) -> Tuple[float, List]:
        """RAG doesn't have a proper risk signal - just counts failures in neighbors."""
        neighbors = self.retrieve(z, k=10)
        if not neighbors:
            return 0.0, []
        
        failure_count = sum(1 for i, _ in neighbors if self.outcomes[i] < 0)
        return failure_count / len(neighbors), neighbors
    
    def coverage_signal(self, z: np.ndarray) -> float:
        """RAG doesn't have coverage - just max similarity."""
        self._ensure_matrix()
        if self._matrix is None:
            return 0.0
        
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        sims = self._matrix @ z_norm
        return float(np.max(sims)) if len(sims) > 0 else 0.0


class NEPBaseline:
    """
    Negative Experience Prediction: stores only failures.
    
    This is a simplified version of BEM that only tracks failure modes.
    - Binary failure detection (no continuous outcomes)
    - No success retrieval
    - Max-similarity coverage (not KDE)
    """
    
    def __init__(self, dim: int = 64, similarity_threshold: float = 0.7):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.failure_embeddings: List[np.ndarray] = []
        self._matrix = None
    
    def add_experience(self, embedding: np.ndarray, outcome: float, context: str):
        """Only stores failures (outcome < 0)."""
        if outcome < 0:  # Only store failures
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            self.failure_embeddings.append(embedding)
            self._matrix = None
    
    def _ensure_matrix(self):
        if self._matrix is None and self.failure_embeddings:
            self._matrix = np.vstack(self.failure_embeddings)
    
    def risk_signal(self, z: np.ndarray) -> Tuple[float, List]:
        """Binary risk: is this similar to any stored failure?"""
        self._ensure_matrix()
        if self._matrix is None:
            return 0.0, []
        
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        sims = self._matrix @ z_norm
        
        # Binary: any failure above threshold?
        max_sim = np.max(sims) if len(sims) > 0 else 0.0
        risk = 1.0 if max_sim > self.similarity_threshold else 0.0
        
        neighbors = [i for i, s in enumerate(sims) if s > self.similarity_threshold]
        return risk, neighbors
    
    def success_signal(self, z: np.ndarray) -> Tuple[float, List]:
        """NEP doesn't track successes."""
        return 0.0, []
    
    def coverage_signal(self, z: np.ndarray) -> float:
        """Max-similarity coverage (not KDE)."""
        self._ensure_matrix()
        if self._matrix is None:
            return 0.0
        
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        sims = self._matrix @ z_norm
        return float(np.max(sims)) if len(sims) > 0 else 0.0


# =============================================================================
# Data Generation (same as main experiments)
# =============================================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)

def generate_modes(n_modes: int, dim: int) -> np.ndarray:
    modes = np.random.randn(n_modes, dim)
    return normalize(modes)

def generate_overlapping_experiences(
    n_failures: int, 
    n_successes: int, 
    dim: int, 
    overlap: float = 0.3,
    failure_modes: Optional[np.ndarray] = None,
    success_modes: Optional[np.ndarray] = None
):
    """
    Generate experiences where some failures cluster near successes.
    """
    if failure_modes is None:
        failure_modes = generate_modes(10, dim)
    if success_modes is None:
        success_modes = generate_modes(5, dim)
    
    embeddings, outcomes = [], []
    
    # Core failures (easy)
    for _ in range(int(n_failures * (1 - overlap))):
        mode = failure_modes[np.random.randint(len(failure_modes))]
        vec = mode + np.random.randn(dim) * 0.05
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(-1.0, -0.5))
    
    # Overlapping failures (hard - near success modes)
    for _ in range(int(n_failures * overlap)):
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = mode + np.random.randn(dim) * 0.1  # Slightly more spread
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(-0.5, -0.3))  # Milder failures
    
    # Core successes (easy)
    for _ in range(int(n_successes * (1 - overlap))):
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = mode + np.random.randn(dim) * 0.05
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(0.5, 1.0))

    # Overlapping successes (hard - near failure modes)
    for _ in range(int(n_successes * overlap)):
        mode = failure_modes[np.random.randint(len(failure_modes))]
        vec = mode + np.random.randn(dim) * 0.1
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(0.3, 0.5))
    
    return embeddings, outcomes, failure_modes, success_modes

def generate_distributed_ood(n_samples: int, dim: int):
    """
    OOD samples that are uniformly distributed, not clustered.
    
    Max-similarity will find SOME similar training point.
    KDE coverage will correctly detect low density.
    """
    samples = []
    for _ in range(n_samples):
        # Random direction, scaled to be in sparse regions
        vec = np.random.randn(dim)
        vec = normalize(vec)
        samples.append(vec)
    return samples

def generate_clustered_ood(n_samples: int, dim: int):
    """Original OOD generation: clustered far away."""
    shift = np.random.randn(dim)
    shift = 3.0 * normalize(shift)
    samples = []
    for _ in range(n_samples):
        vec = shift + np.random.randn(dim) * 0.1
        vec = normalize(vec)
        samples.append(vec)
    return samples


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_failure_retrieval(
    system,
    test_failures: List[np.ndarray],
    test_successes: List[np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate failure retrieval: for failure queries, do we get high risk signal?
    
    Key metric: precision. Do we ONLY flag failures, not successes?
    RAG will have low precision because it retrieves anything similar.
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for z in test_failures:
        risk, _ = system.risk_signal(z)
        if risk > 0.3:  # Lower threshold to allow more positives
            tp += 1
        else:
            fn += 1
    
    for z in test_successes:
        risk, _ = system.risk_signal(z)
        if risk > 0.3:
            fp += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_success_retrieval(
    system,
    test_successes: List[np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate success retrieval: for success queries, do we get success signal?
    """
    if not hasattr(system, 'success_signal'):
        return {"success_rate": 0.0}
    
    success_count = 0
    for z in test_successes:
        success, _ = system.success_signal(z)
        if success > 0.3:
            success_count += 1
    
    return {"success_rate": success_count / len(test_successes)}


def evaluate_ood_detection(
    system,
    id_samples: List[np.ndarray],
    ood_samples: List[np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate OOD detection: can coverage distinguish ID from OOD?
    """
    id_coverage = [system.coverage_signal(z) for z in id_samples]
    ood_coverage = [system.coverage_signal(z) for z in ood_samples]
    
    # Compute AUC
    all_coverage = id_coverage + ood_coverage
    all_labels = [1] * len(id_coverage) + [0] * len(ood_coverage)
    
    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    
    for th in thresholds:
        tp = sum(1 for c, l in zip(all_coverage, all_labels) if c > th and l == 1)
        fn = sum(1 for c, l in zip(all_coverage, all_labels) if c <= th and l == 1)
        fp = sum(1 for c, l in zip(all_coverage, all_labels) if c > th and l == 0)
        tn = sum(1 for c, l in zip(all_coverage, all_labels) if c <= th and l == 0)
        
        tprs.append(tp / (tp + fn + 1e-10))
        fprs.append(fp / (fp + tn + 1e-10))
    
    auc = sum(0.5 * (fprs[i-1] - fprs[i]) * (tprs[i-1] + tprs[i]) for i in range(1, len(fprs)))
    
    return {
        "auc": auc,
        "id_coverage_mean": np.mean(id_coverage),
        "ood_coverage_mean": np.mean(ood_coverage)
    }


# =============================================================================
# Main Ablation Experiment
# =============================================================================

def run_ablation_study(seed: int = 42, verbose: bool = True):
    """
    Compare BEM vs RAG vs NEP on differentiated test scenarios.
    
    Key differentiators tested:
    1. RAG has no outcome awareness - will retrieve similar items regardless
    2. NEP can't retrieve successes - only tracks failures
    3. BEM has KDE coverage - better OOD detection
    """
    if verbose:
        print("=" * 60)
        print(f"Ablation Study: BEM vs RAG vs NEP (Seed={seed})")
        print("=" * 60)
    
    np.random.seed(seed)
    
    dim = 64
    
    # Generate data with some overlap between failures and successes
    print("\n[1/5] Generating data (Overlapping Distributions)...")
    
    # Generate shared modes first
    failure_modes = generate_modes(10, dim)
    success_modes = generate_modes(5, dim)
    
    train_emb, train_out, _, _ = generate_overlapping_experiences(
        500, 500, dim, overlap=0.3, 
        failure_modes=failure_modes, success_modes=success_modes
    )
    test_emb, test_out, _, _ = generate_overlapping_experiences(
        100, 100, dim, overlap=0.3,
        failure_modes=failure_modes, success_modes=success_modes
    )
    
    test_failures = [e for e, o in zip(test_emb, test_out) if o < 0]
    test_successes = [e for e, o in zip(test_emb, test_out) if o > 0]
    
    # ID samples for OOD detection (clustered failures)
    id_samples = []
    for _ in range(200):
        mode_idx = np.random.randint(len(failure_modes))
        vec = failure_modes[mode_idx] + np.random.randn(dim) * 0.05
        vec = normalize(vec)
        id_samples.append(vec)
    
    ood_clustered = generate_clustered_ood(200, dim)
    ood_distributed = generate_distributed_ood(200, dim)
    
    # Initialize systems
    print("[2/5] Initializing systems...")
    
    rag = PlainRAG(dim=dim, similarity_threshold=0.5)
    nep = NEPBaseline(dim=dim, similarity_threshold=0.5)
    bem = BidirectionalExperienceMemory(
        dim=dim,
        similarity_threshold=0.5,
        coverage_mode=CoverageMode.KDE,
        kde_bandwidth=0.3
    )
    
    # Populate all systems
    print("[3/5] Populating memories...")
    for i, (emb, out) in enumerate(zip(train_emb, train_out)):
        rag.add_experience(emb, out, f"exp_{i}")
        nep.add_experience(emb, out, f"exp_{i}")
        bem.add_experience(emb, out, f"exp_{i}")
    
    # Evaluate
    print("[4/5] Evaluating systems...")
    
    results = {}
    
    for name, system in [("RAG", rag), ("NEP", nep), ("BEM", bem)]:
        print(f"\n  Evaluating {name}...")
        
        failure_ret = evaluate_failure_retrieval(system, test_failures, test_successes)
        success_ret = evaluate_success_retrieval(system, test_successes)
        ood_clust = evaluate_ood_detection(system, id_samples, ood_clustered)
        ood_dist = evaluate_ood_detection(system, id_samples, ood_distributed)
        
        results[name] = {
            "failure_retrieval": failure_ret,
            "success_retrieval": success_ret,
            "ood_clustered": ood_clust,
            "ood_distributed": ood_dist
        }
        
        print(f"    Failure Retrieval F1: {failure_ret['f1']:.3f} (P={failure_ret['precision']:.2f}, R={failure_ret['recall']:.2f})")
        print(f"    Success Retrieval Rate: {success_ret['success_rate']:.3f}")
        print(f"    OOD Detection AUC (Clustered): {ood_clust['auc']:.3f}")
        print(f"    OOD Detection AUC (Distributed): {ood_dist['auc']:.3f}")
    
    # Summary table
    if verbose:
        print("\n" + "=" * 80)
        print("Ablation Results Summary")
        print("=" * 80)
        print(f"{'Metric':<30} {'RAG':>10} {'NEP':>10} {'BEM':>10}")
        print("-" * 80)
        print(f"{'Failure Retrieval F1':<30} {results['RAG']['failure_retrieval']['f1']:>10.2f} {results['NEP']['failure_retrieval']['f1']:>10.2f} {results['BEM']['failure_retrieval']['f1']:>10.2f}")
        print(f"{'Success Retrieval Rate':<30} {results['RAG']['success_retrieval']['success_rate']:>10.2f} {results['NEP']['success_retrieval']['success_rate']:>10.2f} {results['BEM']['success_retrieval']['success_rate']:>10.2f}")
        print(f"{'OOD AUC (Clustered)':<30} {results['RAG']['ood_clustered']['auc']:>10.2f} {results['NEP']['ood_clustered']['auc']:>10.2f} {results['BEM']['ood_clustered']['auc']:>10.2f}")
        print(f"{'OOD AUC (Distributed)':<30} {results['RAG']['ood_distributed']['auc']:>10.2f} {results['NEP']['ood_distributed']['auc']:>10.2f} {results['BEM']['ood_distributed']['auc']:>10.2f}")
    
    if verbose:
        print("\nAnalysis:")
        print(f"  - RAG has no success retrieval (no outcome awareness)")
        print(f"  - NEP has no success retrieval (only stores failures)")
        print(f"  - BEM provides success retrieval: {results['BEM']['success_retrieval']['success_rate']:.2f}")

    return results


def run_multi_seed_ablation(n_seeds: int = 10, verbose: bool = True) -> Dict:
    """
    Run ablation study with multiple seeds for statistical significance.

    Returns mean ± std for each metric across seeds.
    """
    all_results = []

    print("=" * 60)
    print(f"Multi-Seed Ablation Study (N={n_seeds})")
    print("=" * 60)

    for seed in range(n_seeds):
        if verbose:
            print(f"\nRunning seed {seed + 1}/{n_seeds}...")
        results = run_ablation_study(seed=seed, verbose=False)
        all_results.append(results)

    # Aggregate results
    metrics = [
        ("failure_retrieval", "f1"),
        ("failure_retrieval", "precision"),
        ("failure_retrieval", "recall"),
        ("success_retrieval", "success_rate"),
        ("ood_clustered", "auc"),
        ("ood_distributed", "auc")
    ]

    systems = ["RAG", "NEP", "BEM"]

    aggregated = {}
    for system in systems:
        aggregated[system] = {}
        for metric_group, metric_name in metrics:
            values = [r[system][metric_group][metric_name] for r in all_results]
            aggregated[system][f"{metric_group}_{metric_name}"] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }

    # Print summary with confidence intervals
    print("\n" + "=" * 90)
    print("Statistical Summary (Mean ± Std over {} seeds)".format(n_seeds))
    print("=" * 90)

    print(f"\n{'Metric':<35} {'RAG':>15} {'NEP':>15} {'BEM':>15}")
    print("-" * 90)

    for metric_group, metric_name in metrics:
        key = f"{metric_group}_{metric_name}"
        label = f"{metric_group.replace('_', ' ').title()} {metric_name.upper()}"

        rag = aggregated["RAG"][key]
        nep = aggregated["NEP"][key]
        bem = aggregated["BEM"][key]

        print(f"{label:<35} {rag['mean']:>6.3f}±{rag['std']:.3f} {nep['mean']:>6.3f}±{nep['std']:.3f} {bem['mean']:>6.3f}±{bem['std']:.3f}")

    # Statistical significance test (paired t-test: BEM vs RAG, BEM vs NEP)
    from scipy import stats

    print("\n" + "=" * 90)
    print("Statistical Significance (Paired t-test, α=0.05)")
    print("=" * 90)

    key_metrics = [
        ("failure_retrieval_f1", "Failure F1"),
        ("ood_distributed_auc", "OOD AUC (Distributed)")
    ]

    for key, label in key_metrics:
        bem_vals = aggregated["BEM"][key]["values"]
        rag_vals = aggregated["RAG"][key]["values"]
        nep_vals = aggregated["NEP"][key]["values"]

        # BEM vs RAG
        t_stat_rag, p_val_rag = stats.ttest_rel(bem_vals, rag_vals)
        sig_rag = "✓ Significant" if p_val_rag < 0.05 else "✗ Not significant"

        # BEM vs NEP
        t_stat_nep, p_val_nep = stats.ttest_rel(bem_vals, nep_vals)
        sig_nep = "✓ Significant" if p_val_nep < 0.05 else "✗ Not significant"

        print(f"\n{label}:")
        print(f"  BEM vs RAG: t={t_stat_rag:.3f}, p={p_val_rag:.4f} → {sig_rag}")
        print(f"  BEM vs NEP: t={t_stat_nep:.3f}, p={p_val_nep:.4f} → {sig_nep}")

    return aggregated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed experiment")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    args = parser.parse_args()

    if args.multi_seed:
        results = run_multi_seed_ablation(n_seeds=args.n_seeds)
    else:
        results = run_ablation_study()
