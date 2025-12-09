
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode

np.random.seed(42)

# =============================================================================
# Baseline Implementations (Copied from ablation_study.py)
# =============================================================================

class PlainRAG:
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
        self._matrix = None
    
    def _ensure_matrix(self):
        if self._matrix is None and self.embeddings:
            self._matrix = np.vstack(self.embeddings)
    
    def retrieve(self, z: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        self._ensure_matrix()
        if self._matrix is None:
            return []
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        sims = self._matrix @ z_norm
        indices = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in indices if sims[i] > self.similarity_threshold]
    
    def risk_signal(self, z: np.ndarray) -> Tuple[float, List]:
        neighbors = self.retrieve(z, k=10)
        if not neighbors:
            return 0.0, []
        failure_count = sum(1 for i, _ in neighbors if self.outcomes[i] < 0)
        return failure_count / len(neighbors), neighbors

# =============================================================================
# Data Generation
# =============================================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)

def generate_modes(n_modes: int, dim: int) -> np.ndarray:
    modes = np.random.randn(n_modes, dim)
    return normalize(modes)

def generate_overlapping_experiences(n_failures, n_successes, dim, overlap=0.3, failure_modes=None, success_modes=None):
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
    
    # Overlapping failures (hard)
    for _ in range(int(n_failures * overlap)):
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = mode + np.random.randn(dim) * 0.1
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(-0.5, -0.3))
    
    # Core successes (easy)
    for _ in range(int(n_successes * (1 - overlap))):
        mode = success_modes[np.random.randint(len(success_modes))]
        vec = mode + np.random.randn(dim) * 0.05
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(0.5, 1.0))

    # Overlapping successes (hard)
    for _ in range(int(n_successes * overlap)):
        mode = failure_modes[np.random.randint(len(failure_modes))]
        vec = mode + np.random.randn(dim) * 0.1
        embeddings.append(normalize(vec))
        outcomes.append(np.random.uniform(0.3, 0.5))
    
    return embeddings, outcomes

# =============================================================================
# Analysis Logic
# =============================================================================

def run_deep_analysis():
    dim = 64
    print("Generating data...")
    
    failure_modes = generate_modes(10, dim)
    success_modes = generate_modes(5, dim)

    train_emb, train_out = generate_overlapping_experiences(
        500, 500, dim, overlap=0.3,
        failure_modes=failure_modes, success_modes=success_modes
    )
    test_emb, test_out = generate_overlapping_experiences(
        100, 100, dim, overlap=0.3,
        failure_modes=failure_modes, success_modes=success_modes
    )

    
    test_failures = [e for e, o in zip(test_emb, test_out) if o < 0]
    test_successes = [e for e, o in zip(test_emb, test_out) if o > 0]
    
    print(f"Test Failures: {len(test_failures)}")
    print(f"Test Successes: {len(test_successes)}")

    # Initialize RAG and BEM
    rag = PlainRAG(dim=dim, similarity_threshold=0.5)
    bem = BidirectionalExperienceMemory(
        dim=dim,
        similarity_threshold=0.5,
        coverage_mode=CoverageMode.KDE,
        kde_bandwidth=0.3
    )

    print("Populating...")
    for i, (emb, out) in enumerate(zip(train_emb, train_out)):
        rag.add_experience(emb, out, f"exp_{i}")
        bem.add_experience(emb, out, f"exp_{i}")

    # Inspect Similarity Stats
    print("\n--- Similarity Analysis ---")
    z = test_failures[0]
    rag._ensure_matrix()
    z_norm = z / (np.linalg.norm(z) + 1e-10)
    sims = rag._matrix @ z_norm
    print(f"Max Sim: {np.max(sims):.4f}, Mean Sim: {np.mean(sims):.4f}, Min Sim: {np.min(sims):.4f}")
    print(f"Count > 0.5: {np.sum(sims > 0.5)}")
    print(f"Count > 0.3: {np.sum(sims > 0.3)}")
    
    # Analyze Risk Signal Internals
    print("\n--- Risk Signal Debug ---")
    risk, neighbors = rag.risk_signal(z)
    print(f"RAG Risk (z0): {risk}, Neighbors: {len(neighbors)}")
    risk, _ = bem.risk_signal(z)
    print(f"BEM Risk (z0): {risk}")

    # Evaluate Precision/Recall for Failures
    print("\n--- Failure Retrieval Analysis ---")
    
    def evaluate(system, name):
        tp, fp, fn = 0, 0, 0
        
        missed_failures = []
        false_positives = []
        
        # Check recall (find failures)
        for i, z in enumerate(test_failures):
            risk, _ = system.risk_signal(z)
            if risk > 0.3:
                tp += 1
            else:
                fn += 1
                missed_failures.append((i, risk))
        
        # Check precision (don't flag successes)
        for i, z in enumerate(test_successes):
            risk, _ = system.risk_signal(z)
            if risk > 0.3:
                fp += 1
                false_positives.append((i, risk))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        print(f"[{name}] F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | TP: {tp}, FP: {fp}, FN: {fn}")
        return missed_failures, false_positives

    rag_miss, rag_fp = evaluate(rag, "RAG")
    bem_miss, bem_fp = evaluate(bem, "BEM")

    # Compare differences
    print("\n--- Comparison ---")
    print(f"RAG Missed: {len(rag_miss)}, BEM Missed: {len(bem_miss)}")
    print(f"RAG FalsePos: {len(rag_fp)}, BEM FalsePos: {len(bem_fp)}")

    # Detailed inspection of specific cases
    print("\n--- Detailed Inspection ---")
    # Find cases where RAG caught it but BEM missed it (Recall gap)
    rag_miss_indices = {i for i, _ in rag_miss}
    bem_miss_indices = {i for i, _ in bem_miss}
    
    bem_only_misses = bem_miss_indices - rag_miss_indices
    if bem_only_misses:
        print(f"\nCases where BEM missed but RAG caught (Count: {len(bem_only_misses)}):")
        example_idx = list(bem_only_misses)[0]
        z = test_failures[example_idx]
        
        print(f"Example Failure Index {example_idx}:")
        rag_risk, rag_nbrs = rag.risk_signal(z)
        bem_risk, bem_nbrs = bem.risk_signal(z)
        print(f"  RAG Risk: {rag_risk:.4f} (from {len(rag_nbrs)} neighbors)")
        print(f"  BEM Risk: {bem_risk:.4f} (from {len(bem_nbrs)} neighbors)")
        
        # Inspect BEM internals for this case if possible
        # BEM risk = sum(sim * I(failure)) / sum(sim)
        # Check similarities
        sims = bem._compute_similarities(z)
        top_k_indices = np.argsort(sims)[::-1][:10]
        print("  Top 10 similar items in BEM:")
        for idx in top_k_indices:
            exp = bem.experiences[idx]
            sim = sims[idx]
            print(f"    ID: {idx}, Sim: {sim:.4f}, Outcome: {exp.outcome:.4f}, IsFail: {exp.is_failure}")

    # Find cases where BEM flagged success as failure (Precision gap)
    rag_fp_indices = {i for i, _ in rag_fp}
    bem_fp_indices = {i for i, _ in bem_fp}
    
    bem_only_fps = bem_fp_indices - rag_fp_indices
    if bem_only_fps:
        print(f"\nCases where BEM false alarmed but RAG didn't (Count: {len(bem_only_fps)}):")
        example_idx = list(bem_only_fps)[0]
        z = test_successes[example_idx]
        print(f"Example Success Index {example_idx}:")
        rag_risk, _ = rag.risk_signal(z)
        bem_risk, _ = bem.risk_signal(z)
        print(f"  RAG Risk: {rag_risk:.4f}")
        print(f"  BEM Risk: {bem_risk:.4f}")

if __name__ == "__main__":
    run_deep_analysis()
