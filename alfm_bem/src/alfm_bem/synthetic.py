"""
Synthetic data generators for ALFM-BEM experiments.

These utilities are shared across experiments/analysis to keep results
reproducible and consistent.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector(s) to unit length."""
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-10)


def generate_modes(n_modes: int, dim: int, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate random unit vectors to act as cluster centroids."""
    if rng is None:
        rng = np.random.default_rng()
    modes = rng.standard_normal((n_modes, dim))
    return normalize(modes)


def generate_overlapping_experiences(
    n_failures: int,
    n_successes: int,
    dim: int,
    *,
    overlap: float = 0.3,
    failure_modes: Optional[np.ndarray] = None,
    success_modes: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[float], np.ndarray, np.ndarray]:
    """
    Generate experiences where some failures cluster near successes (and vice versa).
    """
    if rng is None:
        rng = np.random.default_rng()
    if failure_modes is None:
        failure_modes = generate_modes(10, dim, rng=rng)
    if success_modes is None:
        success_modes = generate_modes(5, dim, rng=rng)

    embeddings: List[np.ndarray] = []
    outcomes: List[float] = []

    # Core failures (easy)
    for _ in range(int(n_failures * (1 - overlap))):
        mode = failure_modes[rng.integers(len(failure_modes))]
        vec = mode + rng.standard_normal(dim) * 0.05
        embeddings.append(normalize(vec))
        outcomes.append(float(rng.uniform(-1.0, -0.5)))

    # Overlapping failures (hard - near success modes)
    for _ in range(int(n_failures * overlap)):
        mode = success_modes[rng.integers(len(success_modes))]
        vec = mode + rng.standard_normal(dim) * 0.1
        embeddings.append(normalize(vec))
        outcomes.append(float(rng.uniform(-0.5, -0.3)))

    # Core successes (easy)
    for _ in range(int(n_successes * (1 - overlap))):
        mode = success_modes[rng.integers(len(success_modes))]
        vec = mode + rng.standard_normal(dim) * 0.05
        embeddings.append(normalize(vec))
        outcomes.append(float(rng.uniform(0.5, 1.0)))

    # Overlapping successes (hard - near failure modes)
    for _ in range(int(n_successes * overlap)):
        mode = failure_modes[rng.integers(len(failure_modes))]
        vec = mode + rng.standard_normal(dim) * 0.1
        embeddings.append(normalize(vec))
        outcomes.append(float(rng.uniform(0.3, 0.5)))

    return embeddings, outcomes, failure_modes, success_modes


def generate_distributed_ood(n_samples: int, dim: int, *, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    """
    OOD samples that are uniformly distributed, not clustered.

    Max-similarity tends to find some neighbor; density-based coverage should drop.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples: List[np.ndarray] = []
    for _ in range(n_samples):
        samples.append(normalize(rng.standard_normal(dim)))
    return samples


def generate_clustered_ood(
    n_samples: int,
    dim: int,
    *,
    shift_magnitude: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Cluster OOD samples around a novel shifted centroid."""
    if rng is None:
        rng = np.random.default_rng()
    shift = normalize(rng.standard_normal(dim)) * shift_magnitude
    samples: List[np.ndarray] = []
    for _ in range(n_samples):
        vec = shift + rng.standard_normal(dim) * 0.1
        samples.append(normalize(vec))
    return samples

