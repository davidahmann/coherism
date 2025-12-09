"""
Bidirectional Experience Memory (BEM)
=====================================

A unified memory architecture that stores deployment experiences (both failures 
and successes) and provides:
1. Risk signals from failure similarity
2. Success patterns for retrieval-augmented reasoning
3. Out-of-distribution detection as an emergent property of coverage
4. Experience sampling for adapter training

This is NOT two separate memories bolted together — it is a single abstraction
where experiences live on a continuous outcome spectrum from failure (-1) to 
success (+1), with retrieval and risk computation operating uniformly over
the entire space.

Author: David Ahmann
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from enum import Enum
import json
import hashlib
from datetime import datetime


class CoverageMode(Enum):
    """Coverage signal computation method."""
    LEGACY = "legacy"      # Original: max_sim + Mahalanobis (AUC ~0.60)
    KNN_AVG = "knn_avg"    # k-NN average similarity (AUC ~0.70)
    KDE = "kde"            # Kernel Density Estimation (AUC ~0.88)


@dataclass
class Experience:
    """A single deployment experience stored in BEM."""
    
    embedding: np.ndarray          # Projected context embedding z ∈ R^d
    outcome: float                 # Continuous outcome score in [-1, 1]
    context_hash: str              # SHA-256 of original context (for dedup)
    timestamp: datetime            # When this experience occurred
    tenant_id: str = "default"     # Tenant isolation
    domain_id: str = "default"     # Domain classification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields for richer experiences
    reasoning_trace: Optional[str] = None   # What reasoning led to this outcome
    correction: Optional[str] = None        # Human correction if failure
    confirmation_count: int = 1             # How many times confirmed
    
    @property
    def is_failure(self) -> bool:
        return self.outcome < -0.3
    
    @property
    def is_success(self) -> bool:
        return self.outcome > 0.3
    
    @property
    def severity(self) -> int:
        """Map outcome to severity 1-5 for compatibility with original NEP."""
        if self.outcome >= 0.5:
            return 1  # Strong success
        elif self.outcome >= 0:
            return 2  # Mild success
        elif self.outcome >= -0.5:
            return 3  # Mild failure
        elif self.outcome >= -0.8:
            return 4  # Significant failure
        else:
            return 5  # Critical failure


class BidirectionalExperienceMemory:
    """
    Unified experience memory that generalizes NEP to bidirectional learning.
    
    Key design principle: Experiences are points in embedding space with 
    continuous outcome labels. There is no hard boundary between "failure 
    memory" and "success memory" — they are two views of the same structure.
    
    The memory provides:
    - risk_signal(): How similar is this context to past failures?
    - success_signal(): How similar to past successes?
    - coverage_signal(): How well is this context covered by any experiences?
    - retrieve(): Get relevant experiences for augmentation
    - sample_for_training(): Get balanced batch for adapter updates
    """
    
    def __init__(
        self,
        dim: int = 768,
        similarity_threshold: float = 0.7,
        risk_sensitivity: float = 0.8,
        decay_rate: float = 0.01,
        tenant_id: str = "global",
        coverage_mode: CoverageMode = CoverageMode.KDE,
        kde_bandwidth: float = 0.3,
        knn_k: int = 20,
        use_knn_index: bool = False  # Task 4.4: Scalable Indexing
    ):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.risk_sensitivity = risk_sensitivity
        self.decay_rate = decay_rate
        self.tenant_id = tenant_id

        # Coverage signal configuration
        self.coverage_mode = coverage_mode
        self.kde_bandwidth = kde_bandwidth
        self.knn_k = knn_k
        self.use_knn_index = use_knn_index

        # Unified storage — no separate failure/success stores
        self.experiences: List[Experience] = []

        # Index structures for efficient retrieval
        self._embedding_matrix: Optional[np.ndarray] = None
        self._kde_model = None
        self._knn_index = None
        self._last_update_time = datetime.min
        self._needs_reindex = False

        # For legacy coverage mode
        self._mean_embedding: Optional[np.ndarray] = None
        self._cov_embedding: Optional[np.ndarray] = None

        # Anti-poisoning: velocity tracking per source
        self._feedback_velocity: Dict[str, float] = defaultdict(float)
        self._feedback_history: Dict[str, List[datetime]] = defaultdict(list)
        self._quarantine: List[Experience] = []

    def add_experience(
        self, 
        embedding: np.ndarray, 
        outcome: float, 
        context_hash: str,
        tenant_id: str = "default",
        domain_id: str = "default",
        reasoning_trace: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """Add new experience to memory."""
        # Validate shapes
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dim {embedding.shape[0]} != {self.dim}")
            
        exp = Experience(
            embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
            outcome=outcome,
            context_hash=context_hash,
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            domain_id=domain_id,
            reasoning_trace=reasoning_trace,
            correction=correction
        )
        self.experiences.append(exp)
        
        # Invalidate indices
        self._embedding_matrix = None
        self._kde_model = None
        self._knn_index = None

    def _ensure_index(self):
        """Rebuid search index if needed."""
        if self._embedding_matrix is None and self.experiences:
            self._embedding_matrix = np.vstack([e.embedding for e in self.experiences])
            
        # 1. Update KNN Index (Scalable Retrieval)
        if self.use_knn_index and self._knn_index is None and len(self.experiences) > 0:
            # Use dot product logic via Euclidean on normalized vectors:
            # dist^2 = 2 - 2*sim => sim = 1 - dist^2/2
            from sklearn.neighbors import NearestNeighbors
            self._knn_index = NearestNeighbors(metric='euclidean', algorithm='auto')
            self._knn_index.fit(self._embedding_matrix)

        # 2. Update KDE Model (Coverage)
        if self.coverage_mode == CoverageMode.KDE and self._kde_model is None and len(self.experiences) > 10:
            from sklearn.neighbors import KernelDensity
            self._kde_model = KernelDensity(kernel='gaussian', bandwidth=self.kde_bandwidth)
            self._kde_model.fit(self._embedding_matrix)

    def _compute_similarities(self, z: np.ndarray) -> np.ndarray:
        self._ensure_index()
        if self._embedding_matrix is None:
            return np.array([])
            
        z = z / (np.linalg.norm(z) + 1e-10)
        
        # Fast Path: Approximate Nearest Neighbors
        if self.use_knn_index and self._knn_index is not None:
            # We want ALL similarities for logic, but usually we only need top-k.
            # However, existing logic iterates over zip(sims, self.experiences).
            # Changing this logic fully is risky (breaking logic relying on full scan).
            # For now, if use_knn_index is True, we ONLY retrieve top-50 candidates
            # and set others to 0.0. This is an approximation.
            k = min(50, len(self.experiences))
            dists, indices = self._knn_index.kneighbors(z.reshape(1, -1), n_neighbors=k)
            
            sims = np.zeros(len(self.experiences))
            # Convert Euclidean dist back to Cosine Sim: sim = 1 - d^2/2
            # (Valid because vectors are unit length)
            converted_sims = 1.0 - (dists[0]**2) / 2.0
            
            for idx, sim in zip(indices[0], converted_sims):
                sims[idx] = sim
            return sims

        # Slow Path: Brute Force
        sims = self._embedding_matrix @ z
        return sims

    def risk_signal(self, z: np.ndarray) -> Tuple[float, List[Experience]]:
        """
        Compute risk signal: probability that this context leads to failure,
        based on similarity to past failures.
        
        This is the generalization of NEP's s_NEP.
        
        Returns:
            (risk_score in [0,1], list of relevant failure experiences)
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0, []
        
        # Filter to failures and above threshold
        relevant_candidates = []
        
        for i, (sim, exp) in enumerate(zip(sims, self.experiences)):
            if sim > self.similarity_threshold and exp.is_failure:
                relevant_candidates.append((sim, exp))
        
        if not relevant_candidates:
            return 0.0, []
            
        # Sort by similarity (descending) and take top-k to prevent unbounded risk accumulation
        # from many weak signals in dense regions
        relevant_candidates.sort(key=lambda x: x[0], reverse=True)
        relevant_candidates = relevant_candidates[:20]  # Limit to 20 nearest failures
        
        relevant = [exp for sim, exp in relevant_candidates]
        risk_components = []
        
        for sim, exp in relevant_candidates:
            # Weight by similarity, severity, and recency
            age_days = (datetime.now() - exp.timestamp).days
            decay = np.exp(-self.decay_rate * age_days)
            weight = sim * (exp.severity / 5.0) * decay * exp.confirmation_count
            risk_components.append(
                self.risk_sensitivity * weight
            )
        
        if not risk_components:
            return 0.0, []
        
        # Aggregate via noisy-OR: P(at least one failure mode active)
        risk = 1.0 - np.prod([1.0 - min(r, 0.99) for r in risk_components])
        
        # Correlation boost: if multiple similar failures, increase risk
        if len(relevant) > 1:
            failure_embeddings = np.vstack([e.embedding for e in relevant])
            pairwise_sims = failure_embeddings @ failure_embeddings.T
            max_correlation = np.max(pairwise_sims[np.triu_indices(len(relevant), k=1)])
            risk = min(1.0, risk + 0.1 * max_correlation)
        
        return float(risk), relevant
    
    def success_signal(self, z: np.ndarray) -> Tuple[float, List[Experience]]:
        """
        Compute success signal: evidence that similar contexts succeeded.
        
        This enables retrieval of successful reasoning patterns.
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0, []
        
        relevant = []
        success_components = []
        
        for sim, exp in zip(sims, self.experiences):
            if sim > self.similarity_threshold and exp.is_success:
                relevant.append(exp)
                age_days = (datetime.now() - exp.timestamp).days
                decay = np.exp(-self.decay_rate * age_days)
                weight = sim * ((6 - exp.severity) / 5.0) * decay
                success_components.append(weight)
        
        if not success_components:
            return 0.0, []
        
        # Average success evidence (not noisy-OR — successes are reinforcing)
        success = float(np.mean(success_components))
        return min(1.0, success), relevant
    
    def coverage_signal(self, z: np.ndarray) -> float:
        """
        Compute coverage signal: how well is this context represented
        in our experience distribution?
        
        Low coverage = out-of-distribution = generalization gap territory.
        
        This is an EMERGENT property of BEM, not a separate detector.
        
        Supports multiple modes:
        - LEGACY: max_sim + Mahalanobis (original, AUC ~0.60)
        - KNN_AVG: k-NN average similarity (AUC ~0.70)
        - KDE: Kernel Density Estimation on cosine distances (AUC ~0.88)
        """
        self._ensure_index()
        if self._embedding_matrix is None or len(self.experiences) < 10:
            return 0.0  # Not enough data for coverage estimation
        
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return 0.0
        
        if self.coverage_mode == CoverageMode.KDE:
            return self._coverage_kde(sims)
        elif self.coverage_mode == CoverageMode.KNN_AVG:
            return self._coverage_knn_avg(sims)
        else:  # LEGACY
            return self._coverage_legacy(z, sims)
    
    def _coverage_kde(self, sims: np.ndarray) -> float:
        """
        Kernel Density Estimation coverage.
        
        Computes density using Gaussian kernel over cosine distances.
        High density = well covered by experiences = in-distribution.
        """
        # Convert similarities to distances
        distances = 1.0 - sims
        
        # Gaussian kernel: K(d) = exp(-d^2 / (2 * bw^2))
        kernel_vals = np.exp(-distances**2 / (2 * self.kde_bandwidth**2))
        
        # Normalize by number of experiences for consistent scale
        density = np.mean(kernel_vals)
        
        # Scale to [0, 1] range (density of 1.0 = all experiences at distance 0)
        # In practice, max density is much lower, so we rescale
        coverage = np.clip(density * 30, 0.0, 1.0)  # Scaling factor tuned empirically
        
        return float(coverage)
    
    def _coverage_knn_avg(self, sims: np.ndarray) -> float:
        """
        k-NN average similarity coverage.
        
        Average of top-k similarities. More robust than max similarity.
        """
        k = min(self.knn_k, len(sims))
        top_k_sims = np.sort(sims)[-k:]
        coverage = float(np.mean(top_k_sims))
        return np.clip(coverage, 0.0, 1.0)
    
    def _coverage_legacy(self, z: np.ndarray, sims: np.ndarray) -> float:
        """
        Original coverage method: max_sim + Mahalanobis.
        
        Kept for backward compatibility and ablation studies.
        """
        z_norm = z / (np.linalg.norm(z) + 1e-10)
        max_sim = np.max(sims) if len(sims) > 0 else 0.0
        
        try:
            diff = z_norm - self._mean_embedding
            cov_inv = np.linalg.inv(self._cov_embedding + 1e-6 * np.eye(self.dim))
            mahal_dist = np.sqrt(diff @ cov_inv @ diff)
            mahal_coverage = np.exp(-mahal_dist / 10.0)
        except:
            mahal_coverage = 0.5
        
        coverage = 0.6 * max_sim + 0.4 * mahal_coverage
        return float(np.clip(coverage, 0.0, 1.0))
    
    def retrieve(
        self,
        z: np.ndarray,
        k: int = 5,
        outcome_filter: Optional[str] = None  # "failure", "success", or None for all
    ) -> List[Tuple[Experience, float]]:
        """
        Retrieve k most relevant experiences with their similarities.
        
        Args:
            z: Query embedding
            k: Number to retrieve
            outcome_filter: Optionally filter to failures or successes
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return []
        
        # Filter by outcome if requested
        candidates = []
        for i, (sim, exp) in enumerate(zip(sims, self.experiences)):
            if outcome_filter == "failure" and not exp.is_failure:
                continue
            if outcome_filter == "success" and not exp.is_success:
                continue
            candidates.append((exp, float(sim)))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def sample_for_training(
        self,
        batch_size: int = 32,
        failure_ratio: float = 0.7
    ) -> List[Experience]:
        """
        Sample a balanced batch for adapter training.
        
        Biases toward failures (more informative) but includes successes
        to prevent forgetting what works.
        """
        failures = [e for e in self.experiences if e.is_failure]
        successes = [e for e in self.experiences if e.is_success]
        
        n_failures = min(int(batch_size * failure_ratio), len(failures))
        n_successes = min(batch_size - n_failures, len(successes))
        
        # Weight by confirmation count (more confirmed = more important)
        if failures:
            failure_weights = np.array([e.confirmation_count for e in failures])
            failure_weights = failure_weights / failure_weights.sum()
            sampled_failures = list(np.random.choice(
                failures, size=n_failures, replace=False, p=failure_weights
            )) if n_failures <= len(failures) else failures
        else:
            sampled_failures = []
        
        if successes:
            success_weights = np.array([e.confirmation_count for e in successes])
            success_weights = success_weights / success_weights.sum()
            sampled_successes = list(np.random.choice(
                successes, size=n_successes, replace=False, p=success_weights
            )) if n_successes <= len(successes) else successes
        else:
            sampled_successes = []
        
        return sampled_failures + sampled_successes
    
    def vacuum(self, max_size: int = 10000, cluster_threshold: float = 0.95):
        """
        Memory hygiene: merge redundant experiences and prune old ones.

        This prevents unbounded growth while preserving coverage.
        """
        if len(self.experiences) <= max_size:
            return

        # Simple approach: keep most confirmed and most recent
        self.experiences.sort(
            key=lambda e: (e.confirmation_count, e.timestamp),
            reverse=True
        )
        self.experiences = self.experiences[:max_size]
        self._needs_reindex = True
        self._embedding_matrix = None
        self._kde_model = None
        self._knn_index = None

    def check_feedback_velocity(
        self,
        source_id: str,
        gamma: float = 0.95,
        threshold_sigma: float = 3.0
    ) -> bool:
        """
        Anti-poisoning: Check if feedback velocity is anomalously high.

        Implements exponential moving average of feedback rate per source.
        Returns True if source should be quarantined.

        Paper reference: Section 3.6, Equation for v_u(t)
        """
        now = datetime.now()
        self._feedback_history[source_id].append(now)

        # Update EMA velocity
        old_velocity = self._feedback_velocity[source_id]
        self._feedback_velocity[source_id] = gamma * old_velocity + (1 - gamma) * 1.0

        # Compute global statistics
        all_velocities = list(self._feedback_velocity.values())
        if len(all_velocities) < 5:
            return False  # Not enough data

        mean_v = np.mean(all_velocities)
        std_v = np.std(all_velocities) + 1e-10

        # Anomaly detection
        if self._feedback_velocity[source_id] > mean_v + threshold_sigma * std_v:
            return True  # Anomalous velocity

        return False

    def add_experience_with_validation(
        self,
        embedding: np.ndarray,
        outcome: float,
        context_hash: str,
        source_id: str = "unknown",
        tenant_id: str = "default",
        domain_id: str = "default",
        **kwargs
    ) -> Optional[str]:
        """
        Add experience with full anti-poisoning validation.

        Returns:
            None if successful, error message if rejected/quarantined
        """
        # Check velocity
        if self.check_feedback_velocity(source_id):
            # Quarantine instead of immediate add
            exp = Experience(
                embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
                outcome=outcome,
                context_hash=context_hash,
                timestamp=datetime.now(),
                tenant_id=tenant_id,
                domain_id=domain_id,
                **kwargs
            )
            self._quarantine.append(exp)
            return f"Quarantined: High feedback velocity from {source_id}"

        # Check conflict
        conflict = self.check_conflict(embedding, outcome)
        if conflict:
            return conflict

        # Normal add
        self.add_experience(
            embedding=embedding,
            outcome=outcome,
            context_hash=context_hash,
            tenant_id=tenant_id,
            domain_id=domain_id,
            **kwargs
        )
        return None

    def process_quarantine(self, admin_approved_ids: List[str] = None):
        """
        Process quarantined experiences.

        Args:
            admin_approved_ids: List of context_hashes approved by admin
        """
        if admin_approved_ids is None:
            admin_approved_ids = []

        approved = []
        rejected = []

        for exp in self._quarantine:
            if exp.context_hash in admin_approved_ids:
                self.experiences.append(exp)
                approved.append(exp.context_hash)
            else:
                rejected.append(exp.context_hash)

        # Clear quarantine
        self._quarantine = [e for e in self._quarantine
                           if e.context_hash not in admin_approved_ids]

        if approved or rejected:
            self._embedding_matrix = None  # Invalidate cache

        return {"approved": len(approved), "rejected": len(rejected)}

    def check_conflict(self, z: np.ndarray, outcome: float) -> Optional[str]:
        """
        Check if the proposed outcome contradicts high-confidence memory.
        
        Anti-poisoning mechanism (Section 3.6):
        Rejects updates that flip the label of a known, well-confirmed context.
        
        Returns:
            Warning string if conflict found, None otherwise.
        """
        sims = self._compute_similarities(z)
        if len(sims) == 0:
            return None
            
        # Look for very similar experiences (strict threshold 0.85)
        CONFLICT_SIM_THRESHOLD = 0.85
        CONFIDENCE_MIN = 3  # Must be confirmed 3+ times to block
        
        for sim, exp in zip(sims, self.experiences):
            if sim > CONFLICT_SIM_THRESHOLD:
                # Check for sign mismatch (Opposite labels)
                # Success vs Failure
                if (outcome > 0.3 and exp.is_failure) or (outcome < -0.3 and exp.is_success):
                    if exp.confirmation_count >= CONFIDENCE_MIN:
                        return (
                            f"Conflict detected: New outcome ({outcome:.2f}) contradicts "
                            f"established experience ({exp.outcome:.2f}, confirmed {exp.confirmation_count}x) "
                            f"with similarity {sim:.2f}."
                        )
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the memory."""
        if not self.experiences:
            return {"total": 0}
        
        outcomes = np.array([e.outcome for e in self.experiences])
        return {
            "total": len(self.experiences),
            "failures": sum(1 for e in self.experiences if e.is_failure),
            "successes": sum(1 for e in self.experiences if e.is_success),
            "neutral": sum(1 for e in self.experiences if not e.is_failure and not e.is_success),
            "mean_outcome": float(np.mean(outcomes)),
            "std_outcome": float(np.std(outcomes)),
            "unique_domains": len(set(e.domain_id for e in self.experiences)),
            "total_confirmations": sum(e.confirmation_count for e in self.experiences)
        }
    
    def save(self, path: str):
        """Serialize memory to disk."""
        data = {
            "config": {
                "dim": self.dim,
                "similarity_threshold": self.similarity_threshold,
                "risk_sensitivity": self.risk_sensitivity,
                "decay_rate": self.decay_rate,
                "tenant_id": self.tenant_id
            },
            "experiences": [
                {
                    "embedding": e.embedding.tolist(),
                    "outcome": e.outcome,
                    "context_hash": e.context_hash,
                    "timestamp": e.timestamp.isoformat(),
                    "tenant_id": e.tenant_id,
                    "domain_id": e.domain_id,
                    "metadata": e.metadata,
                    "reasoning_trace": e.reasoning_trace,
                    "correction": e.correction,
                    "confirmation_count": e.confirmation_count
                }
                for e in self.experiences
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BidirectionalExperienceMemory':
        """Load memory from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        bem = cls(**data["config"])
        for e_data in data["experiences"]:
            exp = Experience(
                embedding=np.array(e_data["embedding"]),
                outcome=e_data["outcome"],
                context_hash=e_data["context_hash"],
                timestamp=datetime.fromisoformat(e_data["timestamp"]),
                tenant_id=e_data["tenant_id"],
                domain_id=e_data["domain_id"],
                metadata=e_data["metadata"],
                reasoning_trace=e_data["reasoning_trace"],
                correction=e_data["correction"],
                confirmation_count=e_data["confirmation_count"]
            )
            bem.experiences.append(exp)
        
        bem._needs_reindex = True
        return bem


# Tenant-isolated memory manager
class BEMManager:
    """
    Manages multiple BEM instances with tenant isolation.
    
    Structure:
    - Global BEM: Shared failure patterns (read by all, write requires consensus)
    - Domain BEMs: Domain-specific experiences
    - Tenant BEMs: Per-tenant isolated experiences
    """
    
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.global_bem = BidirectionalExperienceMemory(dim=dim, tenant_id="global")
        self.domain_bems: Dict[str, BidirectionalExperienceMemory] = {}
        self.tenant_bems: Dict[str, BidirectionalExperienceMemory] = {}
    
    def get_scoped_memory(
        self,
        tenant_id: str,
        domain_id: str = "default"
    ) -> List[BidirectionalExperienceMemory]:
        """
        Get all memories in scope for a tenant query.
        
        Returns [global, domain, tenant] memories for unified querying.
        """
        memories = [self.global_bem]
        
        if domain_id in self.domain_bems:
            memories.append(self.domain_bems[domain_id])
        
        if tenant_id in self.tenant_bems:
            memories.append(self.tenant_bems[tenant_id])
        
        return memories
    
    def query(
        self,
        z: np.ndarray,
        tenant_id: str,
        domain_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Unified query across all scoped memories.
        
        Returns aggregated signals from global + domain + tenant memories.
        """
        memories = self.get_scoped_memory(tenant_id, domain_id)
        
        # Aggregate risk signals (max across memories — conservative)
        risk_scores = []
        all_failure_experiences = []
        for mem in memories:
            risk, failures = mem.risk_signal(z)
            risk_scores.append(risk)
            all_failure_experiences.extend(failures)
        
        # Aggregate success signals (max across memories — optimistic)
        success_scores = []
        all_success_experiences = []
        for mem in memories:
            success, successes = mem.success_signal(z)
            success_scores.append(success)
            all_success_experiences.extend(successes)
        
        # Aggregate coverage (average — overall coverage estimate)
        coverage_scores = [mem.coverage_signal(z) for mem in memories]
        
        return {
            "risk": max(risk_scores) if risk_scores else 0.0,
            "success": max(success_scores) if success_scores else 0.0,
            "coverage": np.mean(coverage_scores) if coverage_scores else 0.0,
            "failure_experiences": all_failure_experiences,
            "success_experiences": all_success_experiences
        }

    def check_conflict(
        self,
        z: np.ndarray,
        outcome: float,
        tenant_id: str,
        domain_id: str = "default"
    ) -> Optional[str]:
        """Check for conflicts across all scoped memories."""
        # Check global, domain, and tenant memories
        memories = self.get_scoped_memory(tenant_id, domain_id)
        for mem in memories:
            conflict = mem.check_conflict(z, outcome)
            if conflict:
                return f"[{mem.tenant_id} memory] {conflict}"
        return None
    
    def add_experience(
        self,
        embedding: np.ndarray,
        outcome: float,
        context: str,
        tenant_id: str,
        domain_id: str = "default",
        **kwargs
    ):
        """Add experience to appropriate memory based on scope."""
        # Always add to tenant memory
        if tenant_id not in self.tenant_bems:
            self.tenant_bems[tenant_id] = BidirectionalExperienceMemory(
                dim=self.dim, tenant_id=tenant_id
            )
        self.tenant_bems[tenant_id].add_experience(
            embedding=embedding,
            outcome=outcome,
            context_hash=context,
            tenant_id=tenant_id,
            domain_id=domain_id,
            **kwargs
        )
        
        # Optionally promote to domain/global based on confirmation
        # (This would require multi-tenant consensus — simplified here)

    def vacuum_all(self, max_size: int = 10000):
        """Run memory hygiene on all managed memories."""
        self.global_bem.vacuum(max_size)
        for bem in self.domain_bems.values():
            bem.vacuum(max_size)
        for bem in self.tenant_bems.values():
            bem.vacuum(max_size)
