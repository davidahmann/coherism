"""
ALFM-BEM: Unified System
========================

The complete ALFM-BEM architecture that integrates:
1. Projection Layer: Backbone embeddings → experience-separable space
2. Bidirectional Experience Memory: Stores failures and successes
3. Consensus Engine: Decides Trust/Abstain/Escalate/Query
4. Bounded Adapters: Enable continual learning
5. Experience Loop: Closes the learning cycle

This file provides the unified API that orchestrates all components.

Author: David Ahmann
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from bem import BidirectionalExperienceMemory, BEMManager, Experience
from projection import ContrastiveProjection
from consensus import ConsensusEngine, ConsensusDecision, Action
from adapters import BoundedAdapter, AdapterConfig, AdapterManager


@dataclass
class ALFMConfig:
    """Configuration for the complete ALFM-BEM system."""
    
    # Dimensions
    backbone_dim: int = 768       # Input from backbone model
    projection_dim: int = 256     # BEM embedding dimension
    adapter_hidden_dim: int = 64  # Adapter hidden layer
    
    # BEM parameters
    similarity_threshold: float = 0.7
    risk_sensitivity: float = 0.8
    
    # Adapter parameters  
    max_grad_norm: float = 1.0
    max_param_norm: float = 10.0
    adapter_lr: float = 1e-4
    
    # Experience loop
    replay_batch_size: int = 32
    replay_frequency: int = 100   # Steps between adapter updates


@dataclass 
class InferenceResult:
    """Result of an ALFM-BEM inference."""
    decision: ConsensusDecision
    adapted_embedding: np.ndarray
    bem_query_result: Dict[str, Any]
    
    def should_trust(self) -> bool:
        return self.decision.action == Action.TRUST
    
    def should_query(self) -> bool:
        return self.decision.action == Action.QUERY
    
    def get_query(self) -> Optional[str]:
        return self.decision.query_content if self.should_query() else None


class ALFMBEM:
    """
    The unified ALFM-BEM system.
    
    Usage:
        system = ALFMBEM(config)
        
        # Inference
        result = system.infer(backbone_embedding, context, tenant_id)
        if result.should_trust():
            # Use backbone output
        elif result.should_query():
            # Request: result.get_query()
        
        # Record outcome (closes the loop)
        system.record_outcome(backbone_embedding, context, outcome, tenant_id)
    """
    
    def __init__(self, config: ALFMConfig):
        self.config = config
        
        # Initialize components
        self.projection = ContrastiveProjection(
            input_dim=config.backbone_dim,
            hidden_dim=512,
            output_dim=config.projection_dim
        )
        
        self.bem_manager = BEMManager(dim=config.projection_dim)
        
        self.consensus = ConsensusEngine()
        
        adapter_config = AdapterConfig(
            input_dim=config.projection_dim,
            hidden_dim=config.adapter_hidden_dim,
            output_dim=config.projection_dim,
            max_grad_norm=config.max_grad_norm,
            max_param_norm=config.max_param_norm,
            learning_rate=config.adapter_lr
        )
        self.adapter_manager = AdapterManager(adapter_config)
        
        # Experience loop tracking
        self._inference_count = 0
        self._pending_outcomes: List[Dict] = []
    
    def infer(
        self,
        backbone_embedding: np.ndarray,
        context: Dict[str, Any],
        tenant_id: str,
        domain_id: str = "default"
    ) -> InferenceResult:
        """
        Run inference through the complete ALFM-BEM pipeline.
        
        Args:
            backbone_embedding: Hidden state from frozen backbone
            context: Context dict for heuristic rules
            tenant_id: Tenant identifier for isolation
            domain_id: Domain identifier
        
        Returns:
            InferenceResult with decision and all signals
        """
        # 1. Project to experience-separable space
        z = self.projection(backbone_embedding)
        
        # 2. Apply adapters (tenant-specific corrections)
        z_adapted = self.adapter_manager.forward(z, tenant_id, domain_id)
        
        # 3. Query BEM for risk/success/coverage signals
        bem_result = self.bem_manager.query(z_adapted, tenant_id, domain_id)
        
        # 4. Make decision via Consensus Engine
        decision = self.consensus.decide(
            risk=bem_result["risk"],
            success=bem_result["success"],
            coverage=bem_result["coverage"],
            context=context
        )
        
        self._inference_count += 1
        
        return InferenceResult(
            decision=decision,
            adapted_embedding=z_adapted,
            bem_query_result=bem_result
        )
    
    def record_outcome(
        self,
        backbone_embedding: np.ndarray,
        context: str,
        outcome: float,
        tenant_id: str,
        domain_id: str = "default",
        reasoning_trace: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """
        Record the outcome of an inference, closing the experience loop.
        
        This:
        1. Adds the experience to BEM
        2. Optionally triggers adapter training via experience replay
        
        Args:
            backbone_embedding: Original backbone embedding
            context: String context (for deduplication)
            outcome: Outcome score in [-1, 1]
            tenant_id: Tenant identifier
            domain_id: Domain identifier
            reasoning_trace: Optional reasoning that led to outcome
            correction: Optional human correction for failures
        """
        # Project embedding
        z = self.projection(backbone_embedding)
        z_adapted = self.adapter_manager.forward(z, tenant_id, domain_id)
        
        # Add to tenant BEM
        self.bem_manager.add_experience(
            embedding=z_adapted,
            outcome=outcome,
            context=context,
            tenant_id=tenant_id,
            domain_id=domain_id,
            reasoning_trace=reasoning_trace,
            correction=correction
        )
        
        # Experience replay: periodically train adapters
        if self._inference_count % self.config.replay_frequency == 0:
            self._run_experience_replay(tenant_id, domain_id)
    
    def _run_experience_replay(self, tenant_id: str, domain_id: str):
        """Train adapter using accumulated experiences."""
        if tenant_id not in self.bem_manager.tenant_bems:
            return
        
        bem = self.bem_manager.tenant_bems[tenant_id]
        if len(bem.experiences) < self.config.replay_batch_size:
            return
        
        # Train tenant adapter
        loss = self.adapter_manager.train_tenant_adapter(
            tenant_id=tenant_id,
            bem=bem,
            projection_fn=self.projection,
            batch_size=self.config.replay_batch_size
        )
        
        return loss
    
    def add_heuristic_rule(
        self,
        name: str,
        check_fn: Callable[[Dict], bool],
        is_critical: bool = False,
        description: str = ""
    ):
        """Add a heuristic rule to the Consensus Engine."""
        self.consensus.add_heuristic_rule(name, check_fn, is_critical, description)
    
    def get_statistics(self, tenant_id: str = "global") -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "inference_count": self._inference_count,
            "projection": {
                "input_dim": self.config.backbone_dim,
                "output_dim": self.config.projection_dim
            }
        }
        
        # BEM stats
        if tenant_id in self.bem_manager.tenant_bems:
            stats["bem"] = self.bem_manager.tenant_bems[tenant_id].get_statistics()
        else:
            stats["bem"] = {"total": 0}
        
        # Adapter stats
        if tenant_id in self.adapter_manager.tenant_adapters:
            stats["adapter"] = self.adapter_manager.tenant_adapters[tenant_id].get_statistics()
        else:
            stats["adapter"] = {"update_count": 0}
        
        return stats
    
    def save(self, path: str):
        """Save complete system state."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.projection.save(f"{path}/projection.npz")
        self.bem_manager.global_bem.save(f"{path}/bem_global.json")
        
        for tid, bem in self.bem_manager.tenant_bems.items():
            bem.save(f"{path}/bem_{tid}.json")
        
        for tid, adapter in self.adapter_manager.tenant_adapters.items():
            adapter.save(f"{path}/adapter_{tid}.npz")
    
    @classmethod
    def load(cls, path: str, config: ALFMConfig) -> 'ALFMBEM':
        """Load system from saved state."""
        system = cls(config)
        
        system.projection = ContrastiveProjection.load(f"{path}/projection.npz")
        system.bem_manager.global_bem = BidirectionalExperienceMemory.load(
            f"{path}/bem_global.json"
        )
        
        # Load tenant BEMs and adapters as needed
        # (Would scan directory for tenant-specific files)
        
        return system


# Convenience function for quick setup
def create_alfm_bem(
    backbone_dim: int = 768,
    projection_dim: int = 256
) -> ALFMBEM:
    """Create an ALFM-BEM system with default configuration."""
    config = ALFMConfig(
        backbone_dim=backbone_dim,
        projection_dim=projection_dim
    )
    return ALFMBEM(config)
