"""
Bounded Adapters with Experience Replay
=======================================

Lightweight trainable layers that enable continual learning without
catastrophic forgetting. Key properties:

1. Bounded updates: Gradient clipping + norm constraints prevent drift
2. Experience replay: Training batches sampled from BEM
3. Tenant isolation: Each tenant can have its own adapter
4. Stability guarantees: Provable bounds on parameter drift

The adapters sit between the projection layer and the final output,
allowing the system to adapt to deployment-specific patterns.

Author: David Ahmann
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Import from sibling modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from bem import Experience, BidirectionalExperienceMemory


@dataclass
class AdapterConfig:
    """Configuration for bounded adapter."""
    input_dim: int = 256          # From projection layer
    hidden_dim: int = 64          # Small for efficiency
    output_dim: int = 256         # Back to embedding space
    
    # Stability constraints
    max_grad_norm: float = 1.0    # Gradient clipping threshold
    max_param_norm: float = 10.0  # Maximum parameter norm
    learning_rate: float = 1e-4
    
    # Regularization
    l2_weight: float = 0.01       # L2 regularization
    ema_decay: float = 0.999      # EMA for stable updates


class BoundedAdapter(nn.Module):
    """
    A small MLP adapter with bounded updates, implemented in PyTorch.
    
    Architecture: z -> LayerNorm -> Linear -> ReLU -> Linear -> z + residual
    """
    
    def __init__(self, config: AdapterConfig, tenant_id: str = "global"):
        super().__init__()
        self.config = config
        self.tenant_id = tenant_id
        
        # Architecture
        self.norm = nn.LayerNorm(config.input_dim)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Initialize weights for near-identity start
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        
        # Optimization
        self.optimizer = optim.SGD(self.parameters(), lr=config.learning_rate)
        
        # EMA Shadow (store as separate model or just state dict?)
        # For simplicity, we'll maintain EMA weights manually if needed, 
        # but here we'll simplify: The LIVE model is used, EMA is optional enhancement.
        # Let's keep it simple: No separate EMA model instance for now, 
        # or just manual averaging.
        self.beta = config.ema_decay
        self.shadow = {}
        for name, param in self.named_parameters():
             if param.requires_grad:
                 self.shadow[name] = param.data.clone()
        
        # Tracking
        self.update_count = 0
        self.total_drift = 0.0
        self._initial_param_norm = self._compute_param_norm()

    def _compute_param_norm(self) -> float:
        """Compute total parameter norm."""
        total_norm = 0.0
        for p in self.parameters():
            total_norm += p.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual."""
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual
        
        # Normalize output (embedding space)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-10)
        return out
    
    def __call__(self, z: np.ndarray, use_ema: bool = False) -> np.ndarray:
        """Numpy-compatible call wrapper."""
        # Note: use_ema ignored here for simplicity unless we implement strict swap
        is_batch = (z.ndim > 1)
        with torch.no_grad():
            t_in = torch.from_numpy(z).float()
            if not is_batch:
                t_in = t_in.unsqueeze(0)
            
            t_out = self.forward(t_in)
            
            out = t_out.numpy()
            if not is_batch:
                out = out.squeeze(0)
            return out

    def train_step(
        self,
        experiences: List[Experience],
        projection_fn=None # Ignored, we consume embeddings directly
    ) -> float:
        """
        One training step using experience replay.
        """
        if not experiences:
            return 0.0
        
        self.train()
        self.optimizer.zero_grad()
        
        # Prepare batch
        embeddings = np.stack([e.embedding for e in experiences])
        outcomes = np.array([e.outcome for e in experiences])
        
        z = torch.from_numpy(embeddings).float()
        y = torch.from_numpy(outcomes).float()
        
        # Forward
        z_adapted = self.forward(z)
        target = z.clone().detach() # Target is original embedding
        
        # Loss: Cosine Similarity
        # sim = (z_adapted . target) / (|z_adapted| |target|)
        # Since forward() normalizes z_adapted and input z should be normalized...
        # Let's ensure target is normalized
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-10)
        
        sim = torch.sum(z_adapted * target, dim=-1) # Dot product of normalized vecs
        
        # Loss Logic:
        # If success (y > 0): Maximize sim -> Loss = 1 - sim
        # If failure (y < 0): Minimize sim -> Loss = max(0, sim + margin)
        
        # Vectorized loss
        loss_success = 1.0 - sim
        loss_failure = torch.clamp(sim + 0.5, min=0.0)
        
        # Select based on y > 0
        loss_per_sample = torch.where(y > 0, loss_success, loss_failure)
        base_loss = loss_per_sample.mean()
        
        # L2 Regularization (Weight Decay)
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param)
            
        total_loss = base_loss + self.config.l2_weight * l2_reg
        
        total_loss.backward()
        
        # Clip Gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        
        # Track drift before update (approximation using norm difference)
        with torch.no_grad():
             old_vec = torch.cat([p.view(-1) for p in self.parameters()])
        
        self.optimizer.step()
        
        # Project to Norm Ball (Constraint)
        with torch.no_grad():
            current_norm = self._compute_param_norm()
            if current_norm > self.config.max_param_norm:
                scale = self.config.max_param_norm / current_norm
                for p in self.parameters():
                    p.data.mul_(scale)
        
        # Track drift after update
        with torch.no_grad():
             new_vec = torch.cat([p.view(-1) for p in self.parameters()])
             drift = torch.norm(new_vec - old_vec).item()
             self.total_drift += drift
             
        # Update EMA Shadow
        # shadow = decay * shadow + (1 - decay) * new_param
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.beta).add_(param.data, alpha=(1.0 - self.beta))
        
        self.update_count += 1
        
        return base_loss.item() 

    def get_statistics(self) -> Dict[str, Any]:
        """Return adapter statistics."""
        return {
            "tenant_id": self.tenant_id,
            "update_count": self.update_count,
            "total_drift": self.total_drift,
            "current_param_norm": self._compute_param_norm(),
            "initial_param_norm": self._initial_param_norm,
            "relative_drift": self.total_drift / (self._initial_param_norm + 1e-10)
        }
    
    def save(self, path: str):
        """Save adapter."""
        torch.save({
            'state_dict': self.state_dict(),
            'shadow': self.shadow,
            'stats': self.get_statistics(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str, config: AdapterConfig) -> 'BoundedAdapter':
        """Load adapter."""
        ckpt = torch.load(path)
        adapter = cls(config, tenant_id=ckpt['stats']['tenant_id'])
        adapter.load_state_dict(ckpt['state_dict'])
        adapter.shadow = ckpt['shadow']
        adapter.update_count = ckpt['stats']['update_count']
        adapter.total_drift = ckpt['stats']['total_drift']
        return adapter


class AdapterManager:
    """
    Manages adapters across tiers: global, domain, tenant.
    """
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.global_adapter = BoundedAdapter(config, tenant_id="global")
        self.domain_adapters: Dict[str, BoundedAdapter] = {}
        self.tenant_adapters: Dict[str, BoundedAdapter] = {}
    
    def get_adapter_chain(
        self,
        tenant_id: str,
        domain_id: str = "default"
    ) -> List[BoundedAdapter]:
        """Get chain of adapters for a tenant."""
        chain = [self.global_adapter]
        
        if domain_id in self.domain_adapters:
            chain.append(self.domain_adapters[domain_id])
        
        if tenant_id in self.tenant_adapters:
            chain.append(self.tenant_adapters[tenant_id])
        
        return chain
    
    def forward(
        self,
        z: np.ndarray,
        tenant_id: str,
        domain_id: str = "default",
        use_ema: bool = True
    ) -> np.ndarray:
        """Apply all adapters in chain."""
        chain = self.get_adapter_chain(tenant_id, domain_id)
        
        for adapter in chain:
            z = adapter(z, use_ema=use_ema)
        
        return z
    
    def train_tenant_adapter(
        self,
        tenant_id: str,
        bem: BidirectionalExperienceMemory,
        projection_fn,
        batch_size: int = 32
    ) -> float:
        """Train a tenant's adapter using their BEM experiences."""
        if tenant_id not in self.tenant_adapters:
            self.tenant_adapters[tenant_id] = BoundedAdapter(
                self.config, tenant_id=tenant_id
            )
        
        experiences = bem.sample_for_training(batch_size)
        loss = self.tenant_adapters[tenant_id].train_step(experiences)
        
        return loss
