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


class BoundedAdapter:
    """
    A small MLP adapter with bounded updates.
    
    Architecture: z → LayerNorm → Linear → ReLU → Linear → z + residual
    
    The residual connection ensures the adapter can start as identity
    and gradually learn corrections.
    """
    
    def __init__(self, config: AdapterConfig, tenant_id: str = "global"):
        self.config = config
        self.tenant_id = tenant_id
        
        # Initialize weights (small for near-identity start)
        scale = 0.01
        self.W1 = np.random.randn(config.hidden_dim, config.input_dim) * scale
        self.b1 = np.zeros(config.hidden_dim)
        self.W2 = np.random.randn(config.output_dim, config.hidden_dim) * scale
        self.b2 = np.zeros(config.output_dim)
        
        # LayerNorm parameters
        self.ln_gamma = np.ones(config.input_dim)
        self.ln_beta = np.zeros(config.input_dim)
        
        # EMA copies for stable inference
        self.W1_ema = self.W1.copy()
        self.W2_ema = self.W2.copy()
        
        # Tracking
        self.update_count = 0
        self.total_drift = 0.0
        self._initial_param_norm = self._compute_param_norm()
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return self.ln_gamma * x_norm + self.ln_beta
    
    def _compute_param_norm(self) -> float:
        """Compute total parameter norm."""
        return np.sqrt(
            np.sum(self.W1 ** 2) + np.sum(self.b1 ** 2) +
            np.sum(self.W2 ** 2) + np.sum(self.b2 ** 2)
        )
    
    def forward(self, z: np.ndarray, use_ema: bool = False) -> np.ndarray:
        """
        Forward pass with residual connection.
        
        Args:
            z: Input embedding from projection layer
            use_ema: Use EMA weights for more stable inference
        """
        W1 = self.W1_ema if use_ema else self.W1
        W2 = self.W2_ema if use_ema else self.W2
        
        # LayerNorm
        z_norm = self._layer_norm(z)
        
        # First layer
        h = z_norm @ W1.T + self.b1
        h = np.maximum(0, h)  # ReLU
        
        # Second layer
        delta = h @ W2.T + self.b2
        
        # Residual connection
        out = z + delta
        
        # Normalize output
        out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-10)
        
        return out
    
    def __call__(self, z: np.ndarray, use_ema: bool = False) -> np.ndarray:
        return self.forward(z, use_ema)
    
    def compute_loss(
        self,
        z: np.ndarray,
        target: np.ndarray,
        outcome: float
    ) -> float:
        """
        Compute loss for a single experience.
        
        For failures (outcome < 0): Push output away from target
        For successes (outcome > 0): Pull output toward target
        """
        z_adapted = self.forward(z)
        
        # Cosine similarity
        sim = np.dot(z_adapted, target) / (
            np.linalg.norm(z_adapted) * np.linalg.norm(target) + 1e-10
        )
        
        if outcome > 0:
            # Success: maximize similarity
            loss = 1.0 - sim
        else:
            # Failure: minimize similarity (push away)
            loss = max(0, sim + 0.5)  # Margin of 0.5
        
        # L2 regularization
        l2_loss = self.config.l2_weight * self._compute_param_norm()
        
        return float(loss + l2_loss)
    
    def _clip_gradients(
        self,
        grads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Clip gradients to max_grad_norm."""
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        
        if total_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / (total_norm + 1e-10)
            grads = {k: v * scale for k, v in grads.items()}
        
        return grads
    
    def _project_to_norm_ball(self):
        """Project parameters to norm ball."""
        current_norm = self._compute_param_norm()
        
        if current_norm > self.config.max_param_norm:
            scale = self.config.max_param_norm / current_norm
            self.W1 *= scale
            self.b1 *= scale
            self.W2 *= scale
            self.b2 *= scale
    
    def _update_ema(self):
        """Update EMA copies of parameters."""
        decay = self.config.ema_decay
        self.W1_ema = decay * self.W1_ema + (1 - decay) * self.W1
        self.W2_ema = decay * self.W2_ema + (1 - decay) * self.W2
    
    def train_step(
        self,
        experiences: List[Experience],
        projection_fn  # Callable to project raw embeddings
    ) -> float:
        """
        One training step using experience replay.
        
        Uses numerical gradients for simplicity.
        """
        if not experiences:
            return 0.0
        
        total_loss = 0.0
        grads = {
            'W1': np.zeros_like(self.W1),
            'W2': np.zeros_like(self.W2),
            'b1': np.zeros_like(self.b1),
            'b2': np.zeros_like(self.b2)
        }
        
        eps = 1e-5
        
        for exp in experiences:
            z = exp.embedding
            # Target: the original embedding (we want to correct it)
            target = z.copy()
            
            loss = self.compute_loss(z, target, exp.outcome)
            total_loss += loss
            
            # Numerical gradients for W2 (sample a few dimensions)
            for i in range(min(5, self.W2.shape[0])):
                for j in range(min(5, self.W2.shape[1])):
                    self.W2[i, j] += eps
                    loss_plus = self.compute_loss(z, target, exp.outcome)
                    self.W2[i, j] -= 2 * eps
                    loss_minus = self.compute_loss(z, target, exp.outcome)
                    self.W2[i, j] += eps
                    grads['W2'][i, j] += (loss_plus - loss_minus) / (2 * eps)
        
        # Average gradients
        n = len(experiences)
        grads = {k: v / n for k, v in grads.items()}
        
        # Clip gradients
        grads = self._clip_gradients(grads)
        
        # Record drift before update
        old_params = np.concatenate([
            self.W1.flatten(), self.b1, self.W2.flatten(), self.b2
        ])
        
        # Apply updates
        self.W1 -= self.config.learning_rate * grads['W1']
        self.b1 -= self.config.learning_rate * grads['b1']
        self.W2 -= self.config.learning_rate * grads['W2']
        self.b2 -= self.config.learning_rate * grads['b2']
        
        # Project to norm ball
        self._project_to_norm_ball()
        
        # Compute drift
        new_params = np.concatenate([
            self.W1.flatten(), self.b1, self.W2.flatten(), self.b2
        ])
        step_drift = np.linalg.norm(new_params - old_params)
        self.total_drift += step_drift
        
        # Update EMA
        self._update_ema()
        
        self.update_count += 1
        
        return total_loss / n
    
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
        """Save adapter weights."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W1_ema=self.W1_ema, W2_ema=self.W2_ema,
            ln_gamma=self.ln_gamma, ln_beta=self.ln_beta,
            update_count=self.update_count,
            total_drift=self.total_drift,
            tenant_id=self.tenant_id
        )
    
    @classmethod
    def load(cls, path: str, config: AdapterConfig) -> 'BoundedAdapter':
        """Load adapter weights."""
        data = np.load(path, allow_pickle=True)
        adapter = cls(config, tenant_id=str(data['tenant_id']))
        adapter.W1 = data['W1']
        adapter.b1 = data['b1']
        adapter.W2 = data['W2']
        adapter.b2 = data['b2']
        adapter.W1_ema = data['W1_ema']
        adapter.W2_ema = data['W2_ema']
        adapter.ln_gamma = data['ln_gamma']
        adapter.ln_beta = data['ln_beta']
        adapter.update_count = int(data['update_count'])
        adapter.total_drift = float(data['total_drift'])
        return adapter


class AdapterManager:
    """
    Manages adapters across tiers: global, domain, tenant.
    
    Provides unified forward pass that chains adapters.
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
        loss = self.tenant_adapters[tenant_id].train_step(experiences, projection_fn)
        
        return loss
