"""
Contrastive Projection Layer
============================

Projects backbone embeddings into a space where experiences with different
outcomes are separable. This enables BEM to effectively retrieve relevant
experiences based on semantic similarity in the projected space.

The projection is trained with a contrastive objective:
- Positive pairs: Same outcome polarity (both success or both failure)
- Negative pairs: Different outcome polarity
- Hard negatives: Semantically similar but different outcomes

Author: David Ahmann
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ContrastivePair:
    """A training pair for contrastive learning."""
    anchor: np.ndarray
    positive: np.ndarray
    negatives: List[np.ndarray]
    anchor_outcome: float
    positive_outcome: float


class ContrastiveProjection:
    """
    Two-layer MLP that projects backbone embeddings into experience-separable space.
    
    Architecture: h(x) → ReLU(W1 @ h + b1) → W2 @ ... + b2 → z
    
    The projection is trained to:
    1. Bring similar-outcome experiences closer
    2. Push different-outcome experiences apart
    3. Especially separate hard negatives (similar input, different outcome)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        temperature: float = 0.07
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        # Initialize weights (Xavier/Glorot)
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)
        
        # For training
        self._cache = {}
    
    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Project backbone embedding to experience space.
        
        Args:
            h: Backbone hidden state, shape (d_h,) or (batch, d_h)
        
        Returns:
            z: Projected embedding, shape (d_z,) or (batch, d_z)
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        # Layer 1: Linear + ReLU
        a1 = h @ self.W1.T + self.b1
        h1 = np.maximum(0, a1)  # ReLU
        
        # Layer 2: Linear
        z = h1 @ self.W2.T + self.b2
        
        # L2 normalize for cosine similarity
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-10)
        
        # Cache for backward pass
        self._cache = {'h': h, 'a1': a1, 'h1': h1}
        
        if squeeze:
            z = z.squeeze(0)
        
        return z
    
    def __call__(self, h: np.ndarray) -> np.ndarray:
        return self.forward(h)
    
    def contrastive_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: List[np.ndarray]
    ) -> float:
        """
        InfoNCE contrastive loss.
        
        L = -log(exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + Σ exp(sim(a,n)/τ)))
        """
        z_anchor = self.forward(anchor)
        z_positive = self.forward(positive)
        z_negatives = [self.forward(n) for n in negatives]
        
        # Similarities
        sim_pos = np.dot(z_anchor, z_positive) / self.temperature
        sim_negs = [np.dot(z_anchor, z_n) / self.temperature for z_n in z_negatives]
        
        # Numerator: exp(sim(a,p)/τ)
        numerator = np.exp(sim_pos)
        
        # Denominator: exp(sim(a,p)/τ) + Σ exp(sim(a,n)/τ)
        denominator = numerator + sum(np.exp(s) for s in sim_negs)
        
        # Loss: -log(numerator / denominator)
        loss = -np.log(numerator / (denominator + 1e-10))
        
        return float(loss)
    
    def train_step(
        self,
        pairs: List[ContrastivePair],
        learning_rate: float = 1e-3
    ) -> float:
        """
        One training step over a batch of contrastive pairs using analytical gradients.
        """
        total_loss = 0.0
        
        # Accumulate gradients
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        
        for pair in pairs:
            # Forward pass for all involved vectors
            # We need to cache intermediate values for backprop
            
            # Anchor
            z_a_unnorm = self._forward_unnorm(pair.anchor)
            z_a = z_a_unnorm / (np.linalg.norm(z_a_unnorm) + 1e-10)
            cache_a = self._cache.copy()
            
            # Positive
            z_p_unnorm = self._forward_unnorm(pair.positive)
            z_p = z_p_unnorm / (np.linalg.norm(z_p_unnorm) + 1e-10)
            cache_p = self._cache.copy()
            
            # Negatives
            z_ns_unnorm = []
            z_ns = []
            caches_n = []
            for n_emb in pair.negatives:
                z_n_unnorm = self._forward_unnorm(n_emb)
                z_n = z_n_unnorm / (np.linalg.norm(z_n_unnorm) + 1e-10)
                z_ns_unnorm.append(z_n_unnorm)
                z_ns.append(z_n)
                caches_n.append(self._cache.copy())
            
            # Compute similarities and probabilities
            sim_p = np.dot(z_a, z_p) / self.temperature
            sim_ns = [np.dot(z_a, z_n) / self.temperature for z_n in z_ns]
            
            # Softmax denominator
            exp_p = np.exp(sim_p)
            exp_ns = [np.exp(s) for s in sim_ns]
            denominator = exp_p + sum(exp_ns)
            
            # Loss
            loss = -np.log(exp_p / (denominator + 1e-10))
            total_loss += loss
            
            # Gradients w.r.t similarities
            # dL/d(sim_p) = -(1 - p_p)
            # dL/d(sim_n) = p_n
            p_p = exp_p / (denominator + 1e-10)
            d_sim_p = -(1.0 - p_p) / self.temperature
            
            d_sim_ns = [(e / (denominator + 1e-10)) / self.temperature for e in exp_ns]
            
            # Backprop gradients to embeddings
            # sim(a, b) = a^T b
            # d(sim)/da = b, d(sim)/db = a
            
            # Gradients w.r.t z vectors (normalized)
            dz_a = d_sim_p * z_p
            dz_p = d_sim_p * z_a
            
            for i, z_n in enumerate(z_ns):
                dz_a += d_sim_ns[i] * z_n
                dz_n = d_sim_ns[i] * z_a
                
                # Backprop through normalization and network for negative
                dz_n_unnorm = self._backward_norm(dz_n, z_ns_unnorm[i])
                gW1, gb1, gW2, gb2 = self._backward_network(dz_n_unnorm, caches_n[i])
                grad_W1 += gW1
                grad_b1 += gb1
                grad_W2 += gW2
                grad_b2 += gb2
            
            # Backprop through normalization and network for anchor and positive
            dz_a_unnorm = self._backward_norm(dz_a, z_a_unnorm)
            gW1, gb1, gW2, gb2 = self._backward_network(dz_a_unnorm, cache_a)
            grad_W1 += gW1
            grad_b1 += gb1
            grad_W2 += gW2
            grad_b2 += gb2
            
            dz_p_unnorm = self._backward_norm(dz_p, z_p_unnorm)
            gW1, gb1, gW2, gb2 = self._backward_network(dz_p_unnorm, cache_p)
            grad_W1 += gW1
            grad_b1 += gb1
            grad_W2 += gW2
            grad_b2 += gb2
        
        # Apply gradients
        n = len(pairs)
        self.W1 -= learning_rate * grad_W1 / n
        self.b1 -= learning_rate * grad_b1 / n
        self.W2 -= learning_rate * grad_W2 / n
        self.b2 -= learning_rate * grad_b2 / n
        
        return total_loss / n

    def _forward_unnorm(self, h: np.ndarray) -> np.ndarray:
        """Forward pass returning unnormalized output and caching intermediates."""
        if h.ndim == 1:
            h = h.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        # Layer 1: Linear + ReLU
        a1 = h @ self.W1.T + self.b1
        h1 = np.maximum(0, a1)  # ReLU
        
        # Layer 2: Linear
        z_unnorm = h1 @ self.W2.T + self.b2
        
        # Cache for backward pass
        self._cache = {'h': h, 'a1': a1, 'h1': h1}
        
        if squeeze:
            z_unnorm = z_unnorm.squeeze(0)
            
        return z_unnorm

    def _backward_norm(self, d_z: np.ndarray, z_unnorm: np.ndarray) -> np.ndarray:
        """Backprop through L2 normalization."""
        # z = x / ||x||
        # J = (I - z z^T) / ||x||
        if z_unnorm.ndim == 1:
            z_unnorm = z_unnorm.reshape(1, -1)
            d_z = d_z.reshape(1, -1)
            
        norm = np.linalg.norm(z_unnorm, axis=1, keepdims=True) + 1e-10
        z = z_unnorm / norm
        
        # d_z_unnorm = (d_z - (d_z . z) * z) / norm
        dot_prod = np.sum(d_z * z, axis=1, keepdims=True)
        d_z_unnorm = (d_z - dot_prod * z) / norm
        
        return d_z_unnorm.squeeze()

    def _backward_network(self, d_z_unnorm: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backprop through MLP layers."""
        h = cache['h']
        h1 = cache['h1']
        
        if d_z_unnorm.ndim == 1:
            d_z_unnorm = d_z_unnorm.reshape(1, -1)
            
        # Layer 2 gradients
        # z_unnorm = h1 @ W2.T + b2
        grad_W2 = d_z_unnorm.T @ h1
        grad_b2 = np.sum(d_z_unnorm, axis=0)
        d_h1 = d_z_unnorm @ self.W2
        
        # Layer 1 gradients (ReLU)
        # h1 = ReLU(a1)
        d_a1 = d_h1 * (cache['a1'] > 0)
        
        # a1 = h @ W1.T + b1
        grad_W1 = d_a1.T @ h
        grad_b1 = np.sum(d_a1, axis=0)
        
        return grad_W1, grad_b1, grad_W2, grad_b2

    def fit(self, embeddings: List[np.ndarray], outcomes: List[float], n_epochs: int = 100):
        """
        Train projection using outcome-weighted contrastive pairs.
        """
        # Construct positive/negative pairs
        pairs = construct_contrastive_pairs(embeddings, outcomes)
        
        # Train with SGD on InfoNCE loss
        for epoch in range(n_epochs):
            loss = self.train_step(pairs, learning_rate=1e-3)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}")
    
    def save(self, path: str):
        """Save projection weights."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            config=np.array([self.input_dim, self.hidden_dim, self.output_dim, self.temperature])
        )
    
    @classmethod
    def load(cls, path: str) -> 'ContrastiveProjection':
        """Load projection weights."""
        data = np.load(path)
        config = data['config']
        proj = cls(
            input_dim=int(config[0]),
            hidden_dim=int(config[1]),
            output_dim=int(config[2]),
            temperature=float(config[3])
        )
        proj.W1 = data['W1']
        proj.b1 = data['b1']
        proj.W2 = data['W2']
        proj.b2 = data['b2']
        return proj


def construct_contrastive_pairs(
    embeddings: List[np.ndarray],
    outcomes: List[float],
    n_negatives: int = 5,
    hard_negative_ratio: float = 0.3
) -> List[ContrastivePair]:
    """
    Construct training pairs from a dataset of embeddings and outcomes.
    
    Strategy:
    - Positive: Same outcome polarity, high similarity
    - Easy negatives: Random, different outcome polarity
    - Hard negatives: High similarity but different outcome polarity
    """
    pairs = []
    n = len(embeddings)
    
    # Precompute pairwise similarities
    emb_matrix = np.vstack(embeddings)
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10)
    similarities = emb_matrix @ emb_matrix.T
    
    for i in range(n):
        anchor = embeddings[i]
        anchor_outcome = outcomes[i]
        anchor_polarity = 1 if anchor_outcome > 0 else -1
        
        # Find positive: same polarity, most similar
        same_polarity = [j for j in range(n) if j != i and 
                        (outcomes[j] > 0) == (anchor_outcome > 0)]
        if not same_polarity:
            continue
        
        pos_sims = [(j, similarities[i, j]) for j in same_polarity]
        pos_idx = max(pos_sims, key=lambda x: x[1])[0]
        positive = embeddings[pos_idx]
        
        # Find negatives
        diff_polarity = [j for j in range(n) if (outcomes[j] > 0) != (anchor_outcome > 0)]
        if not diff_polarity:
            continue
        
        # Hard negatives: different polarity but similar embedding
        neg_sims = [(j, similarities[i, j]) for j in diff_polarity]
        neg_sims.sort(key=lambda x: x[1], reverse=True)
        
        n_hard = int(n_negatives * hard_negative_ratio)
        n_easy = n_negatives - n_hard
        
        hard_negs = [embeddings[j] for j, _ in neg_sims[:n_hard]]
        
        # Sample easy negatives from remaining pool
        easy_pool = neg_sims[n_hard:]
        if easy_pool and n_easy > 0:
            easy_indices = np.random.choice(len(easy_pool), size=min(n_easy, len(easy_pool)), replace=False)
            easy_negs = [embeddings[easy_pool[idx][0]] for idx in easy_indices]
        else:
            easy_negs = []
        
        negatives = hard_negs + easy_negs
        
        pairs.append(ContrastivePair(
            anchor=anchor,
            positive=positive,
            negatives=negatives,
            anchor_outcome=anchor_outcome,
            positive_outcome=outcomes[pos_idx]
        ))
    
    return pairs
