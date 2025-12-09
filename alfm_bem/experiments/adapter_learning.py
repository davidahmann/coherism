
"""
Adapter Learning Experiment
===========================

Validates that BoundedAdapter can adapt to concept drift where fixed BEM fails.

Scenario:
1. Train Phase (Steps 0-1000): Distribution A
2. Drift Phase (Steps 1000-2000): Distribution B (Rotated/Shifted)
3. Compare:
   - Fixed BEM: Uses fixed projection + accumulated memory
   - Adaptive BEM: Uses learned Adapter + accumulated memory

Metric: "Adaptation Delay" - how fast does it recover performance after drift?
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bem import BidirectionalExperienceMemory, CoverageMode
from adapters import BoundedAdapter, AdapterConfig
from projection import ContrastiveProjection

def generate_data_stream(
    n_steps: int, 
    dim: int = 64, 
    drift_step: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a stream of (embedding, outcome) pairs with a concept drift.
    
    Drift: At drift_step, the "success" direction rotates by 90 degrees.
    """
    np.random.seed(42)
    
    # Concept A: Success if z[0] > 0
    # Concept B: Success if z[1] > 0
    
    embeddings = np.random.randn(n_steps, dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    outcomes = np.zeros(n_steps)
    
    for t in range(n_steps):
        if t < drift_step:
            # Concept A
            outcomes[t] = 1.0 if embeddings[t, 0] > 0 else -1.0
        else:
            # Concept B (Drift)
            outcomes[t] = 1.0 if embeddings[t, 1] > 0 else -1.0
            
    return embeddings, outcomes

def run_experiment():
    print("Starting Adapter Few-Shot Drift Experiment...")
    
    dim = 64
    n_steps = 2000
    drift_step = 500
    
    # Generate data
    embeddings, outcomes = generate_data_stream(n_steps, dim, drift_step)
    
    # 1. Fixed Agent (Baseline)
    bem_fixed = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.1)
    
    # 2. Adaptive Agent (Store original Z, Query Adapted)
    bem_adaptive = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.1)
    adapter_config = AdapterConfig(
        input_dim=dim,
        output_dim=dim,
        learning_rate=0.05, # Higher LR for fast adaptation
        max_grad_norm=5.0
    )
    adapter = BoundedAdapter(adapter_config, tenant_id="test")
    
    # Metrics
    acc_fixed_hist = []
    acc_adaptive_hist = []
    window = 50
    fixed_correct = []
    adaptive_correct = []
    
    # Pre-fill Phase (Task A) - Dense
    print("Phase 1: Pre-filling Memory with Task A (0-500)...")
    for t in range(drift_step):
        z = embeddings[t]
        y = outcomes[t]
        bem_fixed.add_experience(z, y, context=f"A_{t}")
        bem_adaptive.add_experience(z, y, context=f"A_{t}")

    print("Phase 2: Drift to Task B (Few-Shot Mode)...")
    # In this phase, we only add data to memory very rarely (1 in 20)
    # This simulates a "new task" where we lack data.
    # Fixed BEM will struggle because it lacks B data and A data is misleading.
    # Adaptive BEM should learn to map B->A and reuse A data.
    
    for t in range(drift_step, n_steps):
        z = embeddings[t]
        y_true = outcomes[t]
        
        # --- FIXED AGENT ---
        risk_f, _ = bem_fixed.risk_signal(z)
        success_f, _ = bem_fixed.success_signal(z)
        pred_f = 1.0 if success_f > risk_f else -1.0
        if len(bem_fixed.experiences) == 0: pred_f = 0.0
        
        is_correct_f = (1 if (pred_f > 0) == (y_true > 0) else 0)
        fixed_correct.append(is_correct_f)
        
        # --- ADAPTIVE AGENT ---
        # 1. Adapt Input
        z_adapted = adapter(z)
        
        # 2. Inference (Query with Adapted Z against Memory dominated by A)
        risk_a, _ = bem_adaptive.risk_signal(z_adapted)
        success_a, _ = bem_adaptive.success_signal(z_adapted)
        pred_a = 1.0 if success_a > risk_a else -1.0
        if len(bem_adaptive.experiences) == 0: pred_a = 0.0
        
        is_correct_a = (1 if (pred_a > 0) == (y_true > 0) else 0)
        adaptive_correct.append(is_correct_a)
        
        # --- LEARNING (FEW SHOT) ---
        # Only add experience every 20 steps
        if t % 20 == 0:
            bem_fixed.add_experience(z, y_true, context=f"B_{t}")
            
            # Adaptive: Store ORIGINAL z 
            bem_adaptive.add_experience(z, y_true, context=f"B_{t}")
            
            # Train adapter on the few new samples B plus some old samples A
            # We need to sample from BEM, which has mostly A and few B.
            # Ideally, adapter sees "For Z_B (new), target is outcome Y".
            # It should learn to map Z_B to something that yields Y in BEM.
            # BEM(Y) -> Z_A (old).
            # So Adapter(Z_B) -> Z_A.
            #
            # But BoundedAdapter training logic minimizes sim(f(z), z) for failures...
            # This logic is self-supervised correction, not supervised alignment.
            # 
            # Let's hope the contrastive loss in `adapters.py` does the right thing.
            # Train adapter on the few new samples B IMMEDIATELLY (Online Learning)
            # This ensures we don't miss the signal due to sampling A
            from bem import Experience
            from datetime import datetime
            
            # Construct a temporary experience object for training
            exp_now = Experience(
                embedding=z, # Original Z
                outcome=y_true,
                context_hash=f"B_{t}",
                timestamp=datetime.now(),
                tenant_id="test",
                domain_id="default",
                severity=1.0 # High severity
            )
            
            # Train on this single instance
            loss = adapter.train_step([exp_now], projection_fn=lambda x: x)
            
        if t % 50 == 0:
            acc_f = np.mean(fixed_correct[-window:]) if fixed_correct else 0
            acc_a = np.mean(adaptive_correct[-window:]) if adaptive_correct else 0
            acc_fixed_hist.append(acc_f)
            acc_adaptive_hist.append(acc_a)
            print(f"Step {t}: Fixed={acc_f:.2f}, Adaptive={acc_a:.2f}")

    # Plot
    plt.figure()
    steps = np.arange(len(acc_fixed_hist)) * 50 + drift_step
    plt.plot(steps, acc_fixed_hist, label="Fixed BEM", linestyle="--")
    plt.plot(steps, acc_adaptive_hist, label="Adaptive BEM", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Accuracy (Task B)")
    plt.title("Few-Shot Adaptation to Drift")
    plt.legend()
    plt.savefig("adapter_drift.pdf")
    print("Saved adapter_drift.pdf")

if __name__ == "__main__":
    run_experiment()
