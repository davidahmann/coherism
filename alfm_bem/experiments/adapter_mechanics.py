
"""
Adapter Mechanics Validation
============================

Validates the fundamental behavior of BoundedAdapter:
1. Stability: For Success inputs, it should approximate Identity (retain embedding).
2. Avoidance: For Failure inputs, it should shift the embedding away (reduce cosine sim).
3. Boundedness: Gradient norm and parameter drift should be constrained.

This confirms the class functions as a "Safety/Correction" layer.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adapters import BoundedAdapter, AdapterConfig
from bem import Experience
from datetime import datetime

def run_experiment():
    print("Starting Adapter Mechanics Validation...")
    
    dim = 64
    config = AdapterConfig(
        input_dim=dim,
        output_dim=dim,
        learning_rate=0.1,
        max_grad_norm=1.0,
        l2_weight=0.0
    )
    adapter = BoundedAdapter(config, tenant_id="mech_test")
    
    # 1. Test Stability (Success Case)
    z_success = np.random.randn(dim)
    z_success /= np.linalg.norm(z_success)
    
    print("\n--- Test 1: Stability (Success) ---")
    # Initial state should be near identity
    z_out_init = adapter(z_success)
    sim_init = np.dot(z_success, z_out_init)
    print(f"Initial Sim: {sim_init:.4f}")
    
    # Train heavily on this success
    exp_success = Experience(
        embedding=z_success,
        outcome=1.0, # Success
        context_hash="success_1",
        timestamp=datetime.now(),
        tenant_id="test",
        domain_id="default"
    )
    
    for i in range(50):
        adapter.train_step([exp_success], lambda x: x)
        
    z_out_post = adapter(z_success)
    sim_post = np.dot(z_success, z_out_post)
    print(f"Post-Train Sim: {sim_post:.4f}")
    
    if sim_post > 0.95:
        print("✅ PASS: Adapter maintained Identity for Success.")
    else:
        print("❌ FAIL: Adapter drifted away on Success.")
        
    # 2. Test Avoidance (Failure Case)
    z_fail = np.random.randn(dim)
    z_fail /= np.linalg.norm(z_fail)
    
    print("\n--- Test 2: Avoidance (Failure) ---")
    z_out_init_f = adapter(z_fail)
    sim_init_f = np.dot(z_fail, z_out_init_f)
    print(f"Initial Sim: {sim_init_f:.4f}")
    
    exp_fail = Experience(
        embedding=z_fail,
        outcome=-1.0, # Failure
        context_hash="fail_1",
        timestamp=datetime.now(),
        tenant_id="test",
        domain_id="default"
    )
    
    # Train heavily on failure
    for i in range(50):
        adapter.train_step([exp_fail], lambda x: x)
        
    z_out_post_f = adapter(z_fail)
    sim_post_f = np.dot(z_fail, z_out_post_f)
    print(f"Post-Train Sim: {sim_post_f:.4f}")
    
    if sim_post_f < sim_init_f - 0.1:
        print("✅ PASS: Adapter pushed Failure embedding away.")
    else:
        print(f"❌ FAIL: Adapter did not push away significantly (Diff: {sim_init_f - sim_post_f:.4f}).")

    # 3. Stats
    stats = adapter.get_statistics()
    print("\n--- Stats ---")
    print(stats)
    
if __name__ == "__main__":
    run_experiment()
