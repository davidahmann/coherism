"""
Sensitivity Analysis
====================

Grid search to explore the interaction between:
1. Similarity Threshold (S): Filters which neighbors are considered relevant.
2. Risk Sensitivity (R): Weights the impact of retrieved failures.

Metric: Failure F1 Score.
Goal: Find optimal hyperparameters for "Overlapping" scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Insert experiments to reuse data gen
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from bem import BidirectionalExperienceMemory
from ablation_study import generate_modes, generate_overlapping_experiences

def run_analysis():
    print("Starting Sensitivity Analysis (Grid Search)...")
    
    dim = 64
    np.random.seed(42)
    
    # Generate Data
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
    
    # Populate BEM once
    bem = BidirectionalExperienceMemory(dim=dim, similarity_threshold=0.5)
    for i, (emb, out) in enumerate(zip(train_emb, train_out)):
        bem.add_experience(emb, out, f"exp_{i}")
        
    # Grid
    sim_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    risk_sensitivities = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    heatmap = np.zeros((len(sim_thresholds), len(risk_sensitivities)))
    
    best_f1 = 0.0
    best_params = (0, 0)
    
    print(f"Grid: {len(sim_thresholds)}x{len(risk_sensitivities)} = {heatmap.size} points")
    
    for i, sim_th in enumerate(sim_thresholds):
        for j, risk_sens in enumerate(risk_sensitivities):
            # Update BEM params
            bem.similarity_threshold = sim_th
            bem.risk_sensitivity = risk_sens
            
            # Evaluate F1
            tp, fp, fn = 0, 0, 0
            
            # Failures (should be high risk)
            for z in test_failures:
                risk, _ = bem.risk_signal(z)
                if risk > 0.3: tp += 1
                else: fn += 1
                
            # Successes (should be low risk)
            for z in test_successes:
                risk, _ = bem.risk_signal(z)
                if risk > 0.3: fp += 1
                
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            heatmap[i, j] = f1
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = (sim_th, risk_sens)
                
            # print(f"  S={sim_th}, R={risk_sens} -> F1={f1:.3f}")

    print(f"\nBest F1: {best_f1:.3f} at Sim={best_params[0]}, Sens={best_params[1]}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Failure F1 Score')
    
    plt.xticks(range(len(risk_sensitivities)), risk_sensitivities)
    plt.yticks(range(len(sim_thresholds)), sim_thresholds)
    
    plt.xlabel('Risk Sensitivity')
    plt.ylabel('Similarity Threshold')
    plt.title(f'BEM Sensitivity Analysis (Best F1={best_f1:.2f})')
    
    plt.savefig('sensitivity_heatmap.pdf')
    print("Saved sensitivity_heatmap.pdf")

if __name__ == "__main__":
    run_analysis()
