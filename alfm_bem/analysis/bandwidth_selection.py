"""
KDE Bandwidth Selection
=======================

Justifies the bandwidth parameter h for Kernel Density Estimation.
Currently hardcoded to 0.3. This script uses Cross-Validation to find the optimal h.

Method:
- Generate synthetic "In-Distribution" (ID) data (clustered mixture).
- Use GridSearchCV with KernelDensity estimator.
- Scoring: Log-Likelihood of held-out test data.

Goal: Find h that maximizes predictive log-likelihood.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import sys
from pathlib import Path

# Reuse data generation
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from ablation_study import generate_modes, normalize

def run_selection():
    print("Starting KDE Bandwidth Selection...")
    
    np.random.seed(42)
    dim = 64
    n_samples = 1000
    
    # Generate ID data (Mixture of Gaussians)
    # This represents the "Safe" region populated by BEM
    modes = generate_modes(10, dim)
    data = []
    
    for _ in range(n_samples):
        mode = modes[np.random.randint(len(modes))]
        # Spread should match "similarity_threshold" roughly?
        # If sim threshold is 0.5, then cos dist is big.
        # But here we model the underlying density BEM observes.
        # Let's assume standard deviation 0.1 (compact clusters)
        vec = mode + np.random.randn(dim) * 0.1
        data.append(normalize(vec))
        
    data = np.vstack(data)
    
    # Grid Search
    # Bandwidths to test: 0.1 to 1.0
    bandwidths = np.linspace(0.1, 1.0, 20)
    
    print(f"Grid searching {len(bandwidths)} bandwidths on shape {data.shape}...")
    
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': bandwidths},
        cv=5, # 5-fold Cross Validation
        n_jobs=-1 # Parallel
    )
    
    grid.fit(data)
    
    best_h = grid.best_params_['bandwidth']
    print("\n" + "="*40)
    print(f"Optimal Bandwidth: {best_h:.3f}")
    print("="*40)
    
    # Plot score vs bandwidth
    results = grid.cv_results_
    scores = results['mean_test_score']
    stds = results['std_test_score']
    
    plt.figure(figsize=(10, 6))
    plt.plot(bandwidths, scores, lw=2, label='Mean Log-Likelihood')
    plt.fill_between(bandwidths, scores - stds, scores + stds, alpha=0.3)
    plt.axvline(x=best_h, color='r', linestyle='--', label=f'Best h={best_h:.2f}')
    plt.axvline(x=0.3, color='g', linestyle=':', label='Current Default (0.3)')
    
    plt.xlabel('Bandwidth')
    plt.ylabel('Log-Likelihood Score')
    plt.title('KDE Bandwidth Selection (CV-Likelihood)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('bandwidth_selection.pdf')
    print("Saved bandwidth_selection.pdf")
    
    return best_h

if __name__ == "__main__":
    run_selection()
