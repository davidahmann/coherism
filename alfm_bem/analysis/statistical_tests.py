"""
Statistical Significance Tests
==============================

Runs the ablation study multiple times (N=10) with different random seeds
to determine if performance differences are statistically significant.

Metrics:
- Failure F1 Score
- Success Retrieval Rate
- OOD AUC (Clustered)

Test: Welch's t-test (BEM vs RAG, BEM vs NEP).
"""

import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from ablation_study import run_ablation_study

def run_tests():
    print("Starting Statistical Significance Tests (N=10)...")
    
    n_runs = 10
    seeds = range(42, 42 + n_runs)
    
    # Storage for metrics
    metrics = {
        "Failure F1": {"BEM": [], "RAG": [], "NEP": []},
        "Success Rate": {"BEM": [], "RAG": [], "NEP": []},
        "OOD AUC": {"BEM": [], "RAG": [], "NEP": []}
    }
    
    for seed in seeds:
        print(f"  Run Seed={seed}...")
        res = run_ablation_study(seed=seed, verbose=False)
        
        # Extract metrics
        metrics["Failure F1"]["BEM"].append(res["BEM"]["failure_retrieval"]["f1"])
        metrics["Failure F1"]["RAG"].append(res["RAG"]["failure_retrieval"]["f1"])
        metrics["Failure F1"]["NEP"].append(res["NEP"]["failure_retrieval"]["f1"])
        
        metrics["Success Rate"]["BEM"].append(res["BEM"]["success_retrieval"]["success_rate"])
        metrics["Success Rate"]["RAG"].append(res["RAG"]["success_retrieval"]["success_rate"])
        metrics["Success Rate"]["NEP"].append(res["NEP"]["success_retrieval"]["success_rate"])
        
        metrics["OOD AUC"]["BEM"].append(res["BEM"]["ood_clustered"]["auc"])
        metrics["OOD AUC"]["RAG"].append(res["RAG"]["ood_clustered"]["auc"])
        metrics["OOD AUC"]["NEP"].append(res["NEP"]["ood_clustered"]["auc"])

        # Also capture Distributed OOD if interesting, but let's stick to Clustered (harder?)

    # Analysis
    print("\n" + "=" * 80)
    print("Statistical Significance Results (Mean ± Std, p-value vs BEM)")
    print("=" * 80)
    
    for metric_name, data in metrics.items():
        print(f"\nMetric: {metric_name}")
        
        bem_vals = data["BEM"]
        bem_mean = np.mean(bem_vals)
        bem_std = np.std(bem_vals)
        
        print(f"  BEM: {bem_mean:.3f} ± {bem_std:.3f}")
        
        for sys_name in ["RAG", "NEP"]:
            sys_vals = data[sys_name]
            sys_mean = np.mean(sys_vals)
            sys_std = np.std(sys_vals)
            
            # T-test
            t_stat, p_val = stats.ttest_ind(bem_vals, sys_vals, equal_var=False)
            
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {sys_name}: {sys_mean:.3f} ± {sys_std:.3f} (p={p_val:.4f} {sig})")

if __name__ == "__main__":
    run_tests()
