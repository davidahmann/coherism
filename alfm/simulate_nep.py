import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def simulate_nep(n_failures=1000, dim=768, noise_level=0.1):
    # Generate random failure modes (centroids)
    n_modes = 50
    modes = np.random.randn(n_modes, dim)
    modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    
    # Generate failure instances clustered around modes
    failures = []
    labels = []
    for _ in range(n_failures):
        mode_idx = np.random.randint(n_modes)
        noise = np.random.randn(dim) * noise_level
        vec = modes[mode_idx] + noise
        vec = vec / np.linalg.norm(vec)
        failures.append(vec)
        labels.append(mode_idx)
    failures = np.array(failures)
    
    # Generate non-failures (random uniform)
    n_non_failures = 1000
    non_failures = np.random.randn(n_non_failures, dim)
    non_failures = non_failures / np.linalg.norm(non_failures, axis=1, keepdims=True)
    
    # NEP: Store first 50% of failures
    train_size = int(n_failures * 0.5)
    nep_memory = failures[:train_size]
    
    # Test on remaining 50% failures + non-failures
    test_failures = failures[train_size:]
    test_set = np.vstack([test_failures, non_failures])
    test_labels = np.array([1] * len(test_failures) + [0] * len(non_failures))
    
    # Compute similarities
    sims = cosine_similarity(test_set, nep_memory)
    max_sims = np.max(sims, axis=1)
    
    # Compute Precision-Recall
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    
    for t in thresholds:
        preds = (max_sims > t).astype(int)
        tp = np.sum((preds == 1) & (test_labels == 1))
        fp = np.sum((preds == 1) & (test_labels == 0))
        fn = np.sum((preds == 0) & (test_labels == 1))
        
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        precisions.append(p)
        recalls.append(r)
        
    return precisions, recalls, thresholds

# Run simulation for Projected (Low Noise)
precisions_proj, recalls_proj, _ = simulate_nep(noise_level=0.1)

# Run simulation for Baseline (High Noise/Raw)
precisions_base, recalls_base, _ = simulate_nep(noise_level=0.3)

# Save data for TikZ
with open('nep_data.txt', 'w') as f:
    f.write("recall_proj precision_proj recall_base precision_base\n")
    # Downsample
    step = 2
    min_len = min(len(recalls_proj), len(recalls_base))
    for i in range(0, min_len, step):
        rp = recalls_proj[i]
        pp = precisions_proj[i]
        rb = recalls_base[i]
        pb = precisions_base[i]
        f.write(f"{rp:.4f} {pp:.4f} {rb:.4f} {pb:.4f}\n")

print("Data saved to nep_data.txt")
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, linewidth=2, label='NEP (Synthetic)')
plt.xlabel('Recall (Failure Detection)')
plt.ylabel('Precision')
plt.title('NEP Performance on Synthetic Failure Modes')
plt.grid(True)
plt.legend()
plt.savefig('nep_simulation.png')
print("Simulation complete. Plot saved to nep_simulation.png")
