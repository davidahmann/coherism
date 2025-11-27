import numpy as np

def simulate_drift(steps=1000, dim=256, lr=0.01, clip=0.1, bound=1.0):
    # Initialize adapter delta
    delta = np.zeros(dim)
    
    # Store drift norms
    drift_unbounded = []
    drift_bounded = []
    
    # Simulation 1: Unbounded (Standard SGD)
    d = np.zeros(dim)
    for _ in range(steps):
        grad = np.random.randn(dim) # Random gradient direction
        d = d - lr * grad
        drift_unbounded.append(np.linalg.norm(d))
        
    # Simulation 2: Bounded (ALFM with Clipping + Norm Constraint)
    d = np.zeros(dim)
    for _ in range(steps):
        grad = np.random.randn(dim)
        update = -lr * grad
        
        # Gradient Clipping
        if np.linalg.norm(update) > clip:
            update = update * (clip / np.linalg.norm(update))
            
        d = d + update
        
        # Norm Constraint (Projection)
        if np.linalg.norm(d) > bound:
            d = d * (bound / np.linalg.norm(d))
            
        drift_bounded.append(np.linalg.norm(d))
        
    return drift_unbounded, drift_bounded

# Run simulation
unbounded, bounded = simulate_drift()

# Save data for TikZ
with open('drift_data.txt', 'w') as f:
    f.write("step unbounded bounded\n")
    # Downsample
    step_size = 20
    for i in range(0, len(unbounded), step_size):
        f.write(f"{i} {unbounded[i]:.4f} {bounded[i]:.4f}\n")

print("Data saved to drift_data.txt")
