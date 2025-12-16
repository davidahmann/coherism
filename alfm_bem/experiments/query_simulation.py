
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed
np.random.seed(42)

# --- SETUP ---
# We simulate a "Decision Boundary" problem.
# Inputs are 2D points.
# Decision boundary is a circle: Inside = Class 1, Outside = Class 0.
# The "Ambiguity Zone" is the edge of the circle.

N_SAMPLES = 1000
RADIUS = 1.0
AMBIGUITY_WIDTH = 0.2 # Width of the fuzzy boundary

# Generate Data
X = np.random.randn(N_SAMPLES, 2)
radii = np.linalg.norm(X, axis=1)

# True Labels (deterministic)
y_true = (radii <= RADIUS).astype(int)

# --- MODEL SIMULATION ---
# Simulate a model that is uncertain near the boundary
# Probability of correct classification drops near radius = 1.0

def model_confidence(r):
    # Confidence is 1.0 far from boundary, drops to 0.5 at boundary
    dist_from_boundary = np.abs(r - RADIUS)
    conf = 0.5 + 0.5 * np.minimum(dist_from_boundary / AMBIGUITY_WIDTH, 1.0)
    return conf

confidences = model_confidence(radii)

# Simulate Model Predictions (correct with probability = confidence)
random_draws = np.random.rand(N_SAMPLES)
y_pred = np.where(random_draws < confidences, y_true, 1 - y_true)

# --- STRATEGIES ---

# 1. BASELINE: Trust Model (No memory, no query)
# If model is wrong, we fail.
acc_baseline = np.mean(y_pred == y_true)

# 2. ABSTAIN: If confidence < Threshold, Abstain
# "Safe" but low coverage
THRESHOLD = 0.7
mask_abstain = confidences < THRESHOLD
n_abstained = np.sum(mask_abstain)
# Accuracy on non-abstained
acc_abstain_policy = np.mean(y_pred[~mask_abstain] == y_true[~mask_abstain])

# 3. QUERY: If confidence < Threshold, Ask User
# Cost: Asking the user is expensive (simulated simply)
# Benefit: User gives label with 100% accuracy (or high accuracy)
# We limit Query budget to TOP Uncertainty

# Sort by uncertainty (closest to 0.5)
uncertainty = np.abs(confidences - 0.5)
sort_idx = np.argsort(uncertainty) # ascending uncertainty

QUERY_BUDGET = int(0.15 * N_SAMPLES) # Can query 15% of samples
query_indices = sort_idx[:QUERY_BUDGET] # Most uncertain samples

# Apply Query
y_final_query = y_pred.copy()
y_final_query[query_indices] = y_true[query_indices] # User corrects them

acc_query_policy = np.mean(y_final_query == y_true)

# --- RESULTS ---

print("--- Query Action Simulation Results ---")
print(f"Total Samples: {N_SAMPLES}")
print(f"Ambiguity Width: {AMBIGUITY_WIDTH}")
print(f"Baseline Accuracy (Trust All): {acc_baseline:.4f}")
print(f"Abstain Policy Accuracy (Trust High Conf): {acc_abstain_policy:.4f} (Abstained: {n_abstained})")
print(f"Query Policy Accuracy (Active Correction): {acc_query_policy:.4f} (Queries: {QUERY_BUDGET})")
print(f"Improvement over Baseline: {((acc_query_policy - acc_baseline)/acc_baseline)*100:.1f}%")

# Generate Plot
plt.figure(figsize=(10, 5))

# Plot 1: Accuracy Comparison
plt.subplot(1, 2, 1)
bars = plt.bar(['Trust All', 'Abstain', 'Query'], [acc_baseline, acc_abstain_policy, acc_query_policy], 
        color=['gray', 'orange', 'green'])
plt.ylim(0.8, 1.0)
plt.title('Accuracy by Strategy')
plt.ylabel('Accuracy')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

# Plot 2: Ambiguity vs Confidence
plt.subplot(1, 2, 2)
plt.scatter(radii, confidences, c=confidences, cmap='coolwarm', s=10, alpha=0.5)
plt.axvline(RADIUS, color='black', linestyle='--')
plt.xlabel('Distance from Origin')
plt.ylabel('Model Confidence')
plt.title('Simulated Confidence Profile')

plt.tight_layout()
plt.savefig('query_simulation.pdf')
print("Saved query_simulation.pdf")
