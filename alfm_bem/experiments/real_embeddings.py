
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- CONFIG ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Small, fast, effective
PROJECTION_DIM = 64
BATCH_SIZE = 32
EPOCHS = 20  # Fast training for demonstration

print(f"Loading {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)

# --- DATA PREP ---
# We use 20Newsgroups to simulate "Domains"
# ID (In-Distribution): 'sci.space', 'sci.med', 'sci.electronics' (Science domain)
# OOD (Out-of-Distribution): 'rec.sport.hockey', 'rec.sport.baseball' (Sports domain)
# FAILURE MODES: We simulate failure modes by mapping specific sub-topics to failures.

categories = ['sci.space', 'sci.med', 'sci.electronics', 'rec.sport.hockey', 'rec.sport.baseball']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

texts = data.data
labels = data.target
target_names = data.target_names

print(f"Loaded {len(texts)} documents.")

# Split ID and OOD
id_indices = [i for i, label in enumerate(labels) if target_names[label].startswith('sci')]
ood_indices = [i for i, label in enumerate(labels) if target_names[label].startswith('rec')]

id_texts = [texts[i] for i in id_indices]
id_labels = [labels[i] for i in id_indices]
ood_texts = [texts[i] for i in ood_indices]

# Encode all texts (this might take a minute)
print("Encoding texts...")
id_embeddings = encoder.encode(id_texts, show_progress_bar=True)
ood_embeddings = encoder.encode(ood_texts, show_progress_bar=True)

# --- SIMULATE OUTCOMES ---
# We simulate that the model "fails" on specific sub-topics within ID (e.g., questions about "orbit" in science)
# This creates semantic failure modes.

failures = []
successes = []

for text, emb in zip(id_texts, id_embeddings):
    # Simulate: Fails on 'orbit' or 'circuit' related queries (simulating specific technical gaps)
    if 'orbit' in text.lower() or 'circuit' in text.lower():
        failures.append(emb)
    else:
        successes.append(emb) # Successes

failures = np.array(failures)
successes = np.array(successes)
ood_embeddings = np.array(ood_embeddings)

print(f"ID Failures: {len(failures)}")
print(f"ID Successes: {len(successes)}")
print(f"OOD Samples: {len(ood_embeddings)}")

# Split Train/Test for Validation
# We train the projection layer on a subset of failures/successes
train_fail, test_fail = train_test_split(failures, test_size=0.2)
train_succ, test_succ = train_test_split(successes, test_size=0.2)

# --- PROJECTION LAYER & CONTRASTIVE TRAINING ---
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=384, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# Contrastive Loss: Pull failures together, push failures away from successes
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, proj_fail, proj_succ, proj_fail_pair):
        # 1. Positive pairs (Fail-Fail): Minimize distance
        # Ideally we'd have labels for *which* failure mode, but here we assume general failure similarity
        # A better heuristic: Failures should be closer to *other failures* than to successes
        
        # Loss 1: Pull similar failures together (Simulated by random pairings of failures)
        dist_pos = torch.nn.functional.pairwise_distance(proj_fail, proj_fail_pair)
        loss_pos = torch.mean(torch.pow(dist_pos, 2))
        
        # Loss 2: Push Successes away from Failures
        dist_neg = torch.nn.functional.pairwise_distance(proj_fail, proj_succ)
        loss_neg = torch.mean(torch.nn.functional.relu(self.margin - dist_neg).pow(2))
        
        return loss_pos + loss_neg

# Training Loop
model = ProjectionLayer(input_dim=384, output_dim=PROJECTION_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss(margin=1.0)

# Prepare batches
min_len = min(len(train_fail), len(train_succ))
train_fail = train_fail[:min_len]
train_succ = train_succ[:min_len]
# Create pairs for positive loss
train_fail_pair = np.roll(train_fail, 1, axis=0) # Simple shift to get different failure as pair

inputs_fail = torch.tensor(train_fail).float()
inputs_succ = torch.tensor(train_succ).float()
inputs_fail_pair = torch.tensor(train_fail_pair).float()

print("Training Projection Layer...")
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    p_fail = model(inputs_fail)
    p_succ = model(inputs_succ)
    p_fail_pair = model(inputs_fail_pair)
    
    loss = criterion(p_fail, p_succ, p_fail_pair)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# --- EVALUATION ---
model.eval()
with torch.no_grad():
    z_fail = model(torch.tensor(test_fail).float()).numpy()
    z_succ = model(torch.tensor(test_succ).float()).numpy()
    z_ood = model(torch.tensor(ood_embeddings).float()).numpy()
    
    # Raw embeddings for baseline comparison
    raw_fail = test_fail
    raw_succ = test_succ
    raw_ood = ood_embeddings

# 1. Retrieval Capability (Can we find past failures?)
# Success metric: Failures should retrieve other Failures, not Successes
from sklearn.neighbors import NearestNeighbors

def eval_retrieval(query_set, index_set, index_labels, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(index_set)
    distances, indices = nbrs.kneighbors(query_set)
    
    precision_scores = []
    for i in range(len(query_set)):
        retrieved_labels = index_labels[indices[i]]
        precision = np.mean(retrieved_labels == 1) # 1 = Failure
        precision_scores.append(precision)
    return np.mean(precision_scores)

# Index: Mix of Fail (1) and Success (0) from TRAINING set
index_set_proj = model(torch.tensor(np.concatenate([train_fail, train_succ])).float()).detach().numpy()
index_labels = np.array([1]*len(train_fail) + [0]*len(train_succ))

index_set_raw = np.concatenate([train_fail, train_succ])

# Query: Test Failures
print("\n--- Retrieval Evaluation (Precision@5) ---")
prec_proj = eval_retrieval(z_fail, index_set_proj, index_labels)
prec_raw = eval_retrieval(raw_fail, index_set_raw, index_labels)
print(f"Raw Embeddings (384D): {prec_raw:.3f}")
print(f"BEM Projected (64D):   {prec_proj:.3f}")
print(f"Improvement:           {((prec_proj - prec_raw)/prec_raw)*100:.1f}%")

# 2. OOD Detection (Coverage KDE)
from sklearn.neighbors import KernelDensity

def get_kde_scores(train_embeds, test_embeds, bandwidth=0.5):
    # Normalize for Cosine Similarity proxy (L2 on unit vectors)
    train_norm = train_embeds / np.linalg.norm(train_embeds, axis=1, keepdims=True)
    test_norm = test_embeds / np.linalg.norm(test_embeds, axis=1, keepdims=True)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(train_norm)
    return kde.score_samples(test_norm)

print("\n--- OOD Detection Evaluation ---")
# Train KDE on ID data (Training Fail + Success)
kde_scores_id = get_kde_scores(index_set_proj, z_succ) # Evaluate on Test Success (ID)
kde_scores_ood = get_kde_scores(index_set_proj, z_ood) # Evaluate on OOD

# Labels: 1 for ID, 0 for OOD (Standard OOD task is usually detection of OOD, but AUC is symmetric)
y_true = np.concatenate([np.ones(len(kde_scores_id)), np.zeros(len(kde_scores_ood))])
y_scores = np.concatenate([kde_scores_id, kde_scores_ood])

auc = roc_auc_score(y_true, y_scores)
print(f"BEM Coverage AUC (Real Text): {auc:.4f}")

# Baseline: Max Cosine Similarity (standard OOD baseline)
def get_max_sim(train_embeds, test_embeds):
    train_norm = train_embeds / np.linalg.norm(train_embeds, axis=1, keepdims=True)
    test_norm = test_embeds / np.linalg.norm(test_embeds, axis=1, keepdims=True)
    sims = np.dot(test_norm, train_norm.T)
    return np.max(sims, axis=1)

scores_raw_id = get_max_sim(index_set_raw, raw_succ)
scores_raw_ood = get_max_sim(index_set_raw, raw_ood)
y_scores_raw = np.concatenate([scores_raw_id, scores_raw_ood])

auc_raw = roc_auc_score(y_true, y_scores_raw)
print(f"Raw MaxSim AUC (Baseline):    {auc_raw:.4f}")

# --- GENERATE PLOTS ---
plt.figure(figsize=(10, 4))

# Plot 1: Retrieval Improvement
plt.subplot(1, 2, 1)
plt.bar(['Raw (384D)', 'BEM (64D)'], [prec_raw, prec_proj], color=['gray', 'blue'])
plt.title('Failure Retrieval Precision (Real Text)')
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Plot 2: OOD Score Distributions
plt.subplot(1, 2, 2)
sns.kdeplot(kde_scores_id, label='In-Distribution (Science)',  fill=True, color='blue')
sns.kdeplot(kde_scores_ood, label='Out-of-Distribution (Sports)', fill=True, color='red')
plt.title(f'OOD Coverage Signal (AUC={auc:.2f})')
plt.xlabel('Log Likelihood')
plt.legend()

plt.tight_layout()
plt.savefig('real_embedding_results.pdf')
print("Saved real_embedding_results.pdf")

# Generate LaTeX table code
print("\n--- LaTeX Table Row ---")
print(f"Retrieval & {prec_raw:.2f} & {prec_proj:.2f} \\\\")
print(f"OOD AUC & {auc_raw:.2f} & {auc:.2f} \\\\")
