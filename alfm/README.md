# ALFM: Adaptive Latent Feedback Model for Institutional Memory in Foundation Model Deployments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17768608.svg)](https://doi.org/10.5281/zenodo.17768608)

This directory contains the manuscript and supporting code for the ALFM paper.

## Abstract

Foundation models are pretrained once and deployed frozen, creating three fundamental gaps: no memory of past failures, no calibrated self-doubt, and no safe mechanism for continual learning. ALFM (Adaptive Latent Feedback Model) is a modular wrapper architecture that addresses these gaps without modifying backbone model weights.

## Key Components

1. **Negative Evidence Prior (NEP)** — A tenant-isolated vector memory that stores failure patterns and provides risk signals at inference time
2. **Consensus Engine** — A multi-agent system ("Society of Agents") that arbitrates between semantic intuition and heuristic rules
3. **Three-Tier Adapters** — Global, domain, and local adapters enabling safe continual learning with cryptographic isolation

## Repository Contents

### Manuscript
- `alfm.tex` — Main LaTeX source (revtex4-2, PRA format)
- `alfm.pdf` — Compiled manuscript
- `alfm_refs.bib` — Bibliography

### Numerical Simulations

#### 1. NEP Performance Simulation
Validates that contrastive projection enables effective failure retrieval where raw embeddings fail.

- `simulate_nep.py` — Synthetic failure mode detection experiment
- `nep_data.txt` — Precision-recall data for projected vs. baseline embeddings
- `nep_simulation.png` — Visualization of NEP performance

**To reproduce:**
```bash
python3 simulate_nep.py
```

#### 2. Adapter Stability Simulation
Demonstrates bounded adapter drift under gradient clipping and norm constraints.

- `simulate_drift.py` — Compares unbounded SGD vs. ALFM bounded updates
- `drift_data.txt` — Drift norm trajectories over training steps

**To reproduce:**
```bash
python3 simulate_drift.py
```

### Build Artifacts (can be regenerated)
- `alfm.aux`, `alfm.bbl`, `alfm.blg`, `alfm.log`, `alfm.out` — LaTeX build files
- `alfmNotes.bib` — Auto-generated notes bibliography
- `venv/` — Python virtual environment (not tracked)

## Dependencies

### LaTeX
- `revtex4-2` (APS document class)
- Standard packages: `amsmath`, `algorithm`, `algpseudocode`, `tikz`, `pgfplots`, `listings`, `booktabs`

### Python (for simulations)
- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

## Compiling the Manuscript

```bash
pdflatex alfm.tex
bibtex alfm
pdflatex alfm.tex
pdflatex alfm.tex
```

Or use `latexmk`:
```bash
latexmk -pdf alfm.tex
```

## Key Results

1. **NEP Validation** — Contrastive projection achieves higher precision at equivalent recall compared to raw embeddings
2. **Adapter Stability** — Bounded updates provably prevent catastrophic drift (Proposition 4.2)
3. **Inference Overhead** — O(log |N|) query complexity with approximate nearest neighbor search
4. **Tenant Isolation** — Cryptographic guarantees prevent cross-tenant data leakage

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Input Context                                          │
│       ↓                                                 │
│  ┌─────────────┐    ┌─────────────┐                    │
│  │  Backbone   │───→│  Projection │                    │
│  │  (Frozen)   │    │    Layer φ  │                    │
│  └─────────────┘    └──────┬──────┘                    │
│                            │ z_ctx                      │
│                            ↓                            │
│                     ┌─────────────┐                    │
│                     │ NEP Memory  │──→ Risk Signal     │
│                     └─────────────┘                    │
│                            ↓                            │
│                     ┌─────────────┐                    │
│                     │  Consensus  │──→ Action          │
│                     │   Engine    │   (Trust/Abstain/  │
│                     └─────────────┘    Escalate/Clarify)│
└─────────────────────────────────────────────────────────┘
```

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann  
Independent Researcher, Toronto, Canada
