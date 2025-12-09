# ALFM-BEM: Bidirectional Experience Memory for Continuous Learning in Foundation Model Deployments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17868262.svg)](https://doi.org/10.5281/zenodo.17868262)

This directory contains the manuscript and supporting code for the ALFM-BEM paper.

## Abstract

Foundation models are deployed frozen, creating a fundamental gap: they cannot learn from deployment experiences. We introduce ALFM-BEM (Adaptive Latent Feedback Model with Bidirectional Experience Memory), a unified wrapper architecture that enables continuous learning without modifying backbone weights. The central abstraction is **Bidirectional Experience Memory (BEM)**, a single memory structure where experiences live on a continuous outcome spectrum from failure to success. BEM naturally provides: (1) risk signals from failure similarity, (2) success patterns for retrieval-augmented reasoning, and (3) out-of-distribution detection as an emergent property of coverage. We extend the Consensus Engine with a fourth action—**Query**—that transforms passive abstention into active learning.

## Key Components

1.  **Bidirectional Experience Memory (BEM)** — A unified memory architecture storing both failures and successes on a continuous outcome spectrum.
2.  **Consensus Engine with Query Action** — Arbitrates between semantic signals and heuristic rules, capable of actively requesting information when OOD.
3.  **Bounded Adapters** — Enables safe continuous improvement via experience replay with provable stability guarantees.

## Repository Contents

### Manuscript
-   `alfm_bem.tex` — Main LaTeX source (JMLR format)
-   `alfm_bem.pdf` — Compiled manuscript
-   `alfm_bem_refs.bib` — Bibliography
-   `cover_letter.tex` — Submission cover letter

### Experiments

#### 1. Phase 1: Synthetic Validation & Sensitivity
Validates BEM's core mechanisms: failure retrieval, OOD detection, and parameter sensitivity.

-   `experiments/phase1_runner.py` — Orchestrates multi-seed ablation, scale, and sensitivity experiments.
-   `experiments/ablation_study.py` — Core logic for comparing BEM vs. RAG/NEP baselines.
-   `experiments/generate_figures.py` — Generates `ood_roc.pdf` and `drift.pdf`.

**To reproduce:**
```bash
python3 experiments/phase1_runner.py
python3 experiments/generate_figures.py
```

#### 2. Phase 2: Healthcare Claims Case Study
Simulates a realistic healthcare claims processing pipeline with latent payer rules.

-   `experiments/healthcare_simulator.py` — Full simulation: claim generation, hidden rules, BEM agent, and learning loop.
-   `experiments/learning_curve.pdf` — Generated plot showing rejection rate reduction.

**To reproduce:**
```bash
python3 experiments/healthcare_simulator.py
```

## Dependencies

### LaTeX
-   `jmlr2e.sty` (JMLR style file)
-   Standard packages: `amsmath`, `algorithm`, `algpseudocode`, `tikz`, `booktabs`, `hyperref`

### Python
-   Python 3.8+
-   NumPy
-   Matplotlib
-   Seaborn
-   Pandas

Install dependencies:
```bash
pip install numpy matplotlib seaborn pandas
```

## Compiling the Manuscript

```bash
pdflatex alfm_bem
bibtex alfm_bem
pdflatex alfm_bem
pdflatex alfm_bem
```

## Key Results

1.  **Failure Retrieval** — BEM achieves F1 > 0.99 on synthetic failure modes.
2.  **OOD Detection** — Coverage signal achieves AUC $\approx$ 1.0 for clustered OOD patterns.
3.  **Healthcare Case Study** — ALFM-BEM reduces claim rejection rates by **88\%** (from 12.5\% to 1.5\%) by learning latent rules from binary feedback.
4.  **Query Action** — Active clarification improves success rates by 8.0\% in high-uncertainty scenarios.

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
│                     │     BEM     │──→ Risk/Success/   │
│                     │   Memory    │    Coverage        │
│                     └─────────────┘                    │
│                            ↓                            │
│                     ┌─────────────┐                    │
│                     │  Consensus  │──→ Action          │
│                     │   Engine    │   (Trust/Abstain/  │
│                     └─────────────┘    Escalate/Query) │
│                                                         │
│   [Experience Loop] ←── (Outcome Feedback) ─────────────┘
└─────────────────────────────────────────────────────────┘
```

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann
Independent Researcher, Toronto, Canada
