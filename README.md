# Coherism & ALFM: The Feedback Loop Project

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the source code and manuscripts for two parallel research initiatives exploring the role of **feedback loops** in fundamental physics and artificial intelligence.

While operating at opposite ends of the abstraction spectrum—one at the theoretical frontier of quantum gravity, the other at the practical frontier of enterprise AI—both projects share a core intellectual DNA: the emergence of structure through error correction.

## 📂 Repository Structure

### 1. `physics/` - Coherism
**Title:** *Coherism: A Variational Feedback Framework for Quantum Information and Spacetime Geometry*

This directory contains the LaTeX source for the "Coherism" preprint.

*   **The Big Idea:** Gravity is not a fundamental force, but a feedback mechanism where spacetime curves to compensate for entropic "noise" in quantum fields.
*   **Key Concept:** The **Coherence Functional**, a mathematical tool that derives Einstein's equations by minimizing the relative entropy between a quantum state and its geometric reference.
*   **Files:**
    *   `coherism.tex`: Main manuscript
    *   `coherism_refs.bib`: Bibliography
    *   `generate_data.py`: 0D QFT harmonic oscillator simulation
    *   `coherism_frw_simulation.py`: FRW cosmology simulation demonstrating "Coherist friction"

#### 🧬 The Feedback Loop (Coherism)
```mermaid
graph TD
    G[Spacetime Geometry g] -->|Induces| S[Reference State σ]
    S -->|Compared with| R[Quantum State ρ]
    R -->|Relative Entropy| C[Coherence Functional]
    C -->|Variation δg| T[Informational Stress]
    C -->|Variation δρ| L[Open System Evolution]
    T -->|Backreaction| G
    L -->|State Update| R
```

### 2. `alfm/` - ALFM (AI Systems)
**Title:** *ALFM: Adaptive Latent Feedback Model for Institutional Memory in Foundation Model Deployments*

This directory contains the LaTeX source and validation code for the ALFM framework.

*   **The Big Idea:** A wrapper architecture that enables frozen foundation models (like GPT-4) to "learn" from mistakes instantly without retraining.
*   **Key Concepts:**
    *   **Negative Evidence Prior (NEP):** Vector memory of failure modes for calibrated self-doubt
    *   **Consensus Engine:** Multi-agent arbitration between semantic intuition and heuristic rules
    *   **Three-Tier Adapters:** Safe continual learning with tenant isolation
*   **Files:**
    *   `alfm.tex`: Main manuscript (includes algorithm pseudocode, API examples, failure taxonomy)
    *   `alfm_refs.bib`: Bibliography
    *   `simulate_nep.py`: NEP validation simulation (precision-recall analysis)
    *   `simulate_drift.py`: Adapter stability simulation

#### 🧠 ALFM Architecture
```mermaid
graph LR
    User[User Input] -->|Context| BB[Frozen Backbone]
    User -->|Context| NEP[NEP Memory]
    NEP -->|Risk Signal| CE[Consensus Engine]
    BB -->|Latent State| CE
    CE -->|Decision| Action{Action}
    Action -->|Low Risk| Out[Output]
    Action -->|High Risk| Abstain[Abstain/Escalate]
```

---

## 🚀 Compilation

Both papers are written in LaTeX and use `revtex4-2`.

**To compile the Physics paper:**
```bash
cd physics
pdflatex coherism.tex
bibtex coherism
pdflatex coherism.tex
pdflatex coherism.tex
```

**To compile the AI paper:**
```bash
cd alfm
pdflatex alfm.tex
bibtex alfm
pdflatex alfm.tex
pdflatex alfm.tex
```

## 🔗 The Connection

*   **Coherism (Physics):** Gravity is spacetime correcting for *entropic errors*.
*   **ALFM (AI):** Intelligence is an AI correcting for *prediction errors*.

Both propose a "Universal Theory of Feedback"—one applied to the fabric of the universe, the other to the fabric of artificial intelligence.

## 📄 License

This work is licensed under [CC-BY 4.0](LICENSE). You are free to share and adapt with attribution.

## 📖 Citation

See [CITATION.cff](CITATION.cff) for citation information.

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---
*Author: David Ahmann*  
*Toronto, Canada*
