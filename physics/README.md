# Coherism: Coherence-Dependent Backreaction in Semiclassical and Analog Gravity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17868263.svg)](https://doi.org/10.5281/zenodo.17868263)

This directory contains the manuscript and supporting code for the Coherism paper, prepared for submission to *Classical and Quantum Gravity*.

## Abstract

We derive a testable prediction for analog gravity experiments in Bose-Einstein condensates: coherent phonon injection near a sonic horizon produces density modulations δρ/ρ₀ ~ 10⁻⁶, while thermal phonon injection of identical energy produces no such effect. This coherent-versus-thermal signature—measurable with current BEC technology—constitutes a qualitative test absent in competing frameworks (Penrose-Diósi, stochastic gravity). The prediction emerges from an *informational stress tensor* Θ_μν derived by varying a coherence functional built from relative entropy, generalized entropy, and a geometric action.

## Repository Contents

### Manuscript
- `coherism.tex` — Main LaTeX source (revtex4-2, PRD format)
- `coherism.pdf` — Compiled manuscript
- `coherism_refs.bib` — Bibliography

### Numerical Simulations

#### 1. BEC Sonic Horizon Simulation (Primary Test)

Implements the key experimental prediction from Appendix N (Analog Gravity Predictions).

- `bec_sonic_horizon_simulation.py` — Main simulation code
- `bec_sonic_horizon_results.png` — Visualization (Figure 6 in paper)
- `bec_sonic_horizon_data.dat` — Output data

**Key result:** Coherent phonon injection produces δρ/ρ₀ ≈ 1.4 × 10⁻⁶; thermal phonons produce zero signal.

**To reproduce:**

```bash
python3 bec_sonic_horizon_simulation.py
```

#### 2. FRW Cosmology Simulation

Implements the coupled FRW-coherence evolution from Appendix O.

- `coherism_frw_simulation.py` — Main simulation code
- `coherism_frw_data.dat` — Output data (log_a, a, N_sq, rho_coh_frac, w_coh)
- `coherism_frw_results.png` — Visualization of results

**To reproduce:**

```bash
python3 coherism_frw_simulation.py
```

#### 3. Toy Model (0D QFT)

Demonstrates "coherist friction" — energy dissipation from informational feedback.

- `generate_data.py` — Harmonic oscillator with coherence-dependent damping
- `simulation_data.dat` — Output comparing baseline vs. coherist evolution

**To reproduce:**

```bash
python3 generate_data.py
```

### Build Artifacts (can be regenerated)

- `coherism.aux`, `coherism.bbl`, `coherism.blg`, `coherism.log`, `coherism.out` — LaTeX build files
- `coherismNotes.bib` — Auto-generated notes bibliography

## Dependencies

### LaTeX

- `revtex4-2` (APS document class)
- Standard packages: `amsmath`, `physics`, `tikz`, `pgfplots`, `hyperref`

### Python (for simulations)

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Compiling the Manuscript

```bash
pdflatex coherism.tex
bibtex coherism
pdflatex coherism.tex
pdflatex coherism.tex
```

Or use `latexmk`:
```bash
latexmk -pdf coherism.tex
```

## Key Results

1. **Primary experimental prediction**: BEC density modulation δρ/ρ₀ ~ 10⁻⁶ for coherent phonons, zero for thermal (measurable with current technology)
2. **Falsification criterion**: Null result at δρ/ρ₀ < 10⁻⁷ would falsify the acoustic implementation
3. **Informational stress tensor** Θ_μν derived for Schwarzschild, FRW, Rindler, and acoustic geometries
4. **Two independent derivations** of the coupling constant κ (holographic and entropic)
5. **Lindblad generator** derived from first principles via Unruh-DeWitt detector dynamics
6. **Unique distinguishing test**: Coherent vs. thermal states produce different effects (η ~ 10⁻¹⁵), absent in Penrose-Diósi and stochastic gravity
7. **Reference state uniqueness**: σ[g] uniquely determined by Hadamard, KMS, maximum entropy, and local Lorentz invariance axioms

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann  
Independent Researcher, Toronto, Canada
