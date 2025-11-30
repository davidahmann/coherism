# Coherism: A Variational Feedback Framework for Quantum Information and Spacetime Geometry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17766365.svg)](https://doi.org/10.5281/zenodo.17766365)

This directory contains the manuscript and supporting code for the Coherism paper.

## Abstract

We propose a variational framework that couples quantum information and spacetime geometry through a coherence functional built from relative entropy, generalized entropy, and a geometric action. The framework yields an informational stress tensor that augments the semiclassical Einstein equations and a geometry-dependent Lindblad evolution for the quantum state.

## Repository Contents

### Manuscript
- `coherism.tex` — Main LaTeX source (revtex4-2, PRD format)
- `coherism.pdf` — Compiled manuscript
- `coherism_refs.bib` — Bibliography

### Numerical Simulations

#### 1. FRW Cosmology Simulation
Implements the coupled FRW-coherence evolution from Appendix: Numerical Simulation.

- `coherism_frw_simulation.py` — Main simulation code
- `coherism_frw_data.dat` — Output data (log_a, a, N_sq, rho_coh_frac, w_coh)
- `coherism_frw_results.png` — Visualization of results

**To reproduce:**
```bash
python3 coherism_frw_simulation.py
```

#### 2. Toy Model (0D QFT)
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
- SciPy
- Matplotlib

Install dependencies:
```bash
pip install numpy scipy matplotlib
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

1. **Informational stress tensor** derived for Schwarzschild, FRW, Rindler, and acoustic geometries
2. **Two independent derivations** of the coupling constant κ (holographic and entropic)
3. **Lindblad generator** derived from Unruh-DeWitt detector dynamics
4. **Analog gravity predictions**: density modulations δρ/ρ₀ ~ 10⁻⁶ within experimental reach

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann  
Independent Researcher, Toronto, Canada
