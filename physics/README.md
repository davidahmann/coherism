# Coherism: Controlled Limits and an Analog-Gravity Test

[![GitHub](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/davidahmann/coherism/tree/main/physics)

This directory contains the manuscript and supporting materials for the current Coherism paper.

## Abstract

The manuscript studies an effective-field-theory extension of semiclassical gravity in which a coarse-grained coherence functional generates an informational stress tensor. The paper focuses on controlled limits and one concrete analog-gravity falsification path. The informational source is fixed by the exact Gaussian-state identity `S(rho||sigma_beta) = beta_H ΔE − ΔS_vN` (relative entropy = free-energy excess over the Hawking-temperature reference), so both coherent and energy-matched thermal injections near a BEC sonic horizon source density modulations at the `delta-rho/rho_0 ~ 10^-6` level. The primary observable is their differential, `ΔA = A_coh − A_th = kappa_eff · ΔS_vN ≈ 2.3 x 10^-7` at the baseline, which directly measures the injected von Neumann entropy and is independent of the Hawking temperature at fixed band occupation. The density-response relation is benchmarked to the hydrodynamic compressibility limit, to a static Gross-Pitaevskii/Bogoliubov linear-response kernel (`0.63%` shift of the differential), and to time-dependent 1D GPE simulations that validate the kernel to `0.5%` and quantify the dominant ordinary-nonlinearity systematic. One-at-a-time `±25%` scans keep `ΔA` above `5 x 10^-8`. A differential below `5 x 10^-8` would falsify the acoustic implementation analyzed in the manuscript, not the broader EFT by itself.

## Repository Contents

### Manuscript
- `coherism.tex` - Main LaTeX source
- `coherism.pdf` - Compiled manuscript
- `coherism_refs.bib` - Bibliography
- `cover_letter.tex` - Submission cover letter for *Foundations of Physics*

### Main Reproducibility Material

#### BEC Sonic Horizon Scaling Script

Implements the illustrative scaling model used in the analog-gravity appendix.

- `bec_sonic_horizon_simulation.py` - Main simulation code
- `bec_sonic_horizon_results.png` - Visualization used in the manuscript
- `bec_sonic_horizon_data.dat` - Output data
- `bec_sonic_horizon_robustness.dat` - One-at-a-time robustness scan data

This script is an illustrative implementation of the acoustic ansatz used in the paper. It is not a full time-dependent Gross-Pitaevskii/Bogoliubov simulation, but it does benchmark the density-response step both to the standard compressibility relation and to a direct static GP/BdG linear-response kernel, and it includes a one-at-a-time robustness scan of the differential around a GP-compatible baseline.

Representative output: `A_coh ~= 1.57 x 10^-6`, `A_th ~= 1.34 x 10^-6`, differential `ΔA ~= 2.3 x 10^-7` (GP/BdG shift `0.63%`); `ΔA` stays above `5 x 10^-8` across all `±25%` scans.

To reproduce:

```bash
python3 bec_sonic_horizon_simulation.py
```

#### Time-Dependent 1D GPE Study

- `gpe_protocol_simulation.py` - Split-step 1D GPE solver
- `gpe_protocol_results.png` - Kernel validation + confound figure used in the manuscript
- `gpe_protocol_data.dat` - Output data

Two studies: (1) imaginary-time relaxation validates the static GP/BdG response kernel to better than `0.6%` over `k·xi ∈ [0.06, 1.5]`; (2) coherent vs energy-matched phase-randomized injection evolved under the bare GPE (no informational term) quantifies the ordinary state-dependent nonlinear response — the protocol's dominant systematic.

To reproduce:

```bash
python3 gpe_protocol_simulation.py
```

### Archived Exploratory Scripts

The repository also contains older exploratory scripts such as `coherism_frw_simulation.py` and `generate_data.py`. These are not part of the current submitted manuscript and should be read as archived exploratory materials rather than as validated results supporting the present paper.

### Build Artifacts

- `coherism.aux`, `coherism.bbl`, `coherism.blg`, `coherism.log`, `coherism.out` - LaTeX build files
- `coherismNotes.bib` - Auto-generated notes bibliography

## Dependencies

### LaTeX

- `revtex4-2`
- Standard packages: `amsmath`, `physics`, `tikz`, `pgfplots`, `hyperref`

### Python

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Compiling the Manuscript

```bash
pdflatex -interaction=nonstopmode -output-directory=. coherism.tex
bibtex coherism
pdflatex -interaction=nonstopmode -output-directory=. coherism.tex
pdflatex -interaction=nonstopmode -output-directory=. coherism.tex
```

Or use `latexmk`:

```bash
latexmk -pdf coherism.tex
```

## Key Results

1. The paper defines a coarse-grained coherence functional whose metric variation produces an informational stress tensor.
2. Controlled leading-order expressions are obtained in a 1+1D conformal toy model and in weak-field/Rindler limits.
3. The primary falsification path is an analog-gravity protocol in a BEC sonic horizon.
4. The coherent-vs-thermal structure of the prediction is derived, not assumed: the exact identity `S(rho||sigma) = beta_H ΔE − ΔS_vN` makes both injections source the stress at `~10^-6`, differing exactly by the injected von Neumann entropy.
5. The primary observable is the differential `ΔA = kappa_eff · ΔS_vN ≈ 2.3 x 10^-7`, independent of the Hawking temperature at fixed band occupation, with a `ΔS_vN` bandwidth scaling that serves as an internal consistency knob.
6. One-at-a-time `±25%` scans keep `ΔA` above the `5 x 10^-8` falsification threshold; the static GP/BdG correction stays below `0.9%`, and time-dependent GPE relaxation validates the kernel to `0.5%`.
7. Time-dependent GPE simulations quantify the dominant systematic: ordinary state-dependent nonlinear response at matched energy, to be discriminated by the static horizon-pinned profile and the bandwidth scaling.
8. A differential below `5 x 10^-8` would falsify the specific acoustic implementation analyzed in the manuscript, not the full EFT by itself.
9. The detector-model appendix motivates, but does not uniquely derive, the geometry-dependent GKLS structure used phenomenologically in the main text; consistency locking of the sourcing and dissipation sectors (à la hybrid classical-quantum dynamics) is identified as the key open theoretical task.

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann  
dahmann@lumyn.cc  
ORCID: [0009-0006-4066-8760](https://orcid.org/0009-0006-4066-8760)  
Independent Researcher, Toronto, Canada
