# Coherism: Controlled Limits and an Analog-Gravity Test

[![GitHub](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/davidahmann/coherism/tree/main/physics)

This directory contains the manuscript and supporting materials for the current Coherism paper.

## Abstract

The manuscript studies an effective-field-theory extension of semiclassical gravity in which a coarse-grained coherence functional generates an informational stress tensor. The paper focuses on controlled limits and one concrete analog-gravity falsification path: in the acoustic implementation studied here, coherent phonon injection near a BEC sonic horizon yields a predicted density modulation `delta-rho/rho_0 ~ 10^-6`, while a matched thermal control gives no leading-order informational contribution. The density-response relation is benchmarked to the standard hydrodynamic compressibility limit of Gross-Pitaevskii/Bogoliubov theory, while the informational source term remains phenomenological. A null result below `10^-7` would falsify the acoustic implementation analyzed in the manuscript, not the broader EFT by itself.

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

This script is an illustrative implementation of the acoustic ansatz used in the paper. It is not a full Gross-Pitaevskii/Bogoliubov simulation, but it does benchmark the density-response step to the standard compressibility relation and includes a one-at-a-time robustness scan around a GP-compatible baseline.

Representative output: coherent phonon injection produces `A = max_{|r-r_H|<=L_coh} |delta-rho/rho_0| ~= 1.4 x 10^-6`; the matched thermal control has no leading-order informational signal.

To reproduce:

```bash
python3 bec_sonic_horizon_simulation.py
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
4. In the acoustic implementation studied here, coherent phonon injection yields `delta-rho/rho_0 ~ 10^-6` and a matched thermal control gives no leading-order informational contribution.
5. One-at-a-time `±25%` scans around the GP-compatible baseline keep the coherent signal above `10^-7`.
6. A null result below `10^-7` would falsify the specific acoustic implementation analyzed in the manuscript, not the full EFT by itself.
7. The detector-model appendix motivates, but does not uniquely derive, the geometry-dependent GKLS structure used phenomenologically in the main text.

## License

This work is licensed under [CC-BY 4.0](../LICENSE).

## Contact

David Ahmann  
dahmann@lumyn.cc  
ORCID: [0009-0006-4066-8760](https://orcid.org/0009-0006-4066-8760)  
Independent Researcher, Toronto, Canada
