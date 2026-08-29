# A Worked Identifiability Audit in a Homogeneous BEC

[![GitHub](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/davidahmann/coherism/tree/main/physics)

This directory contains the manuscript and reproducibility material for
*A Worked Identifiability Audit of a Stipulated Preparation-Indexed Potential in a Homogeneous Bose--Einstein Condensate*.

## Claim boundary

The paper studies a deliberately conditional model:

1. For a specified finite phonon band and full-support Gibbs reference,
   `D(rho || sigma_beta) = beta * Delta E - Delta S` exactly.
2. An ideal displaced Gibbs state and an ideal heated product Gibbs state can
   be constructed with the same added mean energy. Their relative entropies
   differ by the entropy increase of the heated state.
3. A preparation-indexed external potential proportional to that finite-mode relative
   entropy is then **stipulated**. Its profile, sign, intervention, and coupling
   `kappa` are free inputs.
4. At fixed total particle number, a homogeneous stationary GP/BdG kernel maps
   the imposed potential to a density template with its zero mode removed. The
   exact localized response is reproduced by independent stationary GP relaxation.
5. A seeded classical-field example motivates generic nuisance controls only.
   It is not a matched nuisance model, covariance estimate, or uncertainty floor.
6. A deterministic two-bin fixture demonstrates both an identifiable nuisance
   model and complete template absorption.
7. The present work is a conditional identifiability audit, not a new source law,
   analogue-gravity model, detection forecast, or gravitational test.

The model is not a derivation of gravitational backreaction and does not construct
an acoustic analogue. No effective metric, flow, horizon, or acoustic propagation
observable is used.

## Canonical manuscript files

- `../paper/main.tex` — canonical LaTeX source
- `coherism.tex` — compatibility symlink to `../paper/main.tex`
- `../paper/references.bib` — canonical bibliography
- `coherism_refs.bib` — compatibility symlink to the canonical bibliography
- `coherism.pdf` — compiled manuscript
- `cover_letter.tex` and `cover_letter.pdf` — journal cover letter

## Reproducibility material

### Finite-mode identity and conditional response

`bec_sonic_horizon_simulation.py` retains its historical filename so old links
continue to work. It now produces only:

- the finite-mode relative-entropy arithmetic;
- coupling-normalized stationary response templates;
- the prospective mapping from an externally calibrated residual budget to a
  bound on `|kappa|`; and
- a mode-count sensitivity table for the exact entropy contrast.

It does not assign a physical value to `kappa`, simulate a sonic horizon, or
define a detection threshold.

```bash
cd physics
.venv/bin/python bec_sonic_horizon_simulation.py
```

Tracked outputs:

- `bec_sonic_horizon_results.png`
- `bec_sonic_horizon_data.dat`
- `bec_sonic_horizon_robustness.dat`

For the dimensionless illustration `N=1000`, `M=100`, and
`beta*hbar*omega=2`, the script returns
`D_displaced=2000`, `D_heated=1709.264420`, and
`Delta S=290.735580` nats. These are finite-model arithmetic, not absolute
density predictions.

### Stationary response and generic control motivation

`gpe_protocol_simulation.py` contains three checks on a uniform 1D
testbed:

- imaginary-time relaxation compared with the analytic homogeneous static
  GP/BdG kernel; and
- like-for-like validation of the localized fixed-number profile used in the
  manuscript; and
- real-time evolution of deterministic-phase and phase-randomized classical
  fields with the same amplitude spectrum and a fixed random seed.

The third calculation is not a thermal density operator and is not proven to
match full nonlinear GP energy exactly. Its order-`10^-3` differential is a
generic reminder that controls are needed, not a calibrated nuisance floor.

```bash
cd physics
.venv/bin/python gpe_protocol_simulation.py
```

Tracked outputs:

- `gpe_protocol_results.png`
- `gpe_protocol_data.dat`

The current deterministic run gives a maximum nonzero-mode kernel deviation of
`0.53%`. For the localized profile, the maximum pointwise difference is `0.0226%`
of peak and the relative L2 difference is `0.0076%`. The illustrative RMS
preparation differentials are `2.98e-3` (`M=4`) and `3.79e-3` (`M=8`).

### Synthetic identifiability fixture

`identifiability_audit.py` checks the nuisance-projected estimator on two
declared two-bin cases. An orthogonal nuisance leaves unit projected template
norm and recovers `kappa_hat=0.25`; a nuisance spanning the template gives zero
projected norm and fails closed as non-identifiable.

```bash
cd physics
.venv/bin/python identifiability_audit.py
```

Tracked output: `identifiability_audit_data.dat`.

## Build

The locked Python dependencies are in `../paper/requirements-lock.txt`.

```bash
cd physics
SOURCE_DATE_EPOCH=1781222400 latexmk -g -pdf -interaction=nonstopmode -halt-on-error coherism.tex
SOURCE_DATE_EPOCH=1781222400 latexmk -g -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
```

## Archived exploratory material

`coherism_frw_simulation.py`, `coherism_frw_data.dat`,
`coherism_frw_results.png`, `generate_data.py`, `simulation_data.dat`, and
`toy_coherist_friction.png` are retained as project history. They are not part
of the current claim set or submission evidence.

## License and contact

This work is licensed under [CC BY 4.0](../LICENSE).

David Ahmann  
dahmann@lumyn.cc  
[ORCID 0009-0006-4066-8760](https://orcid.org/0009-0006-4066-8760)
