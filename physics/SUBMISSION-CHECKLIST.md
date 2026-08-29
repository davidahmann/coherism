# Methods-Paper Release Checklist

Prepared: 2026-08-28

Paper type: worked cautionary methods/tutorial note

Venue status: no journal submission is recommended until a suitable methods or
tutorial venue is selected. Formal GPD review rejected the prior framing for
*Foundations of Physics* because the standard ingredients and stipulated bridge
do not provide a new foundational result.

## Release set

- [x] `coherism.tex`: compatibility link to canonical `../paper/main.tex`.
- [x] `coherism_refs.bib`: compatibility link to canonical bibliography.
- [x] `coherism.pdf`: rebuilt and visually inspected final six-page methods note.
- [x] `cover_letter.pdf`: rebuilt and visually inspected one-page venue-neutral draft.
- [x] `coherism_submission.zip`: rebuilt as a portable source, evidence, and
  reproducibility package; clean-room manuscript and cover builds pass.

## Scientific and numerical validation

- [x] GPD synchronized to official release 1.2.2 and project state repaired.
- [x] Round 5 preserved and reviewed the inherited manuscript; verdict `reject`.
- [x] Round 6 reviewed the first focused rewrite; verdict `reject` for
  *Foundations of Physics*, with a methods-note recommendation.
- [x] Fixed total particle number selected and the zero Fourier mode removed.
- [x] Finite-mode fixtures reproduce `D_displaced = 2000.000000`,
  `D_heated = 1709.264420`, and `Delta S = 290.735580`.
- [x] Fixed-number response reproduces `R_peak = 0.902969201` and contrast
  normalization `262.525274`.
- [x] Residual budgets from `1e-6` through `1e-2` map only algebraically to
  `3.809e-9` through `3.809e-5`; no budget or coupling is forecast.
- [x] Nonzero-mode stationary GP response agrees with the analytic kernel to at
  most `0.53%` over the six reported wave numbers.
- [x] The exact localized fixed-number profile agrees with independent stationary
  GP relaxation to `0.0226%` of peak and `0.0076%` in relative L2 norm.
- [x] The seeded phase example remains labeled as generic existence-only control
  motivation, not a matched nuisance, covariance, or uncertainty budget.
- [x] Round 7 reviewed the pre-final methods note; verdict `minor_revision`,
  high confidence, with six specified repairs and no blocking issue.
- [x] All six Round 7 repairs are present: canonical audit taxonomy,
  thermodynamic and estimator-space definitions, weak-response condition,
  direct identifiability literature, deterministic GLS success/failure fixture,
  and explicit audience and learning objective.
- [x] Round 8 reviewed the exact final manuscript digest; verdict `accept`, high
  confidence, with zero major, minor, or blocking issues in the reproducible
  methods/technical or postgraduate tutorial class.

## Build and artifact validation

- [x] All three numerical scripts rerun in the pinned repository-local environment.
- [x] Python syntax compilation passes for all three scripts.
- [x] Manuscript and cover letter build without undefined citations, undefined
  references, LaTeX errors, fatal errors, or overfull boxes.
- [x] Render and visually inspect every PDF page for clipping, overlap, stale
  claims, and legibility.
- [x] Refresh and strictly validate bibliography, reproducibility, citation, and
  artifact manifests against the final files.
- [x] Confirm portable archive inventory, paths, and hashes; no LaTeX build
  debris, Python cache, machine-local path, or absolute archive member remains.
- [x] Run the final GPD health, state, review, and repository diff gates. State,
  project contract, strict review preflight, Round 8 artifacts, reproducibility,
  and diff checks pass; health has no failing check and only reports historical
  phase-summary/result bookkeeping as a warning.

## Future submission checks

- [ ] Select a venue appropriate for a worked methods/tutorial note.
- [ ] Confirm the portal title exactly matches the manuscript.
- [ ] Confirm author affiliation, ORCID, correspondence email, and declarations.
- [ ] Upload the tracked PDF from this release, not an older copy.
- [ ] Preserve the scope in all metadata: stipulated potential, fixed-number
  conditional response, unexecuted estimator, and no source-law or gravity claim.
