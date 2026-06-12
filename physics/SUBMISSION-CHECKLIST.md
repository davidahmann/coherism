# Foundations of Physics Submission Checklist

Date prepared: `2026-03-22` (revised `2026-06-10`: free-energy source, differential observable, GPE study)
Target journal: `Foundations of Physics`

## Submission Files

- [ ] Upload [coherism.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/coherism.pdf) as the main manuscript.
- [ ] Upload or paste [cover_letter.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/cover_letter.pdf) / [cover_letter.tex](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/cover_letter.tex) as the cover letter, depending on the journal portal.
- [ ] Keep [coherism.tex](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/coherism.tex) and [coherism_refs.bib](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/coherism_refs.bib) as the canonical editable sources.

## Validation Snapshot

- [x] Analog script rerun from [bec_sonic_horizon_simulation.py](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/bec_sonic_horizon_simulation.py).
- [x] Baseline outputs reproduced: `A_coh = 1.57e-6`, `A_th = 1.34e-6`, `ΔA = 2.29e-7`, GP/BdG shift of `ΔA` `0.63%` (max scan shift `0.84%`), scan range `[6.0e-8, 1.27e-6]`.
- [x] GPE study rerun from [gpe_protocol_simulation.py](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/gpe_protocol_simulation.py): kernel deviation `0.53%` max; confound rms `3.0e-3`/`3.8e-3` (M=4/8) at first-order `δn/n ≈ 0.1`.
- [x] [coherism.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/coherism.pdf) rebuilt successfully (18 pages).
- [x] [cover_letter.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/cover_letter.pdf) rebuilt successfully.
- [x] `physics/coherism.log` and `physics/cover_letter.log` contain no undefined citations or references.
- [x] [ARTIFACT-MANIFEST.json](/Users/davidahmann/Documents/Coherism/coherism_paper/paper/ARTIFACT-MANIFEST.json) hashes match current files.
- [x] [reproducibility-manifest.json](/Users/davidahmann/Documents/Coherism/coherism_paper/paper/reproducibility-manifest.json) hashes match current files.
- [x] GPD updated to 1.2.2 (from 1.1.0) and a full six-pass staged peer review (Round 4) run on the revised manuscript. **Recommendation: `major_revision`** (medium confidence) — an advance from the round-1/round-2 `reject`. See [REFEREE-REPORT-R4.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/.gpd/REFEREE-REPORT-R4.pdf) and [REVALIDATION-R4.md](/Users/davidahmann/Documents/Coherism/coherism_paper/.gpd/review/REVALIDATION-R4.md).
  - All five stage artifacts + claim index + review ledger schema-valid (GPD 1.2.2 CLI, exit 0); the recommendation-floor check confirms `major_revision` is the most favorable supported outcome.
  - **Do not submit as-is.** Four blocking issues remain, all fixable without new physics: (1) the `ΔA < 5e-8` falsification threshold lacks a detectability budget and one discrimination handle is physically void (ΔA is second-order in injection amplitude, like the confound); (2) the FoP significance case rests on that overstated empirical hook; (3) the CLM-017 uniqueness proposition is only partially aligned with its proof (proof red-team `gaps_found`); plus a factual Asenbaum-2020 citation error. Recommended fix: reframe around the conceptual core, demote the analog protocol to a model-level discriminator + proposed design, add the detectability discussion, apply the four proof edits, fix the citation.
  - The strict whole-pipeline gate reports `valid: false` by design (open proof gap `gaps_found` + unregistered GPD manuscript entrypoint); this is the correct fail-closed state for a `major_revision` verdict and was not overridden.
- [ ] GPD health check rerun on the revised manuscript (optional; not blocking the review verdict).

## Manual Final Check Before Submit

- [ ] Confirm title, abstract, and cover letter all use the same claim scope.
- [ ] Confirm the portal metadata matches the manuscript author affiliation and email.
- [ ] Confirm every figure referenced in the text appears correctly in the PDF.
- [ ] Confirm the declaration statements in the manuscript match the journal form fields.
- [ ] Confirm no broader claim than “EFT proposal with an implementation-specific analog falsification path” appears in the submission metadata.
- [ ] Confirm the uploaded PDF is the one in [coherism.pdf](/Users/davidahmann/Documents/Coherism/coherism_paper/physics/coherism.pdf), not a local stale copy.

## Repo Release Set

- [x] Manuscript source and compiled PDF
- [x] Cover letter source and compiled PDF
- [x] Analog script, generated figure, and generated data
- [x] Bibliography and bibliography audit
- [x] Artifact manifest and reproducibility manifest
