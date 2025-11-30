# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-30

### Added

#### ALFM Paper
- Algorithm pseudocode blocks for NEP Query, Consensus Engine Decision, and Bounded Adapter Update
- Python API example showing ALFM wrapper usage
- Explicit definitions for voting loss and calibration loss
- Ablation study design with four component isolation variants
- RAG baseline implementation details for fair comparison
- Latency breakdown table by hardware tier and NEP size
- Consensus Engine decision flow diagram (Figure 2)
- Failure Taxonomy appendix with concrete examples for all five failure types
- Severity assignment guidelines

#### Coherism Paper
- FRW cosmology simulation (`coherism_frw_simulation.py`)
- Expanded theoretical framework

#### Repository
- CC-BY 4.0 License
- CITATION.cff for academic referencing
- CONTRIBUTING.md guidelines
- CODE_OF_CONDUCT.md
- Comprehensive .gitignore

### Changed
- Updated README.md with accurate file listings
- Improved float handling in LaTeX for revtex4-2 compatibility

### Fixed
- Figure 2 (Consensus Engine) formatting to fit single column
- Action box spacing in diagrams

## [0.1.0] - 2024-11-27

### Added
- Initial manuscript drafts for both papers
- Basic simulation scripts
- Repository structure
