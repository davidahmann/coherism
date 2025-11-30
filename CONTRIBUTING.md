# Contributing to Coherism & ALFM

Thank you for your interest in contributing to this research project! This repository contains two parallel research papers exploring feedback mechanisms in physics and AI.

## Ways to Contribute

### 1. Report Issues
- **Typos or errors** in the manuscripts
- **Mathematical errors** or inconsistencies
- **Missing citations** or attribution issues
- **Build problems** with LaTeX compilation

### 2. Suggest Improvements
- Clarity improvements in exposition
- Additional validation experiments
- Extended theoretical analysis
- Improved visualizations or figures

### 3. Code Contributions
- Bug fixes in simulation scripts
- Performance improvements
- Additional validation simulations
- Documentation improvements

## How to Contribute

### For Minor Fixes (typos, small errors)
1. Open an issue describing the problem
2. Or submit a pull request directly with the fix

### For Larger Changes
1. **Open an issue first** to discuss the proposed change
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/your-feature`)
4. Make your changes
5. Test LaTeX compilation:
   ```bash
   cd physics && pdflatex coherism.tex && bibtex coherism && pdflatex coherism.tex
   cd ../alfm && pdflatex alfm.tex && bibtex alfm && pdflatex alfm.tex
   ```
6. Commit with clear messages
7. Submit a pull request

## Guidelines

### LaTeX Style
- Use `revtex4-2` document class conventions
- Keep lines under 100 characters where possible
- Use semantic line breaks (one sentence per line)
- Comment complex TikZ diagrams

### Code Style (Python)
- Follow PEP 8
- Include docstrings for functions
- Add comments for non-obvious logic

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Reference issues when applicable (`Fixes #123`)

## What We're NOT Looking For
- Changes to core theoretical claims without discussion
- Reformatting without functional improvement
- Dependencies on non-standard LaTeX packages

## Questions?

Open an issue with the `question` label.

## License

By contributing, you agree that your contributions will be licensed under CC-BY 4.0.
