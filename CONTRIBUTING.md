# Contributing to genomic-research

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/benrio0923/genomic-research.git
cd genomic-research
pip install -e .
pip install pytest
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

- `genomic_research/templates/prepare.py` — Data loading, tokenization, evaluation (fixed)
- `genomic_research/templates/train.py` — Default model + training loop (template)
- `genomic_research/templates/program.md` — AI agent instructions
- `genomic_research/cli.py` — CLI entry point
- `tests/test_smoke.py` — Smoke tests

## Guidelines

- Keep `prepare.py` evaluation functions deterministic and fair
- New architectures should work within the existing `train.py` template pattern
- All tokenizers must implement `encode()`, `decode()`, `save()`, `load()`
- Test with synthetic data — no real genomic data in the repo

## Reporting Issues

Please include:
- Python version
- PyTorch version
- GPU info (if relevant)
- Full error traceback
