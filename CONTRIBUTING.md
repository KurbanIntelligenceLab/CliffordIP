# Contributing to CliffordIP

Thanks for your interest in CliffordIP. Changes land on `main` through pull requests reviewed by
the maintainer ([@CalciumNitrade](https://github.com/CalciumNitrade), Can Polat).

## Workflow

1. **Fork** the repository (external contributors) or **create a branch** (collaborators).

   ```bash
   git checkout -b feat/my-change
   ```

2. **Set up the development environment.**

   ```bash
   uv sync
   uv run pre-commit install
   ```

3. **Make your change**, keeping it focused — one logical change per pull request.

4. **Verify locally** before opening the PR — CI runs exactly these:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   uv run pytest
   ```

   Any change under `src/cliffordip/cliffordip.py`, `interaction.py` or `neighbors.py` must keep
   `uv run pytest tests/test_equivariance.py` green. O(3) equivariance, including reflections, is
   the core invariant of the model.

5. **Open a pull request** against `main` and fill out the template. The maintainer is requested as
   a reviewer automatically.

## Coding standards

- Type new code. `cliffordip.config`, `cliffordip.neighbors`, `cliffordip.train.registry` and
  `cliffordip.train.checkpoints` are checked with `disallow_untyped_defs`; keep them that way.
- Keep the forward pass allocation-free where the existing code is — pre-allocated multivector
  tensors filled via index slices are required for `torch.compile` stability.
- Update `README.md` and `CHANGELOG.md` whenever user-facing behavior changes.
- Comments explain what the code does now; keep docstrings to one line unless the name needs more.
- New behaviour comes with a test. Anything needing a downloaded dataset is marked `slow`.

## Reporting bugs and requesting features

Open an issue with:

- what you ran (exact command or code snippet),
- what you expected, and what happened instead,
- versions: `python -c "import cliffordip, torch; print(cliffordip.__version__, torch.__version__)"`.

## Citation

If you use CliffordIP in academic work, please cite the paper — see the Citation section of
[`README.md`](README.md).
