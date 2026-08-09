# Contributing to CliffordIP

Thanks for your interest in CliffordIP. This document describes how changes get into this
repository.

## Ground rules

- **`main` is protected.** Nobody pushes to `main` directly — all changes land through a pull
  request.
- **Every pull request needs an approving review from the maintainer**
  ([@CalciumNitrade](https://github.com/CalciumNitrade), Can Polat) before it can be merged. This
  is enforced by branch protection and by [`.github/CODEOWNERS`](.github/CODEOWNERS), which assigns
  ownership of the whole tree to the maintainer.
- Force-pushes and branch deletion on `main` are blocked.
- New commits pushed to a PR dismiss existing approvals — the branch has to be re-approved.

## Workflow

1. **Fork** the repository (external contributors) or **create a branch** (collaborators).

   ```bash
   git checkout -b feat/my-change
   ```

2. **Set up the development environment.**

   ```bash
   uv pip install -e ".[dev]"
   ```

3. **Make your change**, keeping it focused — one logical change per pull request.

4. **Verify locally** before opening the PR:

   ```bash
   # O(3)/Pin(3) equivariance must hold — this is the core invariant of the model
   python -c "from cliffordip import test_equivariance; test_equivariance()"

   pytest
   ruff check .
   ```

5. **Open a pull request** against `main` and fill out the template. The maintainer is requested as
   a reviewer automatically via `CODEOWNERS`.

6. **Address review feedback** by pushing additional commits to the same branch.

7. The **maintainer merges** once the review is approved and any required checks are green.

## Coding standards

- Strict typing: use `Literal` for enumerated values; avoid `None` and `Any` unless genuinely
  necessary, and call it out in the PR description when you do.
- Keep the forward pass allocation-free where the existing code is — pre-allocated multivector
  tensors filled via index slices are required for `torch.compile` stability.
- Update `README.md` whenever user-facing behavior changes.

## Reporting bugs and requesting features

Open an issue with:

- what you ran (exact command or code snippet),
- what you expected, and what happened instead,
- versions: `python -c "import cliffordip, torch; print(cliffordip.__version__, torch.__version__)"`.

## Citation

If you use CliffordIP in academic work, please cite the paper — see the Citation section of
[`README.md`](README.md).
