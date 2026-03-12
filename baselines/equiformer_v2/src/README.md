# Vendored EquiformerV2 Implementation

This directory contains source code from [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) (MIT License).

## Source

- **Repository**: https://github.com/atomicarchitects/equiformer_v2
- **Paper**: EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations (ICLR 2024)
- **License**: MIT
- **Vendored on**: 2026-02-11

## Modifications

- Removed OCP dependencies from `equiformer_v2_oc20.py`:
  - Removed imports: `registry`, `conditional_grad`, `BaseModel`, `CalcSpherePoints`, `GaussianSmearing` (and variants)
  - Changed base class from `BaseModel` to `nn.Module`
  - Removed `@registry.register_model` decorator
  - Removed `@conditional_grad` decorator
  - Switched from `GaussianSmearing` to local `GaussianRadialBasisLayer`
- Added `num_output` property to `GaussianRadialBasisLayer` for compatibility
- Adapted for PyG Data format via thin wrapper in parent directory

## Original License

MIT License

Copyright (c) 2023 Yi-Lun Liao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

```bibtex
@inproceedings{liao2024equiformerv2,
  title={EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations},
  author={Liao, Yi-Lun and Smidt, Tess},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
