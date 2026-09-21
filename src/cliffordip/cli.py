"""Command line interface for CliffordIP."""

import argparse
import json
import sys
from typing import Any

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2


def _emit(payload: dict, as_json: bool, lines: list[str]) -> None:
    """Write a machine-readable object to stdout, or human text to stderr."""
    if as_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for line in lines:
            print(line, file=sys.stderr)


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit a JSON object on stdout")


def cmd_info(args) -> int:
    import torch

    from cliffordip import __version__

    payload: dict[str, Any] = {
        "version": __version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "extras": _extras_status(),
    }
    lines = [
        f"cliffordip {payload['version']}",
        f"torch {payload['torch']} (cuda: {payload['cuda_available']})",
        "extras: "
        + ", ".join(f"{k}={'ok' if v else 'missing'}" for k, v in payload["extras"].items()),
    ]
    _emit(payload, args.json, lines)
    return EXIT_OK


def _extras_status() -> dict[str, bool]:
    import importlib.util

    return {
        name: importlib.util.find_spec(module) is not None
        for name, module in (
            ("oc20", "lmdb"),
            ("oc22", "lmdb"),
            ("qm9", "ase"),
            ("md17", "ase"),
            ("log", "wandb"),
        )
    }


def cmd_datasets(args) -> int:
    import cliffordip.data  # noqa: F401 — registers the in-tree datasets
    from cliffordip.train.dataset_registry import list_datasets, unavailable_datasets

    available = list_datasets()
    unavailable = unavailable_datasets()
    payload = {"available": available, "unavailable": unavailable}
    lines = [f"  {name}" for name in available]
    lines += [f"  {name}  unavailable: {reason}" for name, reason in unavailable.items()]
    _emit(payload, args.json, ["datasets:", *lines])
    return EXIT_OK


def cmd_models(args) -> int:
    import cliffordip.train.register_cliffordip  # noqa: F401 — registers the built-in models
    from cliffordip.train.model_registry import list_models

    payload = {"available": list_models()}
    _emit(payload, args.json, ["models:", *(f"  {n}" for n in payload["available"])])
    return EXIT_OK


def cmd_check_equivariance(args) -> int:
    import torch
    from torch_geometric.data import Data

    from cliffordip.equivariance import check_equivariance
    from cliffordip.wrapper import CliffordIPWrapper

    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    model = (
        CliffordIPWrapper(
            n_atom_types=10,
            n_channels=args.n_channels,
            n_interactions=args.n_interactions,
            n_rbf=8,
            n_hidden_output=8,
            n_heads=2,
        )
        .double()
        .eval()
    )
    n = args.n_atoms
    data = Data(
        z=torch.randint(1, 10, (n,), generator=generator),
        pos=(torch.rand(n, 3, dtype=torch.float64, generator=generator) - 0.5) * 6.0,
        batch=torch.zeros(n, dtype=torch.long),
    )
    result = check_equivariance(
        model, data, n_trials=args.trials, include_improper=args.o3, atol=args.atol, seed=args.seed
    )
    lines = [f"worst deviation: {result['worst']:.3e} (tolerance {result['atol']:.1e})"]
    for name, dev in result["deviations"].items():
        lines.append(f"  {name:<12} energy {dev['energy']:.3e}  forces {dev['forces']:.3e}")
    lines.append("PASS" if result["passed"] else "FAIL")
    _emit(result, args.json, lines)
    return EXIT_OK if result["passed"] else EXIT_FAILURE


def cmd_train(args, extra: list[str]) -> int:
    from cliffordip.lightning.cli import main as lightning_main

    argv = ["cliffordip-train", args.subcommand or "fit"]
    if args.dataset:
        argv += ["--data.cfg.dataset.name", args.dataset]
    argv += extra
    sys.argv = argv
    lightning_main()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cliffordip", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="package, torch and extras status")
    _add_json(p_info)
    p_info.set_defaults(func=cmd_info)

    p_ds = sub.add_parser("datasets", help="list datasets")
    p_ds.add_argument("action", nargs="?", default="list", choices=["list"])
    _add_json(p_ds)
    p_ds.set_defaults(func=cmd_datasets)

    p_m = sub.add_parser("models", help="list models")
    p_m.add_argument("action", nargs="?", default="list", choices=["list"])
    _add_json(p_m)
    p_m.set_defaults(func=cmd_models)

    p_eq = sub.add_parser("check-equivariance", help="verify equivariance of a fresh model")
    p_eq.add_argument("--o3", action="store_true", help="include improper transforms")
    p_eq.add_argument("--trials", type=int, default=20)
    p_eq.add_argument("--atol", type=float, default=1e-6)
    p_eq.add_argument("--seed", type=int, default=0)
    p_eq.add_argument("--n-atoms", type=int, default=6)
    p_eq.add_argument("--n-channels", type=int, default=8)
    p_eq.add_argument("--n-interactions", type=int, default=5)
    _add_json(p_eq)
    p_eq.set_defaults(func=cmd_check_equivariance)

    p_tr = sub.add_parser("train", help="train via the Lightning CLI")
    p_tr.add_argument("subcommand", nargs="?", default="fit")
    p_tr.add_argument("--dataset", help="registered dataset name")
    p_tr.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    try:
        if args.func is cmd_train:
            code = cmd_train(args, extra)
        else:
            if extra:
                parser.error(f"unrecognized arguments: {' '.join(extra)}")
            code = args.func(args)
    except KeyError as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    return code


if __name__ == "__main__":
    raise SystemExit(main())
