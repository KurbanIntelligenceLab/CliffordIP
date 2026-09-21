"""CLI contract: exit codes and machine-readable output."""

import json
import os
import subprocess
import sys

import pytest

from cliffordip.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, main


def run(capsys, *args):
    """Invoke the CLI and return (exit code, parsed stdout, stderr)."""
    code = main(list(args))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else None
    return code, payload, captured.err


def test_info_reports_version_and_torch(capsys):
    code, payload, _ = run(capsys, "info", "--json")
    assert code == EXIT_OK
    assert set(payload) == {"version", "torch", "cuda_available", "cuda_device_count", "extras"}


def test_datasets_list_separates_available_from_unavailable(capsys):
    code, payload, _ = run(capsys, "datasets", "list", "--json")
    assert code == EXIT_OK
    assert "oc20_s2ef" in payload["available"]
    assert isinstance(payload["unavailable"], dict)


def test_models_list_includes_cliffordip(capsys):
    code, payload, _ = run(capsys, "models", "list", "--json")
    assert code == EXIT_OK
    assert "cliffordip" in payload["available"]


def test_check_equivariance_passes_and_reports_deviations(capsys):
    code, payload, _ = run(
        capsys, "check-equivariance", "--o3", "--trials", "3", "--n-interactions", "3", "--json"
    )
    assert code == EXIT_OK
    assert payload["passed"]
    assert {"rotation", "reflection", "inversion", "translation"} <= set(payload["deviations"])
    assert payload["worst"] <= payload["atol"]


def test_check_equivariance_fails_when_the_tolerance_is_impossible(capsys):
    code, payload, _ = run(
        capsys,
        "check-equivariance",
        "--trials",
        "2",
        "--n-interactions",
        "2",
        "--atol",
        "0",
        "--json",
    )
    assert code == EXIT_FAILURE
    assert not payload["passed"]


def test_human_output_goes_to_stderr_leaving_stdout_clean(capsys):
    code, payload, err = run(capsys, "info")
    assert code == EXIT_OK
    assert payload is None
    assert "cliffordip" in err


@pytest.mark.parametrize("args", [("nonsense",), ("info", "--not-a-flag")])
def test_bad_usage_exits_with_the_usage_code(args):
    with pytest.raises(SystemExit) as excinfo:
        main(list(args))
    assert excinfo.value.code == EXIT_USAGE


@pytest.mark.parametrize("command", ["info", "datasets", "models", "check-equivariance", "train"])
def test_help_is_available_for_every_command(command):
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == EXIT_OK


def test_installed_entry_point_runs():
    """The packaged console script reaches the same code path."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.run(
        [sys.executable, "-m", "cliffordip.cli", "info", "--json"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == EXIT_OK
    assert json.loads(proc.stdout)["version"]
