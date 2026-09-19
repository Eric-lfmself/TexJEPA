"""The installed command runs without test sources or pytest by default."""
from pathlib import Path

import pytest

from scripts import run_all


def stub_pipeline(config, output):
    return {"metadata": {"elapsed_seconds": 0.0}, "robustness": [], "lesion": []}


@pytest.mark.parametrize("extra", [[], ["--skip-tests"]])
def test_default_cli_does_not_invoke_pytest(tmp_path, monkeypatch, capsys, extra):
    monkeypatch.setattr(run_all, "run_pipeline", stub_pipeline)
    monkeypatch.setattr(run_all.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected test execution"))
    run_all.main(["--dry-run", "--output", str(tmp_path / "smoke"), *extra])
    assert '"evidence": "synthetic_smoke"' in capsys.readouterr().out


def test_tests_are_explicit_and_scoped_to_source_checkout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_all, "run_pipeline", stub_pipeline)
    monkeypatch.setattr(run_all.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    run_all.main(["--dry-run", "--run-tests", "--output", str(tmp_path / "smoke")])
    args, kwargs = calls[0]
    assert args[0][1:5] == ["-m", "pytest", "tests", "-o"]
    assert kwargs["cwd"] == Path(run_all.__file__).resolve().parents[1]
    assert kwargs["check"] is True


def test_installed_command_explains_missing_test_sources(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, "__file__", str(tmp_path / "site-packages" / "scripts" / "run_all.py"))
    monkeypatch.setattr(run_all, "run_pipeline", lambda *a: pytest.fail("pipeline ran before source check"))
    with pytest.raises(SystemExit) as error:
        run_all.main(["--dry-run", "--run-tests"])
    assert error.value.code == 2
    assert "--run-tests requires a source checkout" in capsys.readouterr().err
