"""Published tables stay current, recover from failed writes, and retain condition identity."""
import csv
import importlib
import json
from pathlib import Path

import pytest

builder = importlib.import_module("report.build_tables")


def _results(evidence="synthetic_smoke", *, robustness=True):
    data = {"metadata": {"evidence": evidence}, "interventions": [
        {"model": "mini", "protocol": "linear", "intervention": "nca",
         "noise_sigma": .05, "noise_auroc": .6543214, "evidence": evidence}]}
    if robustness:
        data["robustness"] = [{"model": "mini", "paper_model": "I-JEPA-H/300", "group": "main",
            "protocol": "linear", "perturbation": "clean", "severity": 0,
            "auroc": .6, "delta_auroc": 0, "drift": 0, "parameters": {}, "evidence": evidence}]
    return data


def _snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _assert_readable_previous(folder, before):
    assert _snapshot(folder) == before
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["metadata"]["evidence"] == "synthetic_smoke"
    for table in manifest["tables"]:
        with (folder / table["csv"]).open(newline="") as handle:
            assert next(csv.reader(handle))
        assert (folder / table["markdown"]).read_text().startswith("# ")


def test_regeneration_removes_old_optional_tables_but_preserves_user_files(tmp_path):
    output = tmp_path / "published"
    builder.build_tables(_results(), output)
    (output / "notes.txt").write_text("Retain my notes exactly.\n")
    (output / "user").mkdir()
    # Reserved filenames in user subdirectories are not tool-owned output files.
    (output / "user" / "table_IV.csv").write_text("User data,not generated\n")
    (output / "note-link").symlink_to("notes.txt")
    output.chmod(0o700)
    manifest = builder.build_tables(_results("measured_local", robustness=False), output)

    assert not (output / "robustness_long.csv").exists()
    assert not (output / "robustness_long.md").exists()
    assert "robustness_long" not in {table["table"] for table in manifest["tables"]}
    assert (output / "notes.txt").read_text() == "Retain my notes exactly.\n"
    assert (output / "user" / "table_IV.csv").read_text() == "User data,not generated\n"
    assert (output / "note-link").is_symlink()
    assert (output / "note-link").readlink() == Path("notes.txt")
    assert output.stat().st_mode & 0o777 == 0o700
    for table in manifest["tables"]:
        assert "synthetic_smoke" not in (output / table["csv"]).read_text()
        markdown = (output / table["markdown"]).read_text()
        assert "synthetic_smoke" not in markdown and "measured_local" in markdown
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert "synthetic_smoke" not in (output / "INDEX.md").read_text()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["published"]


def test_generation_failure_keeps_previous_complete_output(tmp_path, monkeypatch):
    output = tmp_path / "published"
    builder.build_tables(_results(), output)
    before = _snapshot(output)
    original = builder._write_table

    def fail_partway(folder, name, *args):
        if name == "table_VII":
            raise OSError("simulated table write failure")
        return original(folder, name, *args)

    monkeypatch.setattr(builder, "_write_table", fail_partway)
    with pytest.raises(OSError, match="simulated table write failure"):
        builder.build_tables(_results("measured_local", robustness=False), output)
    _assert_readable_previous(output, before)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["published"]


def test_publication_failure_restores_previous_complete_output(tmp_path, monkeypatch):
    output = tmp_path / "published"
    builder.build_tables(_results(), output)
    (output / "notes.txt").write_text("Keep this user note.\n")
    before = _snapshot(output)
    original = Path.replace
    attempted = []

    def fail_new_directory(self, target):
        if self.name == "tables" and Path(target) == output:
            attempted.append(True)
            raise OSError("simulated directory publication failure")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_new_directory)
    with pytest.raises(OSError, match="simulated directory publication failure"):
        builder.build_tables(_results("measured_local", robustness=False), output)
    assert attempted == [True]
    _assert_readable_previous(output, before)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["published"]


def test_failure_does_not_publish_an_incomplete_first_output(tmp_path, monkeypatch):
    output = tmp_path / "published"
    original = builder._write_table

    def fail_partway(folder, name, *args):
        if name == "table_VII":
            raise OSError("simulated first generation failure")
        return original(folder, name, *args)

    monkeypatch.setattr(builder, "_write_table", fail_partway)
    with pytest.raises(OSError, match="simulated first generation failure"):
        builder.build_tables(_results(), output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_intervention_conditions_round_trip_without_rounding_metric_changes(tmp_path):
    data = _results(robustness=False)
    base = data["interventions"][0]
    sigmas = [.05, .05000001, .05000002]
    data["interventions"] = [dict(base, noise_sigma=sigma) for sigma in sigmas]
    builder.build_tables(data, tmp_path / "published")
    with (tmp_path / "published" / "table_XIII.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["noise_sigma"]) for row in rows] == sigmas
    assert len({row["noise_sigma"] for row in rows}) == len(sigmas)
    assert rows[0]["noise_sigma"] == "0.050000"
    assert {row["noise_auroc"] for row in rows} == {"0.654321"}
    markdown = (tmp_path / "published" / "table_XIII.md").read_text()
    for sigma in sigmas[1:]:
        assert f"| {sigma!r} |" in markdown


def test_top_level_symlink_stays_an_alias_for_the_updated_directory(tmp_path):
    target = tmp_path / "actual-tables"
    builder.build_tables(_results(), target)
    (target / "notes.txt").write_text("Keep user notes in the actual directory.\n")
    alias = tmp_path / "published"
    alias.symlink_to("actual-tables", target_is_directory=True)

    builder.build_tables(_results("measured_local", robustness=False), alias)

    assert alias.is_symlink()
    assert alias.readlink() == Path("actual-tables")
    assert not (target / "robustness_long.csv").exists()
    assert not (target / "robustness_long.md").exists()
    assert (target / "notes.txt").read_text() == "Keep user notes in the actual directory.\n"
    assert _snapshot(alias) == _snapshot(target)
    assert json.loads((target / "manifest.json").read_text())["metadata"]["evidence"] == "measured_local"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["actual-tables", "published"]


@pytest.mark.parametrize("rollback_error", [OSError, KeyboardInterrupt])
def test_failed_or_interrupted_rollback_retains_recoverable_previous_output(tmp_path, monkeypatch, rollback_error):
    output = tmp_path / "published"
    builder.build_tables(_results(), output)
    (output / "notes.txt").write_text("Retain this note even when rollback fails.\n")
    before = _snapshot(output)
    original = Path.replace

    def fail_publication_and_rollback(self, target):
        if Path(target) == output:
            if self.name == "tables":
                raise OSError("simulated publication failure")
            if self.name == "previous":
                raise rollback_error("simulated rollback failure")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_rollback)
    expected_error = RuntimeError if rollback_error is OSError else KeyboardInterrupt
    with pytest.raises(expected_error):
        builder.build_tables(_results("measured_local", robustness=False), output)

    backups = list(tmp_path.glob(".published-*/previous"))
    assert len(backups) == 1
    _assert_readable_previous(backups[0], before)
