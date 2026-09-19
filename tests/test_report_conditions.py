import csv
import pytest
from report.build_tables import build_tables


def result():
    return {"metadata": {"evidence": "synthetic_smoke"}, "interventions": [
        {"model": "mini", "protocol": "linear", "intervention": "input_gaussian",
         "noise_sigma": .05, "noise_auroc": .6, "evidence": "synthetic_smoke"}]}


def test_intervention_conditions_survive_export(tmp_path):
    data = result()
    data["interventions"].append(dict(data["interventions"][0], noise_sigma=.1))
    build_tables(data, tmp_path)
    with (tmp_path / "table_XIII.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["noise_sigma"] for row in rows] == ["0.050000", "0.100000"]
    assert all(row["protocol"] == "linear" for row in rows)


def test_numerically_duplicate_intervention_rejected(tmp_path):
    data = result()
    data["interventions"].append(dict(data["interventions"][0], noise_sigma="0.050"))
    with pytest.raises(ValueError, match="Duplicate intervention"):
        build_tables(data, tmp_path)


def test_undefined_delta_keeps_reason_in_frequency_table(tmp_path):
    data = result()
    data["robustness"] = [{"model": "mini", "protocol": "linear", "group": "main",
        "perturbation": "band_stop", "severity": "0.00-0.20", "auroc": .8,
        "delta_auroc": float("nan"), "delta_status": "undefined_support_mismatch",
        "drift": .2, "parameters": {"low": 0., "high": .2}, "evidence": "synthetic_smoke"}]
    build_tables(data, tmp_path)
    with (tmp_path / "table_VI.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["Delta AUROC"] == "NR"
    assert row["Delta status"] == "undefined_support_mismatch"


def test_token_diagnostics_are_exported_with_image_and_measure_space(tmp_path):
    data = result()
    data["token_diagnostics"] = [{"model": "mini", "protocol": "linear", "evidence": "synthetic_smoke",
        "image_id": image_id, "class_index": 2, "noise_sigma": .1,
        "noise": {"mean_drift": .4, "valid_tokens": 16, "total_tokens": 16},
        "lesion_occlusion": {"mean_drift": .05, "valid_tokens": 16, "total_tokens": 16}}
        for image_id in ("a", "b")]
    build_tables(data, tmp_path)
    with (tmp_path / "table_XIII.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["intervention"].startswith("token_")]
    assert len(rows) == 4 and {row["image_id"] for row in rows} == {"a", "b"}
    assert all(row["adapted_drift"] == "NR" for row in rows)
    noise = [row for row in rows if row["intervention"] == "token_drift_noise"]
    assert all(row["noise_sigma"] == "0.100000" and row["mean_token_drift"] == "0.400000" for row in noise)
