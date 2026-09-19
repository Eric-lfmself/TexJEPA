"""Missing values and synthetic labels must survive table generation."""
import csv
import pytest
from report.build_tables import build_tables


def fixture_result():
    return {"metadata":{"evidence":"synthetic_smoke"},"robustness":[
        {"model":"test mini","paper_model":"I-JEPA-H/300","group":"main","protocol":"linear",
         "perturbation":"clean","severity":0,"auroc":0.6,"delta_auroc":0,"drift":0,"parameters":{},"evidence":"synthetic_smoke"}]}


def test_missing_is_not_fabricated(tmp_path):
    result=fixture_result(); manifest=build_tables(result,tmp_path)
    text=(tmp_path/"table_IV.csv").read_text()
    assert "NR" in text and "synthetic_smoke" in text and "0.910" not in text
    assert len(manifest["tables"]) == 11


def test_duplicate_and_mixed_evidence_rejected(tmp_path):
    result=fixture_result(); result["robustness"]*=2
    with pytest.raises(ValueError,match="Duplicate"):
        build_tables(result,tmp_path)
    result=fixture_result(); result["robustness"][0]["evidence"]="measured_local"
    with pytest.raises(ValueError,match="Mixed"):
        build_tables(result,tmp_path)


def test_other_sections_cannot_relabel_evidence(tmp_path):
    result=fixture_result()
    result["lesion"]=[{"model":"mini", "evidence":"measured_local"}]
    with pytest.raises(ValueError,match="evidence"):
        build_tables(result,tmp_path)


def test_severity_text_cannot_overwrite_numeric_condition(tmp_path):
    result=fixture_result()
    a=dict(result["robustness"][0], perturbation="gaussian_noise",severity=0.1)
    result["robustness"] += [a, dict(a,severity="0.10")]
    with pytest.raises(ValueError,match="Duplicate"):
        build_tables(result,tmp_path)


def test_custom_sigmas_preserve_default_columns_and_do_not_collide(tmp_path):
    result = fixture_result()
    clean = result["robustness"][0]
    for sigma, auroc in ((.051, .51), (.052, .52), (.15, .75), (0., .6)):
        result["robustness"].append(dict(clean, perturbation="gaussian_noise", severity=sigma,
                                           auroc=auroc, delta_auroc=auroc-.6))
    build_tables(result, tmp_path)
    with (tmp_path / "table_IV.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    for name in ("sigma=0.05", "sigma=0.10", "sigma=0.20", "sigma=0.30"):
        assert row[name] == "NR"
    assert row["sigma=0.051"] == "0.510000"
    assert row["sigma=0.052"] == "0.520000"
    assert row["sigma=0.15"] == "0.750000"
    assert row["sigma=0.0"] == "0.600000"
    assert row["Clean"] == "0.600000"


def test_unmeasured_configured_sigma_has_explicit_missing_column(tmp_path):
    result = fixture_result()
    result["metadata"]["config"] = {"perturbations": {"gaussian_noise": [.075]}}
    build_tables(result, tmp_path)
    with (tmp_path / "table_IV.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["sigma=0.075"] == "NR"


def test_signed_zero_sigmas_are_the_same_condition(tmp_path):
    result = fixture_result()
    row = dict(result["robustness"][0], perturbation="gaussian_noise", severity=0.)
    result["robustness"] += [row, dict(row, severity=-0.)]
    with pytest.raises(ValueError, match="Duplicate"):
        build_tables(result, tmp_path)


@pytest.mark.parametrize("sigma", [-.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_gaussian_severity_rejected(tmp_path, sigma):
    result = fixture_result()
    result["robustness"].append(dict(result["robustness"][0],
                                       perturbation="gaussian_noise", severity=sigma))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_tables(result, tmp_path)
