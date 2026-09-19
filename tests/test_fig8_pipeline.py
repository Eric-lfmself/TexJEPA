"""Fig.8 must measure real tiny augmented probes and retain both noise levels."""
import csv
from configs import load_config
from scripts.run_all import run_pipeline


def test_fig8_trains_augmented_heads_and_exports_nca_all_sigmas(tmp_path):
    config = load_config(overrides={"data": {"smoke_samples": 4},
        "perturbations": {"gaussian_noise": [.05,.1], "gaussian_blur": [],
                          "brightness_shift": [], "contrast_scaling": [],
                          "frequency_bands": [], "band_modes": []}})
    result = run_pipeline(config, tmp_path / "run")
    trained = [row for row in result["training"] if row["phase"] == "train_aug"]
    assert len(trained) == 2 and all(len(row["history"]) == 1 for row in trained)
    assert all(row["augmentation"]["historical_parameter_recovered"] is False for row in trained)
    assert len(result["train_aug"]) == 2
    with (tmp_path / "run/tables/table_XIII.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for name in {row["model"] for row in trained}:
        nca = [row for row in rows if row["model"] == name and row["intervention"] == "noise_consistency_adapter"]
        assert {row["noise_sigma"] for row in nca} == {"0.050000", "0.100000"}
        assert all(row["protocol"] == "nca" for row in nca)
        aug = [row for row in rows if row["model"] == name and row["intervention"] == "train_aug"]
        assert len(aug) == 2 and all(row["protocol"] == "linear_train_aug" for row in aug)
    assert any(row["intervention"] == "token_drift_noise" for row in rows)
