"""Tiny CPU evidence for complete training, epoch resume and real checkpoint nodes."""
from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import DummyCXRDataset, collate_cxr
from models import MiniViT, IJEPA, PostTrainingObjective, BlockMaskSampler, warm_start
from models.nca import NoiseConsistencyAdapter
from perturbations import gaussian_noise
from probing import ProbeConfig, ProbeModel, fit_probe, positive_class_weights
from training import (TrainingConfig, TrainingStateError, load_training_model,
                      load_training_state, save_training_state,
                      train_post_training, train_nca_resumable)
from training.loops import collect_training_labels


def encoder(registers=0):
    return MiniViT(image_size=32, patch_size=8, embed_dim=8, depth=2,
                   num_heads=2, num_register_tokens=registers)


def jepa(registers=0):
    return IJEPA(encoder(registers), predictor_dim=8, predictor_depth=1, num_heads=2)


def loader():
    return DataLoader(DummyCXRDataset(4), batch_size=2, shuffle=True,
                       generator=torch.Generator(device="cpu").manual_seed(29),
                       collate_fn=collate_cxr, num_workers=0)


def augmentation(images, *, generator):
    return gaussian_noise(images, .05, generator=generator)


def assert_same(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize("protocol", ["linear", "mlp", "partial_ft"])
def test_probe_epoch_resume_matches_uninterrupted_and_eval_only(tmp_path, protocol):
    original = encoder()
    config = ProbeConfig(protocol=protocol, hidden_dim=16, dropout=.2, epochs=3)
    complete, full_info = fit_probe(deepcopy(original), loader(), config, device="cpu",
                                    augmentation=augmentation, augmentation_seed=17, seed=19)
    fit_probe(deepcopy(original), loader(), config, device="cpu", augmentation=augmentation,
               augmentation_seed=17, seed=19, checkpoint_dir=tmp_path, stop_after_epoch=1,
               metadata={"evidence": "synthetic_smoke", "mock": True})
    resumed, info = fit_probe(deepcopy(original), loader(), config, device="cpu",
                               augmentation=augmentation, augmentation_seed=17, seed=19,
                               checkpoint_dir=tmp_path, resume_from=tmp_path / "epoch_0001.pt",
                               metadata={"evidence": "synthetic_smoke", "mock": True})
    assert_same(complete.state_dict(), resumed.state_dict())
    assert full_info["history"] == info["history"]
    assert info["actual_training_steps"] == 6
    assert info["completed_epochs"] == 3
    target = ProbeModel(encoder(), config)
    rng_before = torch.get_rng_state().clone()
    saved = load_training_model(tmp_path / "latest.pt", target, {"mock": True})
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert saved["epoch"] == 3 and saved["epoch_is_lineage_label"] is False
    assert_same(resumed.state_dict(), target.state_dict())


def test_probe_augmentation_only_training_and_full_split_weights(tmp_path):
    dataset = DummyCXRDataset(5)
    batches = DataLoader(dataset, batch_size=2, drop_last=True, collate_fn=collate_cxr)
    expected, _ = positive_class_weights(dataset.labels_tensor())
    calls = []
    def mark(images, *, generator):
        calls.append(len(images))
        return images
    _, info = fit_probe(encoder(), batches, ProbeConfig(epochs=1), augmentation=mark,
                         validation_batches=batches, device="cpu")
    assert calls == [2, 2]
    assert info["pos_weight"] == expected.tolist()
    assert info["class_weight_sample_count"] == 5
    assert info["class_weight_label_source"] == "dataset_metadata"


def test_mutable_label_buffers_are_snapshotted():
    class RefilledBatches:
        def __iter__(self):
            labels = torch.zeros(2, 15)
            yield {"labels": labels}
            labels.fill_(1)
            yield {"labels": labels}
    labels, source = collect_training_labels(RefilledBatches())
    assert torch.equal(labels[:2], torch.zeros(2, 15))
    assert torch.equal(labels[2:], torch.ones(2, 15))


@pytest.mark.parametrize("mode,registers", [("noise", 0), ("register", 4), ("vicreg", 0)])
def test_post_training_epoch_resume_restores_mask_noise_optimizer_and_ema(tmp_path, mode, registers):
    original = jepa(registers)
    config = TrainingConfig(epochs=3, learning_rate=.001, device="cpu", seed=41)
    objective = PostTrainingObjective(mode, noise_types=("gaussian", "poisson"))
    def sampler():
        return BlockMaskSampler(4, variant=mode, seed=7)
    complete, full_info = train_post_training(deepcopy(original), objective, loader(), config,
                                              mask_sampler=sampler())
    train_post_training(deepcopy(original), objective, loader(), config, mask_sampler=sampler(),
                         checkpoint_dir=tmp_path, stop_after_epoch=1)
    resumed, info = train_post_training(deepcopy(original), objective, loader(), config,
                                         mask_sampler=sampler(), checkpoint_dir=tmp_path,
                                         resume_from=tmp_path / "epoch_0001.pt")
    assert_same(complete.state_dict(), resumed.state_dict())
    assert full_info["history"] == info["history"]
    assert info["actual_training_steps"] == 6
    assert info["history"][-1]["ema_momentum"] == 1.0
    assert all(not param.requires_grad for param in resumed.target_encoder.parameters())
    assert any(not torch.equal(original.context_encoder.state_dict()[key], tensor)
               for key, tensor in resumed.context_encoder.state_dict().items())


def test_true_v4_epoch_50_checkpoint_is_compatible_with_warm_start(tmp_path):
    dataset = DummyCXRDataset(2)
    batches = [collate_cxr([dataset[0], dataset[1]])]
    model, info = train_post_training(jepa(), PostTrainingObjective("noise", noise_types=("gaussian",)),
                                      batches, TrainingConfig(epochs=50, checkpoint_every=70, device="cpu"),
                                      mask_sampler=BlockMaskSampler(4), checkpoint_dir=tmp_path,
                                      metadata={"variant": "v4", "evidence": "synthetic_smoke", "mock": True})
    path = tmp_path / "epoch_0050.pt"
    child = jepa()
    inherited = warm_start(path, child, "v6")
    assert inherited["warm_start"]["epoch"] == 50
    assert inherited["warm_start"]["actual_training_steps"] == 50
    assert inherited["warm_start"]["epoch_is_lineage_label"] is False
    assert inherited["warm_start"]["mock"] is True
    assert_same(model.state_dict(), child.state_dict())
    assert len(info["history"]) == 50


def test_nca_resume_matches_and_backbone_remains_frozen(tmp_path):
    original = NoiseConsistencyAdapter(encoder(), feature_dim=8, hidden_dim=4)
    config = TrainingConfig(epochs=3, learning_rate=.003, seed=101, device="cpu")
    complete, full_info = train_nca_resumable(deepcopy(original), loader(), config)
    train_nca_resumable(deepcopy(original), loader(), config,
                         checkpoint_dir=tmp_path, stop_after_epoch=1)
    resumed, info = train_nca_resumable(deepcopy(original), loader(), config,
                                       checkpoint_dir=tmp_path, resume_from=tmp_path / "epoch_0001.pt")
    assert_same(complete.state_dict(), resumed.state_dict())
    assert_same(original.encoder.state_dict(), resumed.encoder.state_dict())
    assert full_info["history"] == info["history"]
    assert info["actual_training_steps"] == 6
    assert any(not torch.equal(original.adapter.state_dict()[key], tensor)
               for key, tensor in resumed.adapter.state_dict().items())


def test_resume_rejects_changed_schedule_and_partial_epoch_limit(tmp_path):
    model = jepa()
    objective = PostTrainingObjective("noise")
    config = TrainingConfig(epochs=3)
    train_post_training(model, objective, loader(), config,
                         mask_sampler=BlockMaskSampler(4), checkpoint_dir=tmp_path, stop_after_epoch=1)
    with pytest.raises(TrainingStateError, match="training_spec"):
        train_post_training(jepa(), objective, loader(), replace(config, epochs=4),
                             mask_sampler=BlockMaskSampler(4), resume_from=tmp_path / "epoch_0001.pt")
    with pytest.raises(ValueError, match="epoch boundary"):
        train_post_training(jepa(), objective, loader(), replace(config, max_steps=1),
                             mask_sampler=BlockMaskSampler(4))
    with pytest.raises(ValueError, match="index explicitly"):
        TrainingConfig(device="cuda")


def test_state_restores_python_numpy_torch_and_named_rng(tmp_path):
    model = encoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    generator = torch.Generator(device="cpu").manual_seed(91)
    path = tmp_path / "state.pt"
    save_training_state(path, model, optimizer, None, 0, [], {}, {"loader": generator}, 0, device="cpu")
    expected = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(3, generator=generator))
    load_training_state(path, model, optimizer, None, generators={"loader": generator}, device="cpu")
    actual = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(3, generator=generator))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])
    before = deepcopy(model.state_dict())
    bad = torch.load(path, map_location="cpu", weights_only=True)
    bad["model_state_dict"] = {key: value + 1 for key, value in bad["model_state_dict"].items()}
    bad["rng_state"]["python"] = (0, (), None)
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(TrainingStateError, match="previous states restored"):
        load_training_state(tmp_path / "bad.pt", model, optimizer, None,
                             generators={"loader": generator}, device="cpu")
    assert_same(before, model.state_dict())


@pytest.mark.parametrize("kind", ["ijepa", "mae"])
def test_baseline_pretraining_complete_epochs_and_resume(tmp_path, kind):
    from models import MAE
    from training import train_pretraining
    original = jepa() if kind == "ijepa" else MAE(encoder(), decoder_dim=8, decoder_depth=1, num_heads=2)
    config = TrainingConfig(epochs=3, seed=211, device="cpu")
    kwargs = lambda: {"kind": kind, "mask_sampler": BlockMaskSampler(4)}
    complete, full_info = train_pretraining(deepcopy(original), loader(), config, **kwargs())
    train_pretraining(deepcopy(original), loader(), config, checkpoint_dir=tmp_path,
                      stop_after_epoch=1, **kwargs())
    resumed, info = train_pretraining(deepcopy(original), loader(), config, checkpoint_dir=tmp_path,
                                     resume_from=tmp_path / "epoch_0001.pt", **kwargs())
    assert_same(complete.state_dict(), resumed.state_dict())
    assert info["history"] == full_info["history"]
    assert info["completed_epochs"] == 3 and info["actual_training_steps"] == 6


def test_probe_resume_restores_stochastic_validation_loader(tmp_path):
    class ValidationDataset(DummyCXRDataset):
        def __init__(self, generator):
            super().__init__(4)
            self.noise_generator = generator
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["image"] = (sample["image"] + .1 * torch.rand(sample["image"].shape,
                                                                 generator=self.noise_generator)).clamp(0, 1)
            return sample
    def validation():
        generator = torch.Generator(device="cpu").manual_seed(909)
        return DataLoader(ValidationDataset(generator), batch_size=2, shuffle=True,
                          generator=generator, collate_fn=collate_cxr, num_workers=0)
    original = encoder()
    config = ProbeConfig(protocol="mlp", hidden_dim=16, dropout=.2, epochs=3)
    complete, full_info = fit_probe(deepcopy(original), loader(), config, validation_batches=validation(), seed=73)
    fit_probe(deepcopy(original), loader(), config, validation_batches=validation(), seed=73,
              checkpoint_dir=tmp_path, stop_after_epoch=1)
    resumed, info = fit_probe(deepcopy(original), loader(), config, validation_batches=validation(), seed=73,
                              resume_from=tmp_path / "epoch_0001.pt")
    assert info["history"] == full_info["history"]
    assert_same(complete.state_dict(), resumed.state_dict())
    checkpoint = torch.load(tmp_path / "epoch_0001.pt", map_location="cpu", weights_only=True)
    assert {"validation_dataloader", "validation_sampler"} <= set(checkpoint["generator_states"])
    class PersistentValidation:
        persistent_workers = True
        def __iter__(self):
            raise AssertionError("Must reject before starting workers")
    with pytest.raises(ValueError, match="persistent_workers"):
        fit_probe(encoder(), loader(), config, validation_batches=PersistentValidation())
