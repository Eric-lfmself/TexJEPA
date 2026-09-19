"""A failed checkpoint resume must preserve model and populated optimizer state."""
import copy

import pytest
import torch

from models.checkpoints import CheckpointError, load_checkpoint, save_checkpoint
from models.vit_mini import MiniViT


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(173)
    torch.set_num_threads(1)


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_state_equal(value, reference)
    else:
        assert actual == expected


def _training_state(*, optimizer_class=torch.optim.SGD, two_groups=False, lr=0.037):
    model = MiniViT().to(device="cpu")
    parameters = list(model.parameters())
    groups = ([{"params": parameters[:1]}, {"params": parameters[1:]}]
              if two_groups else parameters)
    optimizer = optimizer_class(groups, lr=lr, momentum=0.9)
    images = torch.rand(2, 3, 32, 32, device="cpu")
    model(images).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state
    assert all("momentum_buffer" in state for state in optimizer.state.values())
    return model, optimizer


def test_missing_optimizer_state_rejects_without_mutation(tmp_path):
    source = MiniViT().to(device="cpu")
    model, optimizer = _training_state()
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    path = save_checkpoint(tmp_path / "model_only.pt", source, {"mock": True})

    with pytest.raises(CheckpointError, match="no optimizer state"):
        load_checkpoint(path, model, optimizer=optimizer)

    _assert_state_equal(model.state_dict(), before_model)
    _assert_state_equal(optimizer.state_dict(), before_optimizer)


def test_incompatible_optimizer_groups_roll_back_both_states(tmp_path):
    source, source_optimizer = _training_state()
    model, optimizer = _training_state(two_groups=True, lr=0.021)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    path = save_checkpoint(tmp_path / "one_group.pt", source, {"mock": True},
                           optimizer=source_optimizer)

    with pytest.raises(CheckpointError, match="different number of parameter groups") as caught:
        load_checkpoint(path, model, optimizer=optimizer)

    assert isinstance(caught.value.__cause__, ValueError)
    _assert_state_equal(model.state_dict(), before_model)
    _assert_state_equal(optimizer.state_dict(), before_optimizer)


class _FailAfterOptimizerMutation(torch.optim.SGD):
    fail_next_load = False

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        if self.fail_next_load:
            self.fail_next_load = False
            self.param_groups[0]["lr"] = 99.0
            for state in self.state.values():
                state["momentum_buffer"].add_(41.0)
            raise RuntimeError("injected failure after optimizer mutation")
        return result


def test_failure_after_optimizer_mutation_rolls_back_buffers_and_groups(tmp_path):
    source, source_optimizer = _training_state(lr=0.013)
    model, optimizer = _training_state(optimizer_class=_FailAfterOptimizerMutation)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    path = save_checkpoint(tmp_path / "complete.pt", source, {"mock": True},
                           optimizer=source_optimizer)
    optimizer.fail_next_load = True

    with pytest.raises(CheckpointError, match="injected failure") as caught:
        load_checkpoint(path, model, optimizer=optimizer)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert optimizer.fail_next_load is False
    _assert_state_equal(model.state_dict(), before_model)
    _assert_state_equal(optimizer.state_dict(), before_optimizer)


def test_successful_resume_restores_model_optimizer_and_metadata(tmp_path):
    source, source_optimizer = _training_state(lr=0.013)
    model, optimizer = _training_state(lr=0.021)
    metadata = {"mock": True, "actual_training_steps": 1}
    path = save_checkpoint(tmp_path / "valid.pt", source, metadata,
                           optimizer=source_optimizer)

    restored_metadata = load_checkpoint(path, model, optimizer=optimizer)

    assert restored_metadata == metadata
    _assert_state_equal(model.state_dict(), source.state_dict())
    _assert_state_equal(optimizer.state_dict(), source_optimizer.state_dict())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
