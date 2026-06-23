import torch

from satellite_trail_segmentation.ml_utils.checkpoints import load_checkpoint, save_checkpoint, save_weights


class DummySampler:
    pos_fraction = 0.35


def _assert_state_dict_tensors_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if torch.is_tensor(left_value):
            assert torch.allclose(left_value, right_value)
        elif isinstance(left_value, dict):
            _assert_state_dict_tensors_equal(left_value, right_value)
        else:
            assert left_value == right_value


def test_checkpoint_restores_fresh_training_objects(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.01)

    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()
    opt.step()
    sched.step()

    path = tmp_path / "ckpt.pt"
    metrics = {"score": 1.0, "threshold": 0.4}
    model_config = {"in_channels": 1, "base_channels": 4}
    save_checkpoint(path, model, opt, sched, DummySampler(), 3, metrics, model_config)

    restored_model = torch.nn.Linear(2, 1)
    restored_opt = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_sched = torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt, T_max=4, eta_min=0.01)
    loaded = load_checkpoint(path, restored_model, restored_opt, restored_sched)

    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(expected, actual)
    assert restored_opt.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    _assert_state_dict_tensors_equal(restored_opt.state_dict()["state"], opt.state_dict()["state"])
    assert restored_sched.state_dict() == sched.state_dict()
    assert loaded["epoch"] == 3
    assert loaded["metrics"] == metrics
    assert loaded["model_config"] == model_config
    assert loaded["sampler"] == 0.35


def test_checkpoint_allows_missing_sampler(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2)
    path = tmp_path / "ckpt_no_sampler.pt"

    save_checkpoint(path, model, opt, sched, None, 1, {"score": 0.5}, {"model": "linear"})
    loaded = load_checkpoint(path, torch.nn.Linear(2, 1))

    assert loaded["sampler"] is None
    assert loaded["epoch"] == 1
    assert loaded["metrics"] == {"score": 0.5}
    assert loaded["model_config"] == {"model": "linear"}


def test_weights_only_round_trip(tmp_path):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(-1.0)
    path = tmp_path / "weights.pt"
    save_weights(path, model, {"a": 1})

    restored_model = torch.nn.Linear(2, 1)
    loaded = load_checkpoint(path, restored_model)

    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(expected, actual)
    assert loaded["model_config"] == {"a": 1}
    assert "optimizer_state_dict" not in loaded
    assert "scheduler_state_dict" not in loaded
