import random

import numpy as np
import pytest
import torch

from satellite_trail_segmentation.ml_utils.lr_scheduler import create_cos_lr_sched
from satellite_trail_segmentation.ml_utils.seed import make_generator, set_seed


def test_make_generator_reproducible():
    g1 = make_generator(123)
    g2 = make_generator(123)
    assert torch.rand(3, generator=g1).tolist() == torch.rand(3, generator=g2).tolist()


def test_set_seed_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_create_cos_lr_sched_variants():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = create_cos_lr_sched(opt, epochs=4, warmup_epochs=2, eta_min=0.01)
    assert sched is not None
    opt2 = torch.optim.SGD(model.parameters(), lr=0.1)
    sched2 = create_cos_lr_sched(opt2, epochs=4, warmup_epochs=0, eta_min=0.01)
    assert sched2 is not None


def test_cos_lr_scheduler_sequences_evolve_differently():
    base_lr = 0.1
    eta_min = 0.01
    model = torch.nn.Linear(1, 1)

    warm_opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    warm_sched = create_cos_lr_sched(warm_opt, epochs=6, warmup_epochs=2, eta_min=eta_min)
    warm_lrs = [warm_opt.param_groups[0]["lr"]]
    for _ in range(6):
        warm_opt.step()
        warm_sched.step()
        warm_lrs.append(warm_opt.param_groups[0]["lr"])

    cos_opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    cos_sched = create_cos_lr_sched(cos_opt, epochs=6, warmup_epochs=0, eta_min=eta_min)
    cos_lrs = [cos_opt.param_groups[0]["lr"]]
    for _ in range(6):
        cos_opt.step()
        cos_sched.step()
        cos_lrs.append(cos_opt.param_groups[0]["lr"])

    assert warm_lrs[0] == pytest.approx(base_lr * 0.1)
    assert warm_lrs[0] < warm_lrs[1] < warm_lrs[2] == pytest.approx(base_lr)
    assert warm_lrs[-1] == pytest.approx(eta_min)
    assert all(eta_min <= lr <= base_lr for lr in warm_lrs)

    assert cos_lrs[0] == pytest.approx(base_lr)
    assert cos_lrs[-1] == pytest.approx(eta_min)
    assert all(left >= right for left, right in zip(cos_lrs, cos_lrs[1:]))
    assert all(eta_min <= lr <= base_lr for lr in cos_lrs)
    assert not np.allclose(warm_lrs, cos_lrs)
