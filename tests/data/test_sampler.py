import numpy as np
import pytest

from satellite_trail_segmentation.data.sampler import BalancedTrailSampler, FixedStepWeightedTrailSampler


def test_balanced_sampler_length_and_positive_coverage():
    sampler = BalancedTrailSampler([1, 3], [0, 2, 4, 5], pos_fraction=0.5)
    samples = list(iter(sampler))
    assert len(samples) == len(sampler) == 4
    assert sorted([x for x in samples if x in {1, 3}]) == [1, 3]
    assert sum(x in {0, 2, 4, 5} for x in samples) == 2


def test_balanced_sampler_fixed_seed_reproducibility():
    sampler = BalancedTrailSampler(np.array([1, 2, 3]), np.array([0, 4, 5]), pos_fraction=0.6)
    np.random.seed(123)
    first = list(iter(sampler))
    np.random.seed(123)
    second = list(iter(sampler))

    assert first == second
    assert len(first) == 5
    assert sorted([idx for idx in first if idx in {1, 2, 3}]) == [1, 2, 3]
    assert sum(idx in {0, 4, 5} for idx in first) == 2


def test_balanced_sampler_uses_replacement_when_needed(monkeypatch):
    monkeypatch.setattr(np.random, "shuffle", lambda x: x)

    seen = {}

    def fake_choice(arr, size, replace):
        seen["replace"] = replace
        return np.array([arr[0]] * size)

    monkeypatch.setattr(np.random, "choice", fake_choice)
    sampler = BalancedTrailSampler([1, 2, 3, 4], [9], pos_fraction=0.5)
    samples = list(iter(sampler))
    assert seen["replace"] is True
    assert samples.count(9) == len(sampler) - 4
    assert set([1, 2, 3, 4]).issubset(samples)


def test_balanced_sampler_replacement_count_with_real_rng():
    sampler = BalancedTrailSampler(np.array([1, 2, 3]), np.array([0]), pos_fraction=0.5)
    np.random.seed(8)
    samples = list(iter(sampler))

    assert len(samples) == 6
    assert samples.count(0) == 3
    assert sorted([idx for idx in samples if idx != 0]) == [1, 2, 3]


def test_fixed_step_weighted_sampler_length_and_fraction():
    sampler = FixedStepWeightedTrailSampler([1, 3], [0, 2, 4], pos_fraction=0.4, num_samples=10)
    np.random.seed(1)
    samples = list(iter(sampler))

    assert len(samples) == len(sampler) == 10
    assert sum(idx in {1, 3} for idx in samples) == 4
    assert sum(idx in {0, 2, 4} for idx in samples) == 6


def test_fixed_step_weighted_sampler_uses_replacement():
    sampler = FixedStepWeightedTrailSampler([1], [0], pos_fraction=0.5, num_samples=8)
    np.random.seed(2)
    samples = list(iter(sampler))

    assert len(samples) == 8
    assert samples.count(1) == 4
    assert samples.count(0) == 4


def test_fixed_step_weighted_sampler_allows_single_nonempty_pool():
    neg_only = FixedStepWeightedTrailSampler([], [0, 2], pos_fraction=0.5, num_samples=5)
    pos_only = FixedStepWeightedTrailSampler([1, 3], [], pos_fraction=0.5, num_samples=5)

    np.random.seed(3)
    assert set(iter(neg_only)) <= {0, 2}
    assert set(iter(pos_only)) <= {1, 3}


def test_fixed_step_weighted_sampler_validates_inputs():
    with pytest.raises(ValueError, match="pos_fraction"):
        FixedStepWeightedTrailSampler([1], [0], pos_fraction=0.0, num_samples=10)
    with pytest.raises(ValueError, match="pos_fraction"):
        FixedStepWeightedTrailSampler([1], [0], pos_fraction=1.0, num_samples=10)
    with pytest.raises(ValueError, match="num_samples"):
        FixedStepWeightedTrailSampler([1], [0], pos_fraction=0.5, num_samples=0)
    with pytest.raises(ValueError, match="at least one positive or negative"):
        FixedStepWeightedTrailSampler([], [], pos_fraction=0.5, num_samples=10)
