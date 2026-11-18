import torch

from deephedge.evaluation import paired_bootstrap, paired_change


def normal(n, seed, scale=1.0, shift=0.0):
    return shift + scale * torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_identical_strategies_have_no_difference():
    a = normal(10_000, 0)
    d = paired_bootstrap(a, a.clone(), "entropic", reps=50)
    assert d.difference == 0 and d.low == 0 and d.high == 0


def test_difference_in_percent_of_the_reference():
    a, b = normal(20_000, 1), normal(20_000, 1) * 1.1
    d = paired_bootstrap(a, b, "std", reps=100)
    assert abs(d.percent[0] - 10.0) < 1e-9          # the same shocks, so the difference is exact


def test_pairing_narrows_the_interval():
    a = normal(20_000, 2)
    b = a + normal(20_000, 3, scale=0.01, shift=0.02)
    d = paired_bootstrap(a, b, "mean", reps=200)
    assert d.low > 0 and d.high - d.low < 0.002


def test_change_divides_by_the_network_on_the_full_sample():
    # a rule that loses a constant on every path differs from the network by that constant in every resample, so
    # with the fixed denominator the change has no noise at all; dividing by the resampled network would add it
    before, after = normal(5_000, 8, shift=1.0), normal(5_000, 9, shift=2.0)
    point, lo, hi = paired_change((before, before + 0.3), (after, after + 0.1), "mean", reps=40)
    expected = 100 * 0.1 / float(after.mean()) - 100 * 0.3 / float(before.mean())
    assert abs(point - expected) < 1e-9 and abs(lo - expected) < 1e-9 and abs(hi - expected) < 1e-9
