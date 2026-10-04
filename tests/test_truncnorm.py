from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from scipy import stats

from wincurse import RootFindingError, mean_quantile, tn_cdf, tn_ppf, tn_sf

mp.mp.dps = 50


def _phi(t) -> mp.mpf:
    t = mp.mpf(t)
    if t == mp.inf:
        return mp.mpf(1)
    if t == -mp.inf:
        return mp.mpf(0)
    return mp.erfc(-t / mp.sqrt(2)) / 2


def mp_tn_cdf(z, a, b) -> mp.mpf:
    """Truncated standard-normal CDF at 50 digits, using the better-conditioned tail."""
    if a >= 0 or a + b > 0:
        q = lambda t: _phi(-t)  # noqa: E731
        return (q(a) - q(z)) / (q(a) - q(b))
    return (_phi(z) - _phi(a)) / (_phi(b) - _phi(a))


def mp_tn_ppf(p, a, b) -> mp.mpf:
    f = lambda z: mp_tn_cdf(z, a, b) - mp.mpf(p)  # noqa: E731
    lo = mp.mpf(a) if np.isfinite(a) else mp.mpf(-50)
    hi = mp.mpf(b) if np.isfinite(b) else mp.mpf(a) + 60
    return mp.findroot(f, (lo, hi), solver="anderson", tol=1e-45, maxsteps=500)


@pytest.mark.parametrize(
    ("z", "a", "b"),
    [
        (40.5, 40.0, np.inf),
        (1000.001, 1000.0, np.inf),
        (-40.5, -np.inf, -40.0),
        (-45.0, -45.2, -44.0),
        (45.2, 45.0, 46.0),
        (0.3, -1.0, 2.0),
        (-5.0, -np.inf, 1.0),
        (7.5, 7.0, 8.0),
        (30.0001, 30.0, 30.001),
    ],
)
def test_cdf_matches_mpmath_in_extreme_tails(z, a, b):
    expected = float(mp_tn_cdf(z, a, b))
    got = float(tn_cdf(z, 0.0, 1.0, a, b))
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-300)
    sf_expected = float(1 - mp_tn_cdf(z, a, b)) if expected < 0.5 else None
    if sf_expected is not None:
        assert float(tn_sf(z, 0.0, 1.0, a, b)) == pytest.approx(sf_expected, rel=1e-9)


@pytest.mark.parametrize(
    ("p", "a", "b"),
    [
        (0.3, 40.0, np.inf),
        (0.975, 40.0, np.inf),
        (0.01, -np.inf, -30.0),
        (0.5, 5.0, 6.0),
        (0.2, -2.0, 3.0),
    ],
)
def test_ppf_matches_mpmath_in_extreme_tails(p, a, b):
    expected = float(mp_tn_ppf(p, a, b))
    assert float(tn_ppf(p, 0.0, 1.0, a, b)) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("a", "b"), [(-1.0, 2.0), (0.5, np.inf), (-np.inf, -0.5), (-np.inf, np.inf)]
)
def test_matches_scipy_truncnorm_in_ordinary_cases(a, b):
    mu, sigma = 0.7, 2.3
    lower, upper = mu + sigma * a, mu + sigma * b
    x = np.linspace(max(a, -4.0), min(b, 5.0), 27)[1:-1] * sigma + mu
    ref = stats.truncnorm(a, b, loc=mu, scale=sigma)
    np.testing.assert_allclose(
        tn_cdf(x, mu, sigma, lower, upper), ref.cdf(x), rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(tn_sf(x, mu, sigma, lower, upper), ref.sf(x), rtol=1e-10, atol=1e-14)
    p = np.array([0.001, 0.2, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(tn_ppf(p, mu, sigma, lower, upper), ref.ppf(p), rtol=1e-8)


def test_cdf_and_sf_are_complementary_and_clipped():
    z = np.array([-10.0, 0.0, 3.0, 50.0])
    cdf, sf = tn_cdf(z, 0, 1, -5, 20), tn_sf(z, 0, 1, -5, 20)
    np.testing.assert_allclose(cdf + sf, 1.0, atol=1e-15)
    assert cdf[0] == 0.0
    assert sf[-1] == 0.0


def test_textbook_formula_breaks_down_where_log_space_succeeds():
    a = 40.0
    with np.errstate(invalid="ignore"):
        textbook = (stats.norm.cdf(40.5) - stats.norm.cdf(a)) / (1.0 - stats.norm.cdf(a))
    assert np.isnan(textbook)
    assert 0.0 < tn_cdf(40.01, 0, 1, a, np.inf) < 1.0


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="lower < upper"):
        tn_cdf(0.0, 0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        tn_cdf(0.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="NaN"):
        tn_cdf(np.nan, 0.0, 1.0)


def test_mean_quantile_without_truncation_is_the_normal_quantile():
    p = np.array([0.025, 0.5, 0.975])
    got = mean_quantile(-np.inf, np.inf, p)
    np.testing.assert_allclose(got, -stats.norm.ppf(p), atol=1e-10)


def test_mean_quantile_inverts_the_cdf():
    lp, up = -1.7, 2.4
    for p in (0.01, 0.3, 0.5, 0.8, 0.99):
        u = mean_quantile(lp, up, p)
        assert float(tn_cdf(0.0, u, 1.0, lp, up)) == pytest.approx(p, abs=1e-10)


def test_mean_quantile_is_monotone_in_p():
    p = np.linspace(0.01, 0.99, 50)
    u = mean_quantile(-0.3, np.inf, p)
    assert np.all(np.diff(u) < 0)


def test_mean_quantile_spans_thousands_of_standard_errors():
    # The winner barely wins: the window below the observation has width 1e-5 SE.
    lo, mid, hi = mean_quantile(-1e-5, np.inf, [0.975, 0.5, 0.025])
    assert hi < -2000.0
    assert lo < mid < hi
    for u, p in ((lo, 0.975), (mid, 0.5), (hi, 0.025)):
        assert float(tn_cdf(0.0, u, 1.0, -1e-5, np.inf)) == pytest.approx(p, abs=1e-8)


def test_hybrid_window_confines_the_root_to_the_projection_set():
    c = 3.0
    for lp, up in ((-0.01, np.inf), (-2.0, 1.0), (-np.inf, np.inf)):
        u = mean_quantile(lp, up, [0.01, 0.5, 0.99], c)
        assert np.all(np.abs(u) <= c)
        assert np.all(np.diff(u) < 0)


def test_hybrid_cdf_is_monotone_in_the_mean():
    lp, up, c = -1.0, 4.0, 3.2
    u = np.linspace(-c, c, 400)
    a = np.maximum(lp - u, -c)
    b = np.minimum(up - u, c)
    f = np.array([float(tn_cdf(-ui, 0.0, 1.0, ai, bi)) for ui, ai, bi in zip(u, a, b, strict=True)])
    assert np.all(np.diff(f) <= 1e-12)


def test_failed_bracketing_is_reported_not_hidden():
    out = mean_quantile(-1e-5, np.inf, [0.975, 0.5], max_expand=2)
    assert np.all(np.isnan(out))
    assert isinstance(RootFindingError("x"), RuntimeError)


def test_window_must_contain_the_observation():
    with pytest.raises(ValueError, match="contain the observation"):
        mean_quantile(0.5, 1.0, 0.5)
    with pytest.raises(ValueError, match="strictly between"):
        mean_quantile(-1.0, 1.0, 1.0)
