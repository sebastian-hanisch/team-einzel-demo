"""Orakel für die Einzelstärken aus Mannschaftsergebnissen: Ridge-Regression gegen scipy, Rang der Designmatrix gegen numpy, TrueSkill für Teams gegen das Paket `trueskill`,
Elo-Team gegen die Lehrbuchformel, Paartreffer und Skalenangleichung von Hand, Windschatten mit starkem Signal."""

import math

import numpy as np
import pytest

import tei_constants as C
import tei_estimators as E
import tei_evaluation as ev
import tei_games as G
from conftest import make


def _games(seed, teams=4, size=3, n_games=60, q=0.4, wind=0.2):
    return G.simulate_games(teams, size, 0.4, n_games, q, wind, C.AGGREGATES[0], seed)


def test_ridge_equals_the_scipy_maximiser():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(5):
        g = _games(seed, teams=3, size=2, n_games=40)
        X = E.design_matrix(g.a, g.b, g.n)
        lam = [0.3, 1.0, 3.0][seed % 3]

        def neg(th, X=X, y=g.y, lam=lam):
            z = X @ th
            return -(np.sum(y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)) - 0.5 * lam * th @ th)

        res = optimize.minimize(neg, np.zeros(g.n), method="BFGS", options={"gtol": 1e-10})
        assert E.ridge_logit(X, g.y, lam) == pytest.approx(res.x, abs=1e-5), seed


def test_rank_of_the_design_matrix_for_fixed_and_rotating_lineups():
    fixed = _games(1, q=0.0, wind=0.0, n_games=200)
    assert ev.design_rank(fixed) == np.linalg.matrix_rank(E.design_matrix(fixed.a, fixed.b, fixed.n)) == 3        # vier Mannschaften − 1
    mixed = _games(1, q=0.5, wind=0.0, n_games=200)
    assert ev.design_rank(mixed) == 11                                                                           # n − 1
    one = make(6, [([0, 1], [2, 3], True)])
    assert ev.design_rank(one) == 1


def test_one_rotating_game_splits_only_part_of_the_teams():
    """Vier feste Mannschaften zu 2 plus ein Spiel (0,2) gegen (1,3) aus zwei Mannschaften: der Rang steigt um genau eins."""
    base = [([0, 1], [2, 3], True), ([4, 5], [6, 7], False), ([0, 1], [4, 5], True), ([2, 3], [6, 7], True)]
    r0 = np.linalg.matrix_rank(E.design_matrix(*(lambda g: (g.a, g.b))(make(8, base)), 8))
    r1 = np.linalg.matrix_rank(E.design_matrix(*(lambda g: (g.a, g.b))(make(8, base + [([0, 2], [1, 3], True)])), 8))
    assert (r0, r1) == (3, 4)


def test_trueskill_teams_equal_the_trueskill_package():
    trueskill = pytest.importorskip("trueskill")
    env = trueskill.TrueSkill(mu=0.0, sigma=C.TS_SD0, beta=C.BETA, tau=0.0, draw_probability=0.0)
    g = _games(3, teams=3, size=2, n_games=25, q=0.7, wind=0.0)
    mu, sd = E.trueskill_team(g.a, g.b, g.y, g.n)
    ratings = [env.create_rating() for _ in range(g.n)]
    for ta, tb, res in zip(g.a, g.b, g.y):
        win, lose = (ta, tb) if res == 1.0 else (tb, ta)
        new = env.rate([[ratings[i] for i in win], [ratings[i] for i in lose]], ranks=[0, 1])
        for i, r in zip(win, new[0]):
            ratings[i] = r
        for i, r in zip(lose, new[1]):
            ratings[i] = r
    assert mu == pytest.approx([r.mu for r in ratings], abs=1e-6)
    assert sd == pytest.approx([r.sigma for r in ratings], abs=1e-6)


def test_elo_team_equals_the_textbook_formula_in_points():
    """Punkte statt θ-Einheiten: E = 1/(1 + 10^(−ΔR/400)), K Punkte je Spieler; die Summe aller Wertungen bleibt 0."""
    g = _games(2, teams=3, size=2, n_games=40, q=0.6, wind=0.0)
    K = 20.0
    pts = np.zeros(g.n)
    for ta, tb, res in zip(g.a, g.b, g.y):
        e = 1.0 / (1.0 + 10.0 ** (-(pts[ta].sum() - pts[tb].sum()) / 400.0))
        pts[ta] += K * (res - e)
        pts[tb] -= K * (res - e)
    theta = E.elo_team(g.a, g.b, g.y, g.n, K / C.SCALE)
    assert theta * C.SCALE == pytest.approx(pts, abs=1e-6) and abs(pts.sum()) < 1e-9


def test_pair_accuracy_and_scale_alignment_by_hand():
    truth = [0.0, 1.0, 2.0, 3.0]
    assert ev.pair_accuracy([0, 1, 2, 3], truth) == 1.0 and ev.pair_accuracy([3, 2, 1, 0], truth) == 0.0
    assert ev.pair_accuracy([5, 5, 5, 5], truth) == 0.5                           # alles gleich: Zufall, nicht perfekt
    assert ev.pair_accuracy([0, 0, 1, 1], truth) == pytest.approx((4 * 1.0 + 2 * 0.5) / 6)
    est = [0.0, 0.5, 1.0, 1.5]
    assert ev.align_scale(est, truth) == pytest.approx(2.0) and ev.aligned_rmse(est, truth) == pytest.approx(0.0, abs=1e-12)


def test_windschatten_with_a_strong_signal_lifts_the_weakest_in_the_win_rate_but_less_in_the_regression():
    g = G.simulate_games(4, 3, 0.3, 600, 1.0, 0.9, C.AGGREGATES[0], 2)
    est = ev.estimate_all(g, C.EXP_K / C.SCALE, 1.0)
    # Die Siegquote setzt den Schwächsten über mehrere Mitspieler; die Regression ohne Regularisierung kaum
    assert ev.places_above_last(est["Siegquote"], 0) >= 2
    assert abs(ev.player_error(est["Maximum Likelihood"], g.s, 0)) < abs(ev.player_error(est["Siegquote"], g.s, 0))
