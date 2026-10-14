"""Schätzer: Designmatrix, Siegquote, Elo-Team, Ridge-Regression und TrueSkill-Teams an Handwerten und Randfällen."""

import math

import numpy as np
import pytest

import tei_constants as C
import tei_estimators as E
from conftest import make


def test_design_matrix_by_hand():
    g = make(5, [([0, 1], [2, 3], True), ([4, 0], [1, 2], False)])
    X = E.design_matrix(g.a, g.b, 5)
    assert X.tolist() == [[1, 1, -1, -1, 0], [1, -1, -1, 0, 1]]
    assert X.sum(axis=1).tolist() == [0, 0]


def test_win_rate_by_hand():
    g = make(4, [([0, 1], [2, 3], True), ([0, 2], [1, 3], False), ([0, 1], [2, 3], True)])
    r = E.win_rate(g.a, g.b, g.y, 4)
    # Spieler 0: Sieg, Niederlage, Sieg = 2/3; Spieler 1: Sieg, Sieg (B gewinnt das 2. Spiel), Sieg = 1; Spieler 2: 0, 0, 0 = 0; Spieler 3: 0, 1, 0 = 1/3
    assert r.tolist() == pytest.approx([2 / 3, 1.0, 0.0, 1 / 3])


def test_win_rate_of_a_player_without_games_is_one_half():
    g = make(4, [([0, 1], [2, 3], True)])
    assert E.win_rate(g.a, g.b, g.y, 6)[4] == 0.5


def test_elo_team_one_game_by_hand_and_zero_sum():
    g = make(4, [([0, 1], [2, 3], True)])
    r = E.elo_team(g.a, g.b, g.y, 4, 0.2)
    assert r.tolist() == pytest.approx([0.1, 0.1, -0.1, -0.1]) and r.sum() == pytest.approx(0.0)


def test_elo_team_second_game_uses_the_summed_ratings():
    g = make(4, [([0, 1], [2, 3], True), ([0, 1], [2, 3], True)])
    r = E.elo_team(g.a, g.b, g.y, 4, 0.2)
    e = 1.0 / (1.0 + math.exp(-(0.2 - (-0.2))))                         # Summen 0,2 gegen −0,2
    assert r[0] == pytest.approx(0.1 + 0.2 * (1 - e)) and r.sum() == pytest.approx(0.0)


def test_ridge_gradient_is_zero_and_shrinks_with_larger_lambda():
    g = make(4, [([0, 1], [2, 3], True), ([0, 2], [1, 3], True), ([1, 2], [0, 3], False), ([0, 3], [1, 2], True)])
    X = E.design_matrix(g.a, g.b, 4)
    th = E.ridge_logit(X, g.y, 0.7)
    grad = X.T @ (g.y - E.sig(X @ th)) - 0.7 * th
    assert np.abs(grad).max() < 1e-8
    assert np.linalg.norm(E.ridge_logit(X, g.y, 5.0)) < np.linalg.norm(th)


def test_ridge_estimates_are_antisymmetric_when_all_results_flip():
    g = make(4, [([0, 1], [2, 3], True), ([0, 2], [1, 3], True), ([0, 3], [1, 2], False)])
    X = E.design_matrix(g.a, g.b, 4)
    th = E.ridge_logit(X, g.y, 1.0)
    assert E.ridge_logit(X, 1.0 - g.y, 1.0) == pytest.approx(-th, abs=1e-9)


def test_players_of_one_fixed_team_get_identical_ridge_estimates():
    g = make(4, [([0, 1], [2, 3], True)] * 5 + [([0, 1], [2, 3], False)] * 2)
    th = E.ridge_logit(E.design_matrix(g.a, g.b, 4), g.y, 1.0)
    assert th[0] == pytest.approx(th[1], abs=1e-10) and th[2] == pytest.approx(th[3], abs=1e-10) and th[0] > th[2]


def test_ridge_without_regularisation_stays_finite_thanks_to_the_tiny_floor():
    g = make(4, [([0, 1], [2, 3], True)] * 3)                           # lauter Siege: das Maximum Likelihood divergiert ohne Floor
    th = E.ridge_logit(E.design_matrix(g.a, g.b, 4), g.y, 0.0)
    assert np.isfinite(th).all() and th[0] > 0


def test_trueskill_team_single_game_of_two_single_players_by_hand():
    g = make(2, [([0], [1], True)])
    mu, sd = E.trueskill_team(g.a, g.b, g.y, 2, beta=1.0, sd0=1.0)
    v = (1.0 / math.sqrt(2 * math.pi)) / 0.5                            # c² = 2·1·1 + 1 + 1 = 4, t = 0
    assert mu.tolist() == pytest.approx([v / 2.0, -v / 2.0])
    assert sd[0] ** 2 == pytest.approx(1.0 - v * v / 4.0)


def test_trueskill_team_of_two_splits_the_update_by_variance():
    g = make(4, [([0, 1], [2, 3], True)])
    mu, sd = E.trueskill_team(g.a, g.b, g.y, 4, beta=1.0, sd0=1.0)
    c = math.sqrt(2 * 2 * 1.0 + 4.0)                                    # 2 m β² + Σσ² = 4 + 4 = 8
    v = (1.0 / math.sqrt(2 * math.pi)) / 0.5
    assert mu[0] == pytest.approx(v / c) and mu[2] == pytest.approx(-v / c) and mu[0] == pytest.approx(mu[1])


def test_trueskill_team_uncertainty_shrinks_and_the_first_game_is_zero_sum():
    g = make(6, [([0, 1], [2, 3], True), ([2, 3], [4, 5], False), ([0, 4], [1, 5], True)])
    mu, sd = E.trueskill_team(g.a, g.b, g.y, 6)
    assert (sd < C.TS_SD0).all()
    mu1, _ = E.trueskill_team(g.a[:1], g.b[:1], g.y[:1], 6)
    assert mu1.sum() == pytest.approx(0.0, abs=1e-12) and mu1[4] == 0.0


def test_trueskill_team_stays_finite_for_a_huge_upset():
    g = make(2, [([0], [1], True)] * 40 + [([1], [0], True)])
    mu, sd = E.trueskill_team(g.a, g.b, g.y, 2)
    assert np.isfinite(mu).all() and (sd > 0).all()


def test_win_rate_of_a_player_with_a_single_game_is_that_result():
    g = make(4, [([0, 1], [2, 3], True)])
    r = E.win_rate(g.a, g.b, g.y, 4)
    assert r.tolist() == [1.0, 1.0, 0.0, 0.0]
