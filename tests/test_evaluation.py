"""Maße von Hand, Analyse des gezeigten Laufs, Tabelle, Verdikt-Zweige und die Struktur der Experimente."""

import math

import numpy as np
import pytest

import tei_constants as C
import tei_evaluation as ev
from conftest import make
from tei_scenario import Settings


def test_german_number_formatting():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.signed(1.6) == "+1,60" and ev.signed(-0.05) == "−0,05" and ev.signed(0.0) == "+0,00"


def test_pair_accuracy_counts_ties_as_half():
    t = [0.0, 1.0, 2.0]
    assert ev.pair_accuracy([0.0, 1.0, 2.0], t) == 1.0
    assert ev.pair_accuracy([2.0, 1.0, 0.0], t) == 0.0
    assert ev.pair_accuracy([1.0, 1.0, 1.0], t) == 0.5
    assert ev.pair_accuracy([0.0, 0.0, 2.0], t) == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_tie_share_and_tolerance():
    assert ev.tie_share([0.0, 0.0, 1.0, 1.0]) == pytest.approx(2 / 6)
    assert ev.tie_share([0.0, 1e-12, 1.0]) == pytest.approx(1 / 3)
    assert ev.tie_share([0.0, 1e-6, 1.0]) == 0.0


def test_scale_alignment_and_rmse_by_hand():
    truth = [0.0, 1.0, 2.0, 3.0]
    assert ev.align_scale([10.0, 10.5, 11.0, 11.5], truth) == pytest.approx(2.0)
    assert ev.aligned_rmse([10.0, 10.5, 11.0, 11.5], truth) == pytest.approx(0.0, abs=1e-12)
    assert ev.align_scale([1.0, 1.0, 1.0, 1.0], truth) == 0.0                           # nichts zu skalieren
    # Schätzer mit einem Fehler: (0, 1, 2, 4)-Schätzer gegen (0, 1, 2, 3)
    est = np.array([0.0, 1.0, 2.0, 4.0])
    e, t = est - est.mean(), np.array(truth) - 1.5
    c = (e @ t) / (e @ e)
    assert ev.aligned_rmse(est, truth) == pytest.approx(math.sqrt(np.mean((c * e - t) ** 2)))


def test_player_error_has_the_sign_of_the_overestimate():
    truth = [0.0, 1.0, 2.0]
    assert ev.player_error([0.5, 1.0, 2.0], truth, 0) > 0
    assert ev.player_error([-0.5, 1.0, 2.0], truth, 0) < 0
    assert ev.player_error([0.0, 1.0, 2.0], truth, 0) == pytest.approx(0.0)


def test_places_above_last_counts_players_below():
    assert ev.places_above_last([3.0, 1.0, 2.0, 0.5], 0) == 3 and ev.places_above_last([0.0, 1.0, 2.0], 0) == 0 and ev.places_above_last([1.0, 1.0, 2.0], 0) == 0


def test_design_rank_of_hand_games():
    g = make(4, [([0, 1], [2, 3], True), ([0, 1], [2, 3], False)])
    assert ev.design_rank(g) == 1
    g2 = make(4, [([0, 1], [2, 3], True), ([0, 2], [1, 3], False), ([0, 3], [1, 2], True)])
    assert ev.design_rank(g2) == 3


def test_estimate_all_returns_every_estimator_with_the_documented_scales():
    s = Settings(games=100, seed=2)
    a = ev.analyse(s)
    assert set(a.estimates) >= {*C.EST_NAMES, "Maximum Likelihood", "ts_sd"}
    assert a.estimates["Siegquote"].min() >= 0 and a.estimates["Siegquote"].max() <= 1
    assert all(0.0 <= a.accuracy[k] <= 1.0 for k in a.accuracy) and abs(sum(a.kind_share) - 1.0) < 1e-12


def test_analysis_matches_the_direct_computation():
    s = Settings(teams=3, size=2, games=80, q=0.5, seed=5)
    a = ev.analyse(s)
    from tei_scenario import make_games
    g = make_games(s)
    est = ev.estimate_all(g, s.k_theta, s.lam)
    assert a.accuracy["Ridge"] == pytest.approx(ev.pair_accuracy(est["Ridge"], g.s)) and a.rank == ev.design_rank(g)
    assert a.weak_error["Siegquote"] == pytest.approx(ev.player_error(est["Siegquote"], g.s, 0))


def test_table_has_a_row_per_player_with_team_and_aligned_estimates():
    s = Settings(teams=3, size=2, games=80, seed=5)
    rows = ev.table(ev.analyse(s))
    assert len(rows) == 6 and {r["Mannschaft"] for r in rows} == {1, 2, 3}
    assert rows[0]["wahre Stärke"] < rows[-1]["wahre Stärke"] and set(rows[0]) >= {"Spieler", "Siegquote", "Elo-Team", "Ridge", "TrueSkill-Teams", "± TrueSkill"}
    assert np.mean([r["Ridge"] for r in rows]) == pytest.approx(C.BASE, abs=1.0)


def test_verdict_analysis_reports_missing_rank_only_for_fixed_teams():
    fixed = ev.verdict_analysis(ev.analyse(Settings(games=100, q=0.0)))
    assert "nicht zu trennen" in fixed[1] and "Rang 3 von höchstens 11" in fixed[1]
    mixed = ev.verdict_analysis(ev.analyse(Settings(games=300, q=1.0)))
    assert "nicht zu trennen" not in mixed[1] and "Rang 11 von höchstens 11" in mixed[1]


def test_verdict_analysis_mentions_the_windschatten_only_when_it_is_on():
    on = ev.verdict_analysis(ev.analyse(Settings(games=200, q=1.0, use_wind=True, wind=0.9)))[1]
    off = ev.verdict_analysis(ev.analyse(Settings(games=200, q=1.0)))[1]
    assert "Der Schwächste liegt bei der Siegquote um" in on and "Der Schwächste" not in off


def test_experiment_rows_have_the_documented_shape():
    rot = ev.sweep_rotation()
    assert [(r["q"], r["games"]) for r in rot] == [(q, g) for q in C.Q_GRID for g in C.GAMES_GRID]
    assert set(rot[0]["acc"]) == {*C.EST_NAMES, "Maximum Likelihood"} and 0 <= rot[0]["ties_rate"] <= 1
    wind = ev.sweep_wind()
    assert [r["c"] for r in wind] == list(C.WIND_GRID) and set(wind[0]["err"]) == {*C.EST_NAMES, "Maximum Likelihood"}
    lam = ev.sweep_lambda()
    assert [r["lam"] for r in lam] == list(C.LAM_GRID)
    agg = ev.sweep_aggregate()
    assert [r["aggregate"] for r in agg] == list(C.AGGREGATES)


def test_verdicts_quote_their_rows():
    rot = ev.sweep_rotation()
    f = ev.verdict_fixed(rot)
    assert f[0] == "warning" and "Rang 3 von 11" in f[1]
    assert ev.de(ev._rot(rot, 0.1, 600)["acc"]["Ridge"]) in ev.verdict_rotation(rot)[1]
    assert ev.de(ev._rot(rot, 0.1, 200)["acc"]["Siegquote"]) in ev.verdict_methods(rot)[1]


def test_verdict_names_the_estimator_with_the_best_pair_accuracy():
    a = ev.analyse(Settings(games=300, q=1.0, seed=3))
    best = max(C.EST_NAMES, key=lambda k: a.accuracy[k])
    assert f"am besten **{best}**" in ev.verdict_analysis(a)[1]
    worst = min(C.EST_NAMES, key=lambda k: a.accuracy[k])
    assert f"am besten **{worst}**" not in ev.verdict_analysis(a)[1] or worst == best


def _hand_analysis(rank, n=6, use_wind=False, wind=0.0):
    g = make(n, [([0, 1], [2, 3], True)])
    s = Settings(teams=3, size=2, use_wind=use_wind, wind=wind)
    names = (*C.EST_NAMES, "Maximum Likelihood")
    zero = {k: 0.0 for k in names}
    return ev.Analysis(s, g, {}, {k: 0.7 for k in names}, zero, zero, rank, zero, {k: 0 for k in names}, 0.0, (1.0, 0.0, 0.0))


def test_missing_rank_is_reported_from_one_direction_on():
    assert "nicht zu trennen" not in ev.verdict_analysis(_hand_analysis(5))[1]
    one = ev.verdict_analysis(_hand_analysis(4))[1]
    assert "1 Richtungen sind aus den Spielen nicht zu schätzen" in one
    assert "4 Richtungen" in ev.verdict_analysis(_hand_analysis(1))[1]


def test_the_windschatten_sentence_needs_a_positive_share():
    assert "Der Schwächste" not in ev.verdict_analysis(_hand_analysis(5, use_wind=True, wind=0.0))[1]
    assert "Der Schwächste" not in ev.verdict_analysis(_hand_analysis(5, use_wind=False, wind=0.5))[1]
    assert "Der Schwächste" in ev.verdict_analysis(_hand_analysis(5, use_wind=True, wind=0.05))[1]


def test_table_rows_are_numbered_and_assigned_to_their_team():
    s = Settings(teams=3, size=2, games=80, seed=5)
    a = ev.analyse(s)
    rows = ev.table(a)
    assert [r["Spieler"] for r in rows] == [1, 2, 3, 4, 5, 6]
    for p, r in enumerate(rows):
        assert r["Mannschaft"] == 1 + next(k for k, t in enumerate(a.games.teams) if p in t)


def test_table_estimates_are_centred_on_1500_and_follow_the_truth():
    a = ev.analyse(Settings(games=400, q=1.0, seed=4))
    rows = ev.table(a)
    truth = np.array([r["wahre Stärke"] for r in rows], dtype=float)
    assert abs(truth.mean() - C.BASE) < 1.0
    for name in C.EST_NAMES:
        est = np.array([r[name] for r in rows], dtype=float)
        assert abs(est.mean() - C.BASE) < 1.0, name
        assert np.corrcoef(est, truth)[0, 1] > 0.7, name
