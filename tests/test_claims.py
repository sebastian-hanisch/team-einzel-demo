"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import tei_constants as C
import tei_evaluation as ev


@pytest.fixture(scope="module")
def rot():
    return {(r["q"], r["games"]): r for r in ev.sweep_rotation()}


@pytest.fixture(scope="module")
def wind():
    return {r["c"]: r for r in ev.sweep_wind()}


@pytest.fixture(scope="module")
def lam():
    return {r["lam"]: r for r in ev.sweep_lambda()}


@pytest.fixture(scope="module")
def agg():
    return {r["aggregate"]: r for r in ev.sweep_aggregate()}


# --- Experiment 1: feste Mannschaften ---------------------------------------------------------------------------------------------------------

def test_with_fixed_teams_more_games_do_not_help(rot):
    for g, expected in zip(C.GAMES_GRID, (0.656, 0.662, 0.663)):
        r = rot[(0.0, g)]
        assert r["acc"]["Ridge"] == pytest.approx(expected, abs=0.004)
        assert all(abs(r["acc"][k] - r["acc"]["Ridge"]) < 0.01 for k in C.EST_NAMES)
        assert r["rank"] == 3.0 and r["ties_rate"] == pytest.approx(0.182, abs=0.003)
    assert rot[(0.0, 600)]["acc"]["Ridge"] - rot[(0.0, 60)]["acc"]["Ridge"] < 0.02


# --- Experimente 2 und 3: Rotation, Siegquote gegen Regression --------------------------------------------------------------------------------

def test_a_tenth_of_rotating_games_separates_the_players_slowly(rot):
    assert [round(rot[(0.1, g)]["acc"]["Ridge"], 3) for g in C.GAMES_GRID] == pytest.approx([0.735, 0.828, 0.910], abs=0.006)
    assert [round(rot[(0.1, g)]["acc"]["Siegquote"], 3) for g in C.GAMES_GRID] == pytest.approx([0.692, 0.721, 0.734], abs=0.006)
    assert [round(rot[(0.1, g)]["rank"], 1) for g in C.GAMES_GRID] == pytest.approx([8.7, 11.0, 11.0], abs=0.2)


def test_the_regression_and_the_win_rate_come_together_under_full_rotation(rot):
    assert [round(rot[(1.0, g)]["acc"]["Ridge"], 3) for g in C.GAMES_GRID] == pytest.approx([0.890, 0.957, 0.986], abs=0.006)
    assert [round(rot[(1.0, g)]["acc"]["Siegquote"], 3) for g in C.GAMES_GRID] == pytest.approx([0.866, 0.928, 0.974], abs=0.006)
    assert rot[(1.0, 200)]["acc"]["Ridge"] - rot[(1.0, 200)]["acc"]["Siegquote"] < 0.04
    assert rot[(0.1, 200)]["acc"]["Ridge"] - rot[(0.1, 200)]["acc"]["Siegquote"] > 0.09


def test_intermediate_rotation_and_other_estimators(rot):
    assert [round(rot[(0.3, g)]["acc"]["Ridge"], 3) for g in C.GAMES_GRID] == pytest.approx([0.804, 0.900, 0.960], abs=0.006)
    r = rot[(0.1, 200)]["acc"]
    assert r["Elo-Team"] == pytest.approx(0.757, abs=0.006) and r["TrueSkill-Teams"] == pytest.approx(0.757, abs=0.006) and r["Maximum Likelihood"] == pytest.approx(0.750, abs=0.006)
    assert rot[(0.1, 60)]["acc"]["Maximum Likelihood"] == pytest.approx(0.664, abs=0.006) and rot[(0.1, 60)]["acc"]["Ridge"] == pytest.approx(0.735, abs=0.006)


# --- Experiment 4: Windschatten ---------------------------------------------------------------------------------------------------------------

def test_the_win_rate_lifts_the_weakest_player_next_to_the_best(wind):
    w = wind[0.9]["err"]
    assert (w["Siegquote"], w["Elo-Team"], w["TrueSkill-Teams"], w["Ridge"], w["Maximum Likelihood"]) == pytest.approx((1.60, 1.25, 1.41, 0.52, 0.13), abs=0.04)
    assert wind[0.0]["err"]["Siegquote"] == pytest.approx(0.12, abs=0.03)
    assert [round(wind[c]["err"]["Siegquote"], 2) for c in C.WIND_GRID] == pytest.approx([0.12, 0.78, 1.30, 1.60], abs=0.04)


def test_the_regularisation_pulls_the_weakest_towards_the_partner(wind):
    assert wind[0.9]["err"]["Ridge"] > 3 * wind[0.9]["err"]["Maximum Likelihood"]
    assert wind[0.9]["places"]["Siegquote"] == pytest.approx(4.9, abs=0.3) and wind[0.9]["places"]["Ridge"] == pytest.approx(1.5, abs=0.3)
    assert wind[0.9]["places"]["Maximum Likelihood"] == pytest.approx(0.6, abs=0.2)
    assert wind[0.0]["places"]["Siegquote"] == pytest.approx(0.4, abs=0.15) and max(wind[0.0]["places"][k] for k in ("Ridge", "Maximum Likelihood")) < 0.5


# --- Experiment 5: Regularisierung ------------------------------------------------------------------------------------------------------------

def test_a_moderate_regularisation_is_best_and_too_much_levels_everyone(lam):
    assert [round(lam[v]["rmse"], 3) for v in C.LAM_GRID] == pytest.approx([0.731, 0.611, 0.573, 0.581, 0.638, 0.736], abs=0.008)
    assert min(lam, key=lambda v: lam[v]["rmse"]) == 0.5
    assert [round(lam[v]["acc"], 3) for v in C.LAM_GRID] == pytest.approx([0.750, 0.810, 0.831, 0.828, 0.802, 0.766], abs=0.008)


# --- Experiment 6: der Beste trägt ------------------------------------------------------------------------------------------------------------

def test_a_wrong_aggregate_costs_accuracy(agg):
    s, m = agg[C.AGGREGATES[0]]["acc"], agg[C.AGGREGATES[1]]["acc"]
    assert (s["Siegquote"], s["Elo-Team"], s["Ridge"], s["TrueSkill-Teams"]) == pytest.approx((0.763, 0.877, 0.933, 0.869), abs=0.006)
    assert (m["Siegquote"], m["Elo-Team"], m["Ridge"], m["TrueSkill-Teams"]) == pytest.approx((0.681, 0.751, 0.796, 0.751), abs=0.006)
    assert s["Ridge"] - m["Ridge"] > 0.12


# --- Verdikte nennen die Zahlen aus den Daten -------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(rot, wind, lam, agg):
    rows = list(rot.values())
    assert "0,66 / 0,66 / 0,66" in ev.verdict_fixed(rows)[1] and "Rang 3 von 11" in ev.verdict_fixed(rows)[1]
    assert "**0,66 / 0,91 / 0,96 / 0,99**" in ev.verdict_rotation(rows)[1]
    assert "+1,60 Einheiten" in ev.verdict_wind(list(wind.values()))[1]
    assert "bei λ = 0,5 am kleinsten (0,57)" in ev.verdict_lambda(list(lam.values()))[1]
    assert "von 0,93 auf 0,80" in ev.verdict_aggregate(list(agg.values()))[1]
