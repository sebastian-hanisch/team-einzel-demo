"""Aufstellungen und Ausgänge: feste Mannschaften, Rotation, Windschatten, Aggregat, Determinismus, Randfälle."""

import numpy as np
import pytest

import tei_constants as C
import tei_games as G


def _sim(**kw):
    base = dict(teams=4, size=3, delta=0.3, games=300, q=0.0, wind=0.0, aggregate=C.AGGREGATES[0], seed=1)
    base.update(kw)
    return G.simulate_games(**base)


def test_same_inputs_same_games_and_other_seed_other_games():
    a, b, c = _sim(seed=5), _sim(seed=5), _sim(seed=6)
    assert np.array_equal(a.a, b.a) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.y, c.y)


def test_strengths_come_from_the_generator_with_the_weakest_first():
    g = _sim()
    assert g.n == 12 and g.s.mean() == pytest.approx(0.0, abs=1e-12) and g.s[0] < g.s[-1]
    assert np.allclose(np.diff(g.s), 0.3)


def test_fixed_teams_partition_the_players_and_are_not_stacked_by_strength():
    partitions = []
    for seed in range(20):
        g = _sim(seed=seed, games=1)
        assert sorted(p for t in g.teams for p in t) == list(range(12)) and all(len(t) == 3 for t in g.teams)
        partitions.append(g.teams)
    stacked = tuple(tuple(range(k * 3, (k + 1) * 3)) for k in range(4))
    assert all(p != stacked for p in partitions) and len(set(partitions)) > 10


def test_without_rotation_and_windschatten_every_game_is_between_two_distinct_fixed_teams():
    g = _sim(q=0.0, wind=0.0)
    teams = {frozenset(t) for t in g.teams}
    assert (g.kind == 0).all()
    for ta, tb in zip(g.a, g.b):
        assert frozenset(ta) in teams and frozenset(tb) in teams and frozenset(ta) != frozenset(tb)


def test_with_full_rotation_every_game_has_distinct_players_and_new_lineups():
    g = _sim(q=1.0, wind=0.0, games=200)
    assert (g.kind == 1).all()
    for ta, tb in zip(g.a, g.b):
        assert len(set(ta) | set(tb)) == 6
    teams = {frozenset(t) for t in g.teams}
    assert sum(frozenset(ta) in teams for ta in g.a) < 20                # fast nie eine feste Mannschaft


def test_windschatten_puts_the_weakest_next_to_the_best():
    g = _sim(wind=1.0, q=0.0, games=100)
    assert (g.kind == 2).all()
    for ta, tb in zip(g.a, g.b):
        assert 0 in ta and 11 in ta and 0 not in tb and 11 not in tb and len(set(ta) | set(tb)) == 6


def test_shares_of_game_kinds_follow_the_settings():
    g = _sim(q=0.5, wind=0.2, games=2000, seed=3)
    share = [float((g.kind == k).mean()) for k in (0, 1, 2)]
    assert share[2] == pytest.approx(0.2, abs=0.03) and share[1] == pytest.approx(0.8 * 0.5, abs=0.04) and share[0] == pytest.approx(0.8 * 0.5, abs=0.04)


def test_team_strength_is_the_sum_or_m_times_the_best():
    s = np.array([0.0, 1.0, 3.0, -2.0])
    assert G.team_strength(s, [1, 2], C.AGGREGATES[0]) == 4.0
    assert G.team_strength(s, [1, 2], C.AGGREGATES[1]) == 6.0
    assert G.team_strength(s, [0, 3, 1], C.AGGREGATES[1]) == 3.0
    with pytest.raises(ValueError):
        G.team_strength(s, [1], "Mittel")


def test_the_stronger_team_wins_nearly_always_with_a_huge_gap():
    g = _sim(delta=3.0, q=1.0, games=300)
    d = np.array([g.s[ta].sum() - g.s[tb].sum() for ta, tb in zip(g.a, g.b)])
    assert np.all(g.y[d > 15] == 1.0) and np.all(g.y[d < -15] == 0.0) and (np.abs(d) > 15).mean() > 0.5


def test_outcome_frequency_follows_the_logistic_probability():
    g = _sim(delta=0.3, q=1.0, games=6000, seed=11)
    d = np.array([g.s[ta].sum() - g.s[tb].sum() for ta, tb in zip(g.a, g.b)])
    p = 1.0 / (1.0 + np.exp(-d))
    assert g.y.mean() == pytest.approx(p.mean(), abs=0.015)
    assert np.corrcoef(g.y, p)[0, 1] > 0.3


def test_the_best_carries_changes_the_probabilities():
    g = _sim(q=1.0, games=500, aggregate=C.AGGREGATES[1])
    assert np.isfinite(g.y).all() and set(g.y.tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("kw", [dict(teams=1), dict(size=1), dict(q=1.5), dict(wind=-0.1)])
def test_invalid_inputs_are_rejected(kw):
    with pytest.raises(ValueError):
        _sim(**kw)


def test_smallest_field_runs_with_windschatten():
    g = G.simulate_games(3, 2, 0.3, 50, 0.5, 0.5, C.AGGREGATES[0], 1)
    assert g.n == 6 and g.a.shape == (50, 2) and set(g.kind.tolist()) <= {0, 1, 2}
    g4 = G.simulate_games(6, 4, 0.3, 50, 0.5, 0.5, C.AGGREGATES[0], 1)
    assert g4.n == 24 and g4.a.shape == (50, 4)


def test_smallest_allowed_field_runs_with_every_kind_of_game():
    g = G.simulate_games(2, 2, 0.3, 40, 0.5, 0.5, C.AGGREGATES[0], 3)
    assert g.n == 4 and set(g.kind.tolist()) == {0, 1, 2}
    for ta, tb in zip(g.a, g.b):
        assert len(set(ta) | set(tb)) == 4
