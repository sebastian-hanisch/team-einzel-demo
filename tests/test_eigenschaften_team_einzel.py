"""Metamorphe Eigenschaftstests für die Schätzer der Einzelstärken aus Mannschaftsergebnissen (ORACLE-PLAYBOOK §5.2): Man verändert die Eingabe auf eine Weise, deren Wirkung
auf die Ausgabe bekannt ist, und prüft diese Beziehung.

Belegt sind die Aussagen durch das Modell (siehe tei_estimators.py und README „Modell“): A gewinnt mit σ(Stärke A − Stärke B), Mannschaften gleicher Größe; die Designmatrix hat die
Einträge +1 (Mannschaft A) und −1 (Mannschaft B), jede Zeile summiert also zu 0.

Eigenschaften:
  1. Umnummerieren der Spieler permutiert die Schätzungen aller vier Schätzer (und des Maximum-Likelihood-Vergleichs).
  2. Gemeinsame Verschiebung aller Einzelstärken ändert die Mannschaftsdifferenz nicht (bei gleich großen Mannschaften, beide Aggregate); das Niveau der Ridge-Schätzung liegt bei Σθ = 0.
  3. Vertauschen der Mannschaften (A ↔ B, Ausgang umgekehrt) negiert die Designmatrix und lässt alle Schätzungen gleich; Spiegeln aller Ausgänge (ohne Vertauschen) negiert die
     Stärken von Ridge, Elo-Team und TrueSkill-Teams und macht aus der Siegquote 1 − Siegquote.
  4. Rang der Designmatrix bei festen Mannschaften von Hand (Mannschaften − Zusammenhangskomponenten des Spielgraphen), zufällige Spielgraphen gegen eine Union-Find-Rechnung.
  5. Regularisierung: λ → ∞ liefert Gleichheit der Schätzungen (θ ≈ Xᵀ(y − ½)/λ → 0), die Norm der Schätzung fällt mit wachsendem λ.
  6. Die Reihenfolge der Spiele ändert Siegquote und Ridge nicht (Stapelschätzer).
  7. Feste Mannschaften (keine Rotation): Teamkameraden sind bei jedem Schätzer nicht zu trennen (README „Kernfrage“).
  8. Randfälle: ein Spiel (Stationarität von Hand), keine Spiele.
"""

import numpy as np
import pytest

import tei_constants as C
import tei_estimators as E
import tei_evaluation as ev
import tei_games as G
from conftest import make

N_INST = 30


def _games(k, **kw):
    """k-te Zufallsinstanz mit festem Seed: Mannschaftszahl, -größe, Rotation und Windschatten wechseln."""
    rng = np.random.default_rng(2000 + k)
    teams = int(rng.integers(3, 6))
    size = int(rng.choice([2, 3]))
    n_games = int(rng.integers(40, 120))
    q = float(rng.choice([0.3, 0.6, 1.0]))
    wind = float(rng.choice([0.0, 0.0, 0.5]))
    agg = C.AGGREGATES[k % 2]
    args = dict(teams=teams, size=size, delta=0.3, games=n_games, q=q, wind=wind, aggregate=agg, seed=300 + k)
    args.update(kw)
    return G.simulate_games(args["teams"], args["size"], args["delta"], args["games"], args["q"], args["wind"], args["aggregate"], args["seed"])


def _with(g, a=None, b=None, y=None):
    return G.Games(g.n, g.size, g.teams, g.s, g.a if a is None else a, g.b if b is None else b, g.y if y is None else y, g.kind)


def _all(g, lam=0.7, k=0.08):
    X = E.design_matrix(g.a, g.b, g.n)
    mu, sd = E.trueskill_team(g.a, g.b, g.y, g.n)
    return {"Siegquote": E.win_rate(g.a, g.b, g.y, g.n), "Elo": E.elo_team(g.a, g.b, g.y, g.n, k), "Ridge": E.ridge_logit(X, g.y, lam),
            "MLE": E.ridge_logit(X, g.y, C.LAMBDA_MLE), "TS": mu, "TS_sd": sd}


# --- 1. Umnummerieren ----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_relabeling_players_permutes_every_estimate(k):
    g = _games(k)
    perm = np.random.default_rng(k).permutation(g.n)
    g2 = _with(g, a=perm[g.a], b=perm[g.b])
    e1, e2 = _all(g), _all(g2)
    for name in e1:
        assert e2[name][perm] == pytest.approx(e1[name], abs=1e-7), name


# --- 2. Verschiebung der Einzelstärken, Niveau --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_common_shift_of_all_strengths_does_not_change_team_differences(k):
    g = _games(k)
    X = E.design_matrix(g.a, g.b, g.n)
    assert np.abs(X.sum(axis=1)).max() == 0.0                              # jede Zeile summiert zu 0 (gleich große Mannschaften)
    rng = np.random.default_rng(k)
    th = rng.normal(0.0, 1.0, g.n)
    c = float(rng.uniform(-4.0, 4.0))
    assert X @ (th + c) == pytest.approx(X @ th, abs=1e-9)
    for agg in C.AGGREGATES:                                               # auch die Teamstärke der Generatoren: Differenz bleibt (bei gleicher Größe auch für „Der Beste trägt“)
        d0 = [G.team_strength(th, a, agg) - G.team_strength(th, b, agg) for a, b in zip(g.a, g.b)]
        d1 = [G.team_strength(th + c, a, agg) - G.team_strength(th + c, b, agg) for a, b in zip(g.a, g.b)]
        assert d1 == pytest.approx(d0, abs=1e-9)


@pytest.mark.parametrize("k", range(N_INST))
def test_ridge_level_is_pinned_at_zero_and_elo_is_zero_sum(k):
    g = _games(k)
    X = E.design_matrix(g.a, g.b, g.n)
    lam = [0.05, 0.5, 2.0][k % 3]
    # Σ_j (Gradient_j) = Σ_Spiele (y − p) · (Zeilensumme = 0) − λ Σθ = 0  ⇒  Σθ = 0
    assert abs(E.ridge_logit(X, g.y, lam).sum()) < 1e-8
    assert abs(E.elo_team(g.a, g.b, g.y, g.n, 0.1).sum()) < 1e-9


# --- 3. Mannschaften vertauschen, Ausgänge spiegeln -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_swapping_the_teams_negates_the_design_and_leaves_estimates_unchanged(k):
    g = _games(k)
    g2 = _with(g, a=g.b, b=g.a, y=1.0 - g.y)
    assert E.design_matrix(g2.a, g2.b, g2.n) == pytest.approx(-E.design_matrix(g.a, g.b, g.n), abs=0)
    e1, e2 = _all(g), _all(g2)
    for name in e1:
        assert e2[name] == pytest.approx(e1[name], abs=1e-8), name


@pytest.mark.parametrize("k", range(N_INST))
def test_mirroring_all_outcomes_negates_the_strengths(k):
    g = _games(k)
    g2 = _with(g, y=1.0 - g.y)
    e1, e2 = _all(g), _all(g2)
    assert e2["Ridge"] == pytest.approx(-e1["Ridge"], abs=1e-8)
    assert e2["MLE"] == pytest.approx(-e1["MLE"], abs=1e-6)
    assert e2["Elo"] == pytest.approx(-e1["Elo"], abs=1e-9)
    assert e2["TS"] == pytest.approx(-e1["TS"], abs=1e-9)
    assert e2["TS_sd"] == pytest.approx(e1["TS_sd"], abs=1e-9)
    assert e2["Siegquote"] == pytest.approx(1.0 - e1["Siegquote"], abs=1e-12)


# --- 4. Rang der Designmatrix ----------------------------------------------------------------------------------------------------------------------

def test_rank_with_fixed_trios_by_hand():
    """Vier Trios {0,1,2}, {3,4,5}, {6,7,8}, {9,10,11}. Die drei Spalten eines Trios sind in jeder Zeile gleich (alle +1, alle −1 oder 0): höchstens 4 verschiedene Spalten, und weil jede
    Zeile zu 0 summiert, ist der Rang höchstens 3. Treffen sich alle vier Trios (Kette T1–T2, T3–T4, T1–T3), ist er 3, wie in der README (3 von 11)."""
    t = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    connected = make(12, [(t[0], t[1], True), (t[2], t[3], False), (t[0], t[2], True), (t[1], t[3], True)], teams=t)
    assert ev.design_rank(connected) == 3
    two_pairs = make(12, [(t[0], t[1], True), (t[0], t[1], False), (t[2], t[3], True)], teams=t)
    assert ev.design_rank(two_pairs) == 2                                   # zwei getrennte Paare: 4 − 2 Komponenten
    three_only = make(12, [(t[0], t[1], True), (t[1], t[2], False), (t[0], t[2], True)], teams=t)
    assert ev.design_rank(three_only) == 2                                  # Trio 4 spielt nie: 3 Trios verbunden = 2
    assert ev.design_rank(make(12, [(t[0], t[1], True)], teams=t)) == 1


def _components(k, edges):
    parent = list(range(k))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for a, b in edges:
        parent[find(a)] = find(b)
    touched = {v for e in edges for v in e}
    return len(touched) - len({find(v) for v in touched})                   # Mannschaften im Spielgraphen − Komponenten


@pytest.mark.parametrize("k", range(N_INST))
def test_rank_of_random_team_graphs_is_teams_minus_components(k):
    rng = np.random.default_rng(55 + k)
    n_teams, size = int(rng.integers(3, 7)), int(rng.integers(2, 5))
    teams = [tuple(range(i * size, (i + 1) * size)) for i in range(n_teams)]
    edges = []
    for _ in range(int(rng.integers(1, 9))):
        a, b = rng.choice(n_teams, 2, replace=False)
        edges.append((int(a), int(b)))
    g = make(n_teams * size, [(teams[a], teams[b], bool(rng.random() < 0.5)) for a, b in edges], teams=teams)
    assert ev.design_rank(g) == _components(n_teams, edges)


@pytest.mark.parametrize("k", range(10))
def test_rank_is_at_most_n_minus_one_and_fixed_teams_give_teams_minus_one(k):
    g = _games(k)
    assert ev.design_rank(g) <= g.n - 1
    fixed = G.simulate_games(4, 3, 0.3, 300, 0.0, 0.0, C.AGGREGATES[0], 700 + k)
    assert ev.design_rank(fixed) == 3


# --- 5. Regularisierung ----------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_large_lambda_levels_all_estimates_and_follows_the_first_order_direction(k):
    g = _games(k)
    X = E.design_matrix(g.a, g.b, g.n)
    lam = 1e6
    th = E.ridge_logit(X, g.y, lam)
    assert np.abs(th).max() < len(g.y) / (2 * lam) + 1e-9                  # |Xᵀ(y − ½)| ≤ G/2
    first = X.T @ (g.y - 0.5) / lam                                       # Stationarität erster Ordnung: θ ≈ Xᵀ(y − ½)/λ
    assert th == pytest.approx(first, rel=1e-3, abs=1e-9)
    assert np.ptp(th) < 1e-3                                               # alle Schätzungen praktisch gleich (Prior: alle 0)


@pytest.mark.parametrize("k", range(N_INST))
def test_norm_of_the_ridge_estimate_falls_with_lambda(k):
    g = _games(k)
    X = E.design_matrix(g.a, g.b, g.n)
    norms = [np.linalg.norm(E.ridge_logit(X, g.y, lam)) for lam in (0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0)]
    assert all(b <= a + 1e-9 for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 0.5 * norms[0]


# --- 6. Reihenfolge der Spiele ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_game_order_does_not_matter_for_win_rate_and_ridge(k):
    g = _games(k)
    order = np.random.default_rng(900 + k).permutation(len(g.y))
    g2 = _with(g, a=g.a[order], b=g.b[order], y=g.y[order])
    e1, e2 = _all(g), _all(g2)
    assert e2["Siegquote"] == pytest.approx(e1["Siegquote"], abs=1e-12)
    assert e2["Ridge"] == pytest.approx(e1["Ridge"], abs=1e-7)
    assert e2["MLE"] == pytest.approx(e1["MLE"], abs=1e-4)


# --- 7. Feste Mannschaften: Teamkameraden nicht zu trennen -------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(10))
def test_teammates_of_fixed_teams_get_identical_estimates(k):
    teams, size = [(3, 2), (4, 3), (4, 2)][k % 3]
    g = G.simulate_games(teams, size, 0.3, 120, 0.0, 0.0, C.AGGREGATES[0], 800 + k)
    assert set(g.kind.tolist()) == {0}
    e = _all(g)
    for name in ("Siegquote", "Elo", "Ridge", "MLE", "TS", "TS_sd"):
        for team in g.teams:
            vals = e[name][list(team)]
            assert np.ptp(vals) < 1e-7, (name, team)
    assert ev.tie_share(e["Ridge"], 1e-7) >= (size - 1) / (g.n - 1) - 1e-9    # mindestens die Paare innerhalb der Mannschaften sind Gleichstände


# --- 8. Randfälle ------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.3, 1.0, 4.0])
def test_one_game_stationarity_by_hand(lam):
    """Ein Spiel 2 gegen 2 (A gewinnt): Gradient 0 ⇒ θ_A = (1 − σ(4 θ_A))/λ, θ_B = −θ_A (Xθ = 4 θ_A)."""
    g = make(4, [((0, 1), (2, 3), True)])
    X = E.design_matrix(g.a, g.b, g.n)
    th = E.ridge_logit(X, g.y, lam)
    assert th[0] == pytest.approx(th[1], abs=1e-9) and th[2] == pytest.approx(-th[0], abs=1e-9) and th[0] > 0
    assert th[0] == pytest.approx((1.0 - float(E.sig(4.0 * th[0]))) / lam, abs=1e-8)


def test_no_games():
    X = np.zeros((0, 5))
    assert E.ridge_logit(X, np.zeros(0), 1.0) == pytest.approx(np.zeros(5), abs=1e-12)
    e = np.zeros((0, 2), dtype=int)
    assert E.win_rate(e, e, np.zeros(0), 4) == pytest.approx(np.full(4, 0.5))
    assert E.elo_team(e, e, np.zeros(0), 4, 0.1) == pytest.approx(np.zeros(4))
