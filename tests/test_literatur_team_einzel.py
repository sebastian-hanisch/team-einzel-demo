"""Literatur-Referenzwerte (ORACLE-PLAYBOOK §5.1) für die Mannschafts-Demo. Einsicht am 2026-10-17 (Seite per WebFetch gelesen); kein Wert stammt aus dem Code der Demo.

1. Wikipedia „Elo rating system“, https://en.wikipedia.org/wiki/Elo_rating_system, Einleitung: Ein Spieler mit 100 Punkten mehr „is expected to score 64%“, bei 200 Punkten 76 %.
   Abschnitt „Ratings inflation and deflation“, Bildunterschrift: 1800 gegen 1400, Gewinnwahrscheinlichkeit des Stärkeren 0,91 (Sekundärquelle, gerundete Werte, Toleranz 0,005).
   Verwendung: Die Elo-Skala der Demo (SCALE = 400/ln 10 Punkte je θ, 400 Punkte = Faktor 10) und die Elo-Erwartung im Elo-Team-Schätzer. Test: Nach einem Sieg von A gegen B (1 gegen 1)
   beträgt der Wertungsabstand k (θ); der zweite Sieg von A ändert A um k (1 − E) mit der Elo-Erwartung E für diesen Abstand.
2. Herbrich, R., Minka, T., Graepel, T. (2006): „TrueSkill: A Bayesian Skill Rating System“, NIPS 19 (PDF papers.nips.cc; die Textschicht der PDF ist unlesbar, die Seiten wurden am 2026-10-17
   als Bilder gelesen). Abschnitt 2 (S. 2): Jeder Spieler hat die Leistung p_i ~ N(s_i, β²), die Leistung der Mannschaft j ist die Summe t_j = Σ_{i ∈ A_j} p_i; ein Sieg von Mannschaft 1 über 2
   ist das Ereignis t₁ > t₂. Abschnitt 3 und Tabelle 1: Erwartungsabgleich für die abgeschnittene Leistungsdifferenz (Funktionen V und W). Die ausgeschriebenen Team-Formeln µ_i += σ_i² v/c
   stehen im Papier nicht; für eine einzelne Partie sind sie aber die exakten Momente der Posteriori des Papier-Modells (Folgerung, nicht Wortlaut), die der Test mit scipy nachrechnet.
3. Hoerl, A. E., Kennard, R. W. (1970), Ridge regression: Biased estimation for nonorthogonal problems, Technometrics 12 (1), 55–67 (Scan stat.cmu.edu, am 2026-10-17 gelesen).
   Gleichung (2.1), S. 57: β* = [X'X + kI]⁻¹ X'Y, k ≥ 0. Die Demo löst die logistische Regression mit Newton-Verfahren; der erste Newton-Schritt vom Nullpunkt (p = ½, Gewicht ¼) ist genau
   (2.1) mit k = 4λ und Y = 4 (y − ½) = ±2 (Folgerung, nicht Wortlaut). Die Quelle enthält keine Zahlenbeispiele für die logistische Fassung.

Nicht eingesehen bzw. nicht verwertbar (kein Test, siehe Bericht „offen“):
- Dokumentation des Pakets `trueskill` (https://trueskill.org/, eingesehen 2026-10-17): Zahlenbeispiele (rate_1vs1 → 29,396 / 20,604, σ 7,171; 1 gegen 2: 33,731 / 16,269, σ 7,317) gelten
  für Standardwerte mit Remis-Wahrscheinlichkeit 10 % und τ = 0,083; die Demo rechnet ohne Remis und ohne Dynamik, die Werte sind daher nicht vergleichbar.
"""

import numpy as np
import pytest

import tei_constants as C
import tei_estimators as E


def _one_on_one(results):
    """Spiele 1 gegen 1: Spieler 0 gegen Spieler 1, results = Liste Ausgänge für Spieler 0."""
    a = np.zeros((len(results), 1), dtype=int)
    b = np.ones((len(results), 1), dtype=int)
    return a, b, np.array(results, dtype=float)


@pytest.mark.parametrize("points,expected,tol", [(100.0, 0.64, 0.005), (200.0, 0.76, 0.005), (400.0, 0.91, 0.005)])
def test_elo_team_expected_score_matches_the_elo_scale(points, expected, tol):
    k = points / C.SCALE                                                   # θ-Einheiten je Spieler; nach dem ersten Sieg: Abstand 2 · k/2 = k = `points` Punkte
    r1 = E.elo_team(*_one_on_one([1.0]), 2, k)
    assert r1[0] - r1[1] == pytest.approx(k, abs=1e-12)
    r2 = E.elo_team(*_one_on_one([1.0, 1.0]), 2, k)
    step = r2[0] - r1[0]                                                   # = k (1 − E)
    assert step == pytest.approx(k * (1.0 - expected), abs=tol * k)


def test_elo_scale_constant_is_400_points_per_factor_ten():
    assert float(E.sig(400.0 / C.SCALE)) == pytest.approx(10.0 / 11.0, abs=1e-12)
    assert float(E.sig(1800.0 / C.SCALE - 1400.0 / C.SCALE)) == pytest.approx(0.91, abs=0.005)


# --- Herbrich, Minka, Graepel 2006, Abschnitt 2: Mannschaftsleistung als Summe -----------------------------------------------------------------

@pytest.mark.parametrize("size,beta,sd0", [(1, 1.0, 1.0), (2, 0.6, 1.0), (3, 1.2, 0.7), (4, 0.9, 1.6)])
def test_one_game_team_update_is_the_exact_posterior_of_the_papers_summed_performance_model(size, beta, sd0):
    """Alle Spieler µ = 0, σ = sd0; Mannschaft A (Spieler 0..size−1) gewinnt. D = t_A − t_B ~ N(0, c²), c² = 2 m β² + 2 m sd0², wird bei 0 abgeschnitten;
    jede Stärke hängt mit Kovarianz σ² an D (jeder Spieler geht einmal in D ein)."""
    truncnorm = pytest.importorskip("scipy.stats").truncnorm
    n = 2 * size
    a = np.arange(size)[None, :]
    b = np.arange(size, n)[None, :]
    mu, sd = E.trueskill_team(a, b, np.array([1.0]), n, beta=beta, sd0=sd0)
    c2 = 2.0 * size * beta ** 2 + 2.0 * size * sd0 ** 2
    c = np.sqrt(c2)
    mean_d = c * float(truncnorm.mean(0.0, np.inf))                        # untere Grenze (0 − 0)/c = 0
    var_d = c2 * float(truncnorm.var(0.0, np.inf))
    k = sd0 ** 2 / c2
    assert mu[:size] == pytest.approx(k * mean_d, abs=1e-9)
    assert mu[size:] == pytest.approx(-k * mean_d, abs=1e-9)
    assert sd ** 2 == pytest.approx(sd0 ** 2 - k * k * c2 + k * k * var_d, abs=1e-9)


# --- Hoerl, Kennard 1970, Gleichung (2.1) ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed,lam", [(1, 0.5), (2, 3.0), (3, 0.05)])
def test_first_newton_step_of_the_ridge_logit_is_the_ridge_estimator_of_hoerl_and_kennard(seed, lam):
    """Vom Nullpunkt aus ist p = ½ und das Gewicht ¼: β₁ = (¼ X'X + λI)⁻¹ X'(y − ½) = [X'X + kI]⁻¹ X'Y mit k = 4λ und Y = 4 (y − ½)."""
    rng = np.random.default_rng(seed)
    n, games, size = 6, 40, 2
    a = np.array([rng.permutation(n)[:size] for _ in range(games)])
    b = np.array([rng.permutation(np.setdiff1d(np.arange(n), row))[:size] for row in a])
    y = (rng.random(games) < 0.6).astype(float)
    X = E.design_matrix(a, b, n)
    k, Y = 4.0 * lam, 4.0 * (y - 0.5)
    hk = np.linalg.solve(X.T @ X + k * np.eye(n), X.T @ Y)
    assert E.ridge_logit(X, y, lam, iters=1) == pytest.approx(hk, abs=1e-10)
