"""Literatur-Referenzwerte (ORACLE-PLAYBOOK §5.1) für die Mannschafts-Demo. Einsicht am 2026-10-17 (Seite per WebFetch gelesen); kein Wert stammt aus dem Code der Demo.

1. Wikipedia „Elo rating system“, https://en.wikipedia.org/wiki/Elo_rating_system, Einleitung: Ein Spieler mit 100 Punkten mehr „is expected to score 64%“, bei 200 Punkten 76 %.
   Abschnitt „Ratings inflation and deflation“, Bildunterschrift: 1800 gegen 1400, Gewinnwahrscheinlichkeit des Stärkeren 0,91 (Sekundärquelle, gerundete Werte, Toleranz 0,005).
   Verwendung: Die Elo-Skala der Demo (SCALE = 400/ln 10 Punkte je θ, 400 Punkte = Faktor 10) und die Elo-Erwartung im Elo-Team-Schätzer. Test: Nach einem Sieg von A gegen B (1 gegen 1)
   beträgt der Wertungsabstand k (θ); der zweite Sieg von A ändert A um k (1 − E) mit der Elo-Erwartung E für diesen Abstand.

Nicht eingesehen bzw. nicht verwertbar (kein Test, siehe Bericht „offen“):
- Herbrich, Minka, Graepel (2006), TrueSkill, NIPS 19, 569–576: PDF heruntergeladen (papers.nips.cc), der Text ließ sich wegen der Schriftkodierung nicht extrahieren; die Formeln der
  Team-Aktualisierung konnten daher nicht von der Quelle abgelesen werden (die README sagt dasselbe). Das Paket `trueskill` (Orakel-Test) bleibt die Vergleichsimplementierung.
- Dokumentation des Pakets `trueskill` (https://trueskill.org/, eingesehen 2026-10-17): Zahlenbeispiele (rate_1vs1 → 29,396 / 20,604, σ 7,171; 1 gegen 2: 33,731 / 16,269, σ 7,317) gelten
  für Standardwerte mit Remis-Wahrscheinlichkeit 10 % und τ = 0,083; die Demo rechnet ohne Remis und ohne Dynamik, die Werte sind daher nicht vergleichbar.
- Hoerl und Kennard (1970), Ridge regression: Zahlenbeispiele nicht eingesehen.
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
