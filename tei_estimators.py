"""Schätzer der Einzelstärken aus Mannschaftsergebnissen, alle in θ-Einheiten (außer der Siegquote):

  Siegquote       Anteil der Spiele, die ein Spieler mit seiner Mannschaft gewonnen hat (naiv: der Mitspieler zählt mit).
  Elo-Team        Wertung einer Mannschaft = Summe der Wertungen; nach jedem Spiel erhält jeder Spieler k (Ergebnis − Erwartung) (Gewinner plus, Verlierer minus).
  Ridge           logistische Regression der Ausgänge auf die Designmatrix (+1 Mannschaft A, −1 Mannschaft B) mit Prior N(0, 1/λ) auf jeder Stärke (MAP); Newton-Verfahren.
  TrueSkill-Teams Erwartungsabgleich in einem Durchlauf in Spielreihenfolge: Mannschaftsstärke = Summe, c² = 2 m β² + Σσ² (Herbrich, Minka, Graepel 2006)."""

import math

import numpy as np

import tei_constants as C


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def design_matrix(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """(G, n): +1 für die Spieler der Mannschaft A, −1 für die der Mannschaft B."""
    X = np.zeros((len(a), n))
    rows = np.arange(len(a))[:, None]
    X[rows, a] += 1.0
    X[rows, b] -= 1.0
    return X


def win_rate(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    wins, cnt = np.zeros(n), np.zeros(n)
    for ta, tb, r in zip(a, b, y):
        wins[ta] += r
        cnt[ta] += 1
        wins[tb] += 1.0 - r
        cnt[tb] += 1
    return np.where(cnt > 0, wins / np.maximum(cnt, 1), 0.5)


def elo_team(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int, k: float) -> np.ndarray:
    """`k` in θ-Einheiten je Spieler."""
    r = np.zeros(n)
    for ta, tb, res in zip(a, b, y):
        d = k * (res - float(sig(r[ta].sum() - r[tb].sum())))
        r[ta] += d
        r[tb] -= d
    return r


def ridge_logit(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 80, tol: float = 1e-10) -> np.ndarray:
    """MAP: maximiert Σ ln σ(± Xθ) − λ/2 |θ|²; ohne Regularisierung (λ ≈ 0) ist nur das Niveau fixiert (ein winziger Ridge hält das System lösbar)."""
    n = X.shape[1]
    lam = max(float(lam), C.LAMBDA_MLE)
    th = np.zeros(n)
    for _ in range(iters):
        p = sig(X @ th)
        g = X.T @ (y - p) - lam * th
        H = (X * (p * (1.0 - p))[:, None]).T @ X + lam * np.eye(n)
        step = np.linalg.solve(H, g)
        th += step
        if np.abs(step).max() < tol:
            break
    return th


def trueskill_team(a: np.ndarray, b: np.ndarray, y: np.ndarray, n: int, beta: float = C.BETA, sd0: float = C.TS_SD0) -> tuple:
    """Rückgabe: (µ, σ) je Spieler. Remis gibt es nicht; Mannschaften gleicher Größe."""
    mu, var = np.zeros(n), np.full(n, sd0 * sd0)
    for ta, tb, res in zip(a, b, y):
        w, l = (ta, tb) if res == 1.0 else (tb, ta)
        c2 = 2.0 * len(ta) * beta * beta + var[w].sum() + var[l].sum()
        c = math.sqrt(c2)
        t = (mu[w].sum() - mu[l].sum()) / c
        pdf = math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
        v = pdf / max(0.5 * math.erfc(-t / math.sqrt(2.0)), 1e-300)
        wf = v * (v + t)
        vw, vl = var[w].copy(), var[l].copy()
        mu[w] += vw / c * v
        mu[l] -= vl / c * v
        var[w] = vw * (1.0 - vw / c2 * wf)
        var[l] = vl * (1.0 - vl / c2 * wf)
    return mu, np.sqrt(var)
