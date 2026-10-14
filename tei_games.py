"""Mannschaftsspiele: Aufstellungen (feste Mannschaften, Rotation, Windschatten) und Ausgänge. Zufall nur über SplitMix64 (Generator der Linie).

Spieler 0 … n−1 mit gleichabständigen Stärken s (Generator, Mittel 0, aufsteigend: Spieler 0 ist der Schwächste, Spieler n−1 der Beste). Die festen Mannschaften sind zufällig
zugeordnet (nicht nach Stärke gestapelt). Ein Spiel hat drei Zufallsentscheidungen vorab: Windschatten (mit Wahrscheinlichkeit c steht der Schwächste neben dem Besten, die übrigen
Plätze werden zufällig besetzt), sonst Rotation (mit Wahrscheinlichkeit q wird die Aufstellung aus allen Spielern neu gewürfelt) oder zwei feste Mannschaften. Ausgang: A gewinnt mit
σ(Stärke_A − Stärke_B); die Stärke einer Mannschaft ist die Summe ihrer Spieler, im Szenario „Der Beste trägt“ das m-fache ihres Besten."""

from dataclasses import dataclass

import numpy as np

import tei_constants as C
import turnier_generator as T

_SALT = 0x7E41


@dataclass(frozen=True)
class Games:
    n: int
    size: int
    teams: tuple               # feste Mannschaften (Tupel von Spielernummern)
    s: np.ndarray              # (n,) wahre Stärken
    a: np.ndarray              # (G, m) Mannschaft A je Spiel
    b: np.ndarray              # (G, m) Mannschaft B
    y: np.ndarray              # (G,) 1, wenn A gewinnt
    kind: np.ndarray           # (G,) 0 = feste Mannschaften, 1 = Rotation, 2 = Windschatten


def team_strength(s: np.ndarray, members, aggregate: str) -> float:
    vals = s[list(members)]
    if aggregate == C.AGGREGATES[0]:
        return float(vals.sum())
    if aggregate == C.AGGREGATES[1]:
        return float(vals.max() * len(vals))
    raise ValueError(f"unbekanntes Aggregat: {aggregate}")


def simulate_games(teams: int, size: int, delta: float, games: int, q: float, wind: float, aggregate: str, seed: int) -> Games:
    """Deterministisch aus den Eingaben. `wind` ist der Anteil der Windschatten-Spiele (0 = keine)."""
    if teams < 2 or size < 2:
        raise ValueError("Mindestens zwei Mannschaften zu je zwei Spielern")
    if not (0.0 <= q <= 1.0 and 0.0 <= wind <= 1.0):
        raise ValueError("q und wind liegen in [0, 1]")
    n = teams * size
    s = T.make_strengths(n, delta)
    rng = T.SplitMix64(int(seed) ^ _SALT)
    order = np.argsort(rng.uniform(n), kind="stable")
    fixed = tuple(tuple(int(v) for v in order[k * size:(k + 1) * size]) for k in range(teams))
    width = 3 + n + teams
    u = rng.uniform(games * width).reshape(games, width)
    a = np.zeros((games, size), dtype=int)
    b = np.zeros((games, size), dtype=int)
    y = np.zeros(games)
    kind = np.zeros(games, dtype=int)
    others = [i for i in range(1, n - 1)]
    for g in range(games):
        keys_p = u[g, 3:3 + n]
        keys_t = u[g, 3 + n:]
        if u[g, 0] < wind:
            pick = sorted(others, key=lambda i: (keys_p[i], i))
            team_a = [0, n - 1] + pick[:size - 2]
            team_b = pick[size - 2:2 * size - 2]
            kind[g] = 2
        elif u[g, 1] < q:
            pick = sorted(range(n), key=lambda i: (keys_p[i], i))
            team_a, team_b = pick[:size], pick[size:2 * size]
            kind[g] = 1
        else:
            t_order = sorted(range(teams), key=lambda k: (keys_t[k], k))
            team_a, team_b = list(fixed[t_order[0]]), list(fixed[t_order[1]])
        a[g], b[g] = team_a, team_b
        d = team_strength(s, team_a, aggregate) - team_strength(s, team_b, aggregate)
        y[g] = 1.0 if u[g, 2] < 1.0 / (1.0 + np.exp(-d)) else 0.0
    return Games(n, size, fixed, s, a, b, y, kind)
