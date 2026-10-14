"""Gemeinsame Fixtures: von Hand gebaute Spiele, an denen jede Einheit nachgerechnet wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tei_games import Games  # noqa: E402


def make(n, games, teams=(), s=None, kind=None):
    """Spiele aus Hand. `games`: Liste (Mannschaft A, Mannschaft B, A gewinnt); `s`: wahre Stärken (Standard: alle 0)."""
    a = np.array([list(g[0]) for g in games], dtype=int)
    b = np.array([list(g[1]) for g in games], dtype=int)
    y = np.array([1.0 if g[2] else 0.0 for g in games])
    size = a.shape[1]
    return Games(n, size, tuple(tuple(t) for t in teams), np.zeros(n) if s is None else np.asarray(s, dtype=float), a, b, y,
                 np.zeros(len(games), dtype=int) if kind is None else np.asarray(kind, dtype=int))
