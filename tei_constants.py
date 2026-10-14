"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ (400 Punkte = Faktor 10 der Gewinnchancen), 173,72
BASE = 1500.0                               # Mitte der Elo-Skala in der Anzeige
BETA = 1.702 / math.sqrt(2.0)               # TrueSkill-Leistungsrauschen in θ-Einheiten (Probit passend zum logistischen Modell)
TS_SD0 = 1.0                                # Anfangsunsicherheit von TrueSkill (θ-Einheiten)
LAMBDA_MLE = 1e-6                           # „Maximum Likelihood“: Ridge so klein, dass nur das Niveau fixiert ist

TEAMS_MIN, TEAMS_MAX, DEFAULT_TEAMS = 3, 6, 4            # feste Mannschaften
SIZE_CHOICES, DEFAULT_SIZE = (2, 3, 4), 3                # Spieler je Mannschaft
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 0.6, 0.05, 0.3
GAMES_MIN, GAMES_MAX, GAMES_STEP, DEFAULT_GAMES = 20, 1500, 10, 200
Q_MIN, Q_MAX, Q_STEP, DEFAULT_Q = 0.0, 1.0, 0.05, 0.1   # Anteil der Spiele mit neu gewürfelter Aufstellung
WIND_MIN, WIND_MAX, WIND_STEP, DEFAULT_WIND = 0.0, 0.95, 0.05, 0.6    # Anteil der Spiele, in denen der Schwächste neben dem Besten steht
AGGREGATES = ("Summe der Stärken", "Der Beste trägt")
DEFAULT_AGGREGATE = AGGREGATES[0]
K_MIN, K_MAX, DEFAULT_K = 1, 100, 17                      # Elo-Team, Punkte je Spieler (17 Punkte ≈ 0,1 θ)
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.0, 10.0, 0.1, 1.0   # Ridge (Prior N(0, 1/λ)), 0 heißt: kaum Regularisierung
SEED_MAX, DEFAULT_SEED = 999999, 7

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_TEAMS, EXP_SIZE, EXP_DELTA = 4, 3, 0.3                # 12 Spieler, Mannschaften zu 3
Q_GRID = (0.0, 0.1, 0.3, 1.0)
GAMES_GRID = (60, 200, 600)
WIND_GRID = (0.0, 0.3, 0.6, 0.9)
LAM_GRID = (0.0, 0.1, 0.5, 1.0, 3.0, 10.0)
EXP_Q, EXP_GAMES = 0.1, 200                                # Standardlage der Experimente 4 und 5
WIND_GAMES = 300
AGG_Q, AGG_GAMES = 0.3, 300
EXP_K = 17.0
EXP_LAM = 1.0
EST_NAMES = ("Siegquote", "Elo-Team", "Ridge", "TrueSkill-Teams")

COLOR_RATE = "#9aa0a6"
COLOR_ELO = "#2F6B65"
COLOR_RIDGE = "#d98c1f"
COLOR_TS = "#8e6bb0"
COLOR_TRUTH = "#e67e22"
COLORS = {"Siegquote": COLOR_RATE, "Elo-Team": COLOR_ELO, "Ridge": COLOR_RIDGE, "TrueSkill-Teams": COLOR_TS}

_BASE = {"teams": DEFAULT_TEAMS, "size": DEFAULT_SIZE, "delta": DEFAULT_DELTA, "games": DEFAULT_GAMES, "q": DEFAULT_Q, "use_wind": False, "wind": DEFAULT_WIND,
         "aggregate": DEFAULT_AGGREGATE, "k": DEFAULT_K, "lam": DEFAULT_LAM, "seed": DEFAULT_SEED}
PRESETS = {
    "Immer dieselben Trios": {**_BASE, "q": 0.0},
    "Teilweise Rotation": dict(_BASE),
    "Vollständige Rotation": {**_BASE, "q": 1.0},
    "Windschatten": {**_BASE, "q": 1.0, "use_wind": True, "wind": 0.9, "games": 300},
}
PRESET_HELP = {
    "Immer dieselben Trios": "Vier feste Mannschaften zu drei Spielern: Innerhalb einer Mannschaft sind die Spieler nicht zu trennen, mehr Spiele ändern daran nichts.",
    "Teilweise Rotation": "In einem von zehn Spielen werden alle sechs Spieler neu gewürfelt: Das genügt, um die Einzelnen langsam zu trennen; die Siegquote bleibt hängen.",
    "Vollständige Rotation": "Jedes Spiel mit neuer Aufstellung: Die Mitspieler mitteln sich heraus, und auch die Siegquote ordnet recht gut.",
    "Windschatten": "Der Schwächste steht in 90 % der Spiele neben dem Besten: Die Siegquote hebt ihn um mehrere Plätze, die Regression weniger.",
}
