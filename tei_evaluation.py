"""Maße, Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Alle Schätzer sehen dieselben Spiele. Hauptmaß: Anteil richtig geordneter Spielerpaare, Gleichstände des Schätzers zählen halb (0,5 = Zufall); sonst sähe ein Schätzer, der
Spieler derselben Mannschaft gleichsetzt, perfekt aus. Dazu der RMSE nach Angleichen der Skala (die Schätzer haben verschiedene Skalen) und der Fehler eines gewählten Spielers."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import tei_constants as C
import tei_estimators as E
import tei_games as G
from tei_scenario import Settings, make_games

NAN = float("nan")


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace(".", ",").replace("-", "−")


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Maße -------------------------------------------------------------------------------------------------------------------------------------

def pair_accuracy(est, truth) -> float:
    """Anteil der Spielerpaare, die der Schätzer richtig ordnet; ein Gleichstand des Schätzers zählt halb."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    iu = np.triu_indices(len(est), 1)
    se, st = np.sign(est[iu[0]] - est[iu[1]]), np.sign(truth[iu[0]] - truth[iu[1]])
    return float(np.where(se == 0, 0.5, (se == st).astype(float)).mean())


def tie_share(est, tol: float = 1e-9) -> float:
    """Anteil der Spielerpaare mit (praktisch) gleichem Schätzwert."""
    est = np.asarray(est, dtype=float)
    iu = np.triu_indices(len(est), 1)
    return float((np.abs(est[iu[0]] - est[iu[1]]) <= tol).mean())


def align_scale(est, truth) -> float:
    """Faktor, der den zentrierten Schätzer am besten auf die zentrierte Wahrheit abbildet (kleinste Quadrate)."""
    e = np.asarray(est, dtype=float) - np.mean(est)
    t = np.asarray(truth, dtype=float) - np.mean(truth)
    denom = float(e @ e)
    return float(e @ t) / denom if denom > 1e-18 else 0.0


def aligned_rmse(est, truth) -> float:
    e = np.asarray(est, dtype=float) - np.mean(est)
    t = np.asarray(truth, dtype=float) - np.mean(truth)
    return float(np.sqrt(np.mean((align_scale(est, truth) * e - t) ** 2)))


def player_error(est, truth, player: int) -> float:
    """Geschätzte minus wahre Stärke eines Spielers (zentriert, Skala angeglichen), in θ-Einheiten."""
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    scale = align_scale(est, truth)
    return float(scale * (est[player] - est.mean()) - (truth[player] - truth.mean()))


def places_above_last(est, player: int = 0) -> int:
    """Wie viele Spieler der Schätzer unter den gewählten Spieler setzt (0 = er ist Letzter)."""
    est = np.asarray(est, dtype=float)
    return int((est < est[player]).sum())


def design_rank(games: G.Games) -> int:
    """Rang der Designmatrix: wie viele Richtungen im Raum der Stärken die Spiele überhaupt unterscheiden (höchstens n − 1)."""
    return int(np.linalg.matrix_rank(E.design_matrix(games.a, games.b, games.n)))


def estimate_all(g: G.Games, k_theta: float, lam: float) -> dict:
    """Alle Schätzer (Siegquote in [0, 1], die übrigen in θ-Einheiten); „Ridge“ mit λ, „Maximum Likelihood“ mit sehr kleinem λ."""
    X = E.design_matrix(g.a, g.b, g.n)
    mu, sd = E.trueskill_team(g.a, g.b, g.y, g.n)
    return {"Siegquote": E.win_rate(g.a, g.b, g.y, g.n), "Elo-Team": E.elo_team(g.a, g.b, g.y, g.n, k_theta), "Ridge": E.ridge_logit(X, g.y, lam),
            "TrueSkill-Teams": mu, "ts_sd": sd, "Maximum Likelihood": E.ridge_logit(X, g.y, C.LAMBDA_MLE)}


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    games: G.Games
    estimates: dict            # Name -> Schätzwerte je Spieler
    accuracy: dict             # Name -> Paartreffer
    rmse: dict                 # Name -> RMSE nach Angleichen der Skala
    ties: dict                 # Name -> Anteil der Gleichstände
    rank: int                  # Rang der Designmatrix
    weak_error: dict           # Name -> Fehler des Schwächsten
    weak_places: dict          # Name -> Spieler unter dem Schwächsten
    wind_share: float          # Anteil der Windschatten-Spiele
    kind_share: tuple          # Anteile (feste Mannschaften, Rotation, Windschatten)


def analyse(s: Settings) -> Analysis:
    g = make_games(s)
    est = estimate_all(g, s.k_theta, s.lam)
    names = (*C.EST_NAMES, "Maximum Likelihood")
    shares = tuple(float((g.kind == k).mean()) for k in (0, 1, 2))
    return Analysis(s, g, est, {k: pair_accuracy(est[k], g.s) for k in names}, {k: aligned_rmse(est[k], g.s) for k in names}, {k: tie_share(est[k]) for k in names},
                    design_rank(g), {k: player_error(est[k], g.s, 0) for k in names}, {k: places_above_last(est[k], 0) for k in names}, shares[2], shares)


def table(a: Analysis) -> list:
    """Tabelle der Spieler: wahre Stärke und Schätzungen in Elo-Punkten um 1500, die Skala je Schätzer an die wahre Stärke angeglichen."""
    g = a.games
    rows = []
    for p in range(g.n):
        row = {"Spieler": p + 1, "Mannschaft": 1 + next(k for k, t in enumerate(g.teams) if p in t), "wahre Stärke": round(C.BASE + C.SCALE * float(g.s[p] - g.s.mean()))}
        for name in C.EST_NAMES:
            e = a.estimates[name]
            row[name] = round(C.BASE + C.SCALE * align_scale(e, g.s) * float(e[p] - e.mean()))
        row["± TrueSkill"] = round(C.SCALE * float(a.estimates["ts_sd"][p]))
        rows.append(row)
    return rows


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _games(teams: int, size: int, delta: float, games: int, q: float, wind: float, aggregate: str, seed: int) -> G.Games:
    return G.simulate_games(teams, size, delta, games, q, wind, aggregate, seed)


@functools.lru_cache(maxsize=4096)
def _estimates(teams: int, size: int, games: int, q: float, wind: float, aggregate: str, seed: int, lam: float) -> dict:
    g = _games(teams, size, C.EXP_DELTA, games, q, wind, aggregate, seed)
    return estimate_all(g, C.EXP_K / C.SCALE, lam)


def _run(games: int, q: float, wind: float = 0.0, aggregate: str = C.DEFAULT_AGGREGATE, lam: float = C.EXP_LAM):
    """Über die festen Läufe: Liste von (Spiele, Schätzer)."""
    out = []
    for sd in C.SWEEP_SEEDS:
        g = _games(C.EXP_TEAMS, C.EXP_SIZE, C.EXP_DELTA, games, q, wind, aggregate, sd)
        out.append((g, _estimates(C.EXP_TEAMS, C.EXP_SIZE, games, q, wind, aggregate, sd, lam)))
    return out


def _accuracy_row(runs) -> dict:
    names = (*C.EST_NAMES, "Maximum Likelihood")
    return {k: _mean([pair_accuracy(est[k], g.s) for g, est in runs]) for k in names}


def sweep_rotation(qs=C.Q_GRID, games=C.GAMES_GRID) -> list:
    """Experimente 1 bis 3: Paartreffer je Rotation q und Spielzahl, alle Schätzer; dazu Rang der Designmatrix und Anteil der Gleichstände der Siegquote."""
    rows = []
    for q in qs:
        for gcount in games:
            runs = _run(gcount, q)
            rows.append({"q": q, "games": gcount, "acc": _accuracy_row(runs), "rank": _mean([float(design_rank(g)) for g, _ in runs]),
                         "rmse": {k: _mean([aligned_rmse(est[k], g.s) for g, est in runs]) for k in C.EST_NAMES},
                         "ties_rate": _mean([tie_share(est["Siegquote"]) for g, est in runs])})
    return rows


def sweep_wind(winds=C.WIND_GRID, games=C.WIND_GAMES) -> list:
    """Experiment 4: Der Schwächste steht in einem Anteil c der Spiele neben dem Besten; Fehler seiner Einzelstärke und die Plätze, die er zu hoch liegt (sonst Rotation q = 1)."""
    rows = []
    names = (*C.EST_NAMES, "Maximum Likelihood")
    for c in winds:
        runs = _run(games, 1.0, wind=c)
        rows.append({"c": c, "err": {k: _mean([player_error(est[k], g.s, 0) for g, est in runs]) for k in names},
                     "places": {k: _mean([float(places_above_last(est[k], 0)) for g, est in runs]) for k in names}})
    return rows


def sweep_lambda(lams=C.LAM_GRID, q=C.EXP_Q, games=C.EXP_GAMES) -> list:
    """Experiment 5: Ridge-λ bei teilweiser Rotation: RMSE und Paartreffer."""
    rows = []
    for lam in lams:
        runs = _run(games, q, lam=lam)
        rows.append({"lam": lam, "rmse": _mean([aligned_rmse(est["Ridge"], g.s) for g, est in runs]), "acc": _mean([pair_accuracy(est["Ridge"], g.s) for g, est in runs])})
    return rows


def sweep_aggregate(q=C.AGG_Q, games=C.AGG_GAMES) -> list:
    """Experiment 6: Wahrheit „Summe“ gegen „Der Beste trägt“, die Schätzer additiv."""
    rows = []
    for agg in C.AGGREGATES:
        runs = _run(games, q, aggregate=agg)
        rows.append({"aggregate": agg, "acc": _accuracy_row(runs)})
    return rows


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis):
    s = a.settings
    best = max(C.EST_NAMES, key=lambda k: a.accuracy[k])
    text = (f"Paartreffer (0,5 = Zufall, Gleichstände zählen halb): " + ", ".join(f"{k} {de(a.accuracy[k])}" for k in C.EST_NAMES) + f"; am besten **{best}**. "
            f"Die Designmatrix hat Rang {a.rank} von höchstens {s.n - 1}.")
    if a.rank < s.n - 1:
        text += f" **{s.n - 1 - a.rank} Richtungen sind aus den Spielen nicht zu schätzen:** Spieler derselben Mannschaft sind nicht zu trennen."
    if s.use_wind and s.wind > 0:
        text += (f" Der Schwächste liegt bei der Siegquote um {signed(a.weak_error['Siegquote'])} Einheiten zu hoch ({a.weak_places['Siegquote']} Spieler unter ihm), bei der Regression "
                 f"um {signed(a.weak_error['Ridge'])} ({a.weak_places['Ridge']}).")
    return "info", text


def _rot(rows, q, games):
    return next(r for r in rows if r["q"] == q and r["games"] == games)


def verdict_fixed(rows):
    vals = [_rot(rows, 0.0, g)["acc"]["Ridge"] for g in C.GAMES_GRID]
    r = _rot(rows, 0.0, C.GAMES_GRID[-1])
    return "warning", (f"Mit festen Mannschaften liegt der Paartreffer der Regression bei {de(vals[0])} / {de(vals[1])} / {de(vals[2])} nach {C.GAMES_GRID[0]} / {C.GAMES_GRID[1]} / "
                       f"{C.GAMES_GRID[2]} Spielen: **mehr Spiele ändern daran nichts**. Die Designmatrix hat Rang {de(r['rank'], 0)} von {C.EXP_TEAMS * C.EXP_SIZE - 1}: nur die "
                       "Unterschiede zwischen den Mannschaften sind schätzbar, innerhalb der Mannschaft ist die Reihenfolge Zufall.")


def verdict_rotation(rows):
    a = [_rot(rows, q, C.GAMES_GRID[-1])["acc"]["Ridge"] for q in C.Q_GRID]
    return "info", (f"Nach {C.GAMES_GRID[-1]} Spielen liegt der Paartreffer der Regression bei Rotation q = {de(C.Q_GRID[0], 1)} / {de(C.Q_GRID[1], 1)} / {de(C.Q_GRID[2], 1)} / "
                    f"{de(C.Q_GRID[3], 1)} bei **{de(a[0])} / {de(a[1])} / {de(a[2])} / {de(a[3])}**. **Schon jedes zehnte Spiel mit neuer Aufstellung** trennt die Spieler, "
                    "aber langsam: Die Information über Unterschiede innerhalb der Mannschaften steckt nur in den rotierenden Spielen.")


def verdict_methods(rows):
    low, full = _rot(rows, C.Q_GRID[1], C.GAMES_GRID[1]), _rot(rows, C.Q_GRID[-1], C.GAMES_GRID[1])
    return "info", (f"Bei teilweiser Rotation (q = {de(C.Q_GRID[1], 1)}, {C.GAMES_GRID[1]} Spiele) liegt die **Siegquote** bei {de(low['acc']['Siegquote'])}, Elo-Team bei "
                    f"{de(low['acc']['Elo-Team'])}, TrueSkill-Teams bei {de(low['acc']['TrueSkill-Teams'])}, die **Regression** bei {de(low['acc']['Ridge'])}: Der Mitspieler zählt "
                    f"bei der Siegquote mit. Bei vollständiger Rotation rücken sie zusammen ({de(full['acc']['Siegquote'])} gegen {de(full['acc']['Ridge'])}): Die Mitspieler "
                    "mitteln sich heraus.")


def verdict_wind(rows):
    top = rows[-1]
    return "warning", (f"Steht der Schwächste in {de(100 * top['c'], 0)} % der Spiele neben dem Besten, liegt er bei der **Siegquote** um {signed(top['err']['Siegquote'])} Einheiten zu hoch "
                       f"({de(top['places']['Siegquote'], 1)} Spieler unter ihm), bei Elo-Team {signed(top['err']['Elo-Team'])}, bei TrueSkill-Teams {signed(top['err']['TrueSkill-Teams'])}, "
                       f"bei der **Regression** {signed(top['err']['Ridge'])} und bei der Regression ohne Regularisierung {signed(top['err']['Maximum Likelihood'])}. Die Regularisierung "
                       "zieht ihn bei dieser Spielzahl zum Mitspieler hin (mit 3000 Spielen ist der Effekt verschwunden); ohne sie bleibt nur das Rauschen.")


def verdict_lambda(rows):
    best = min(rows, key=lambda r: r["rmse"])
    first, last = rows[0], rows[-1]
    return "info", (f"Bei teilweiser Rotation ({de(C.EXP_Q, 1)}, {C.EXP_GAMES} Spiele) ist der Fehler bei λ = {de(best['lam'], 1)} am kleinsten ({de(best['rmse'])}), ohne "
                    f"Regularisierung {de(first['rmse'])} und bei λ = {de(last['lam'], 0)} {de(last['rmse'])}: zu wenig lässt die Mitspieler-Verwechslung stehen, zu viel macht alle "
                    "gleich. Es gibt kein für alle Lagen bestes λ, nur eines je Lage (Experiment 4).")


def verdict_aggregate(rows):
    s, m = rows[0]["acc"], rows[1]["acc"]
    return "warning", (f"Gilt „Der Beste trägt“ statt der Summe, fällt der Paartreffer der Regression von {de(s['Ridge'])} auf {de(m['Ridge'])}, der von TrueSkill-Teams von "
                       f"{de(s['TrueSkill-Teams'])} auf {de(m['TrueSkill-Teams'])}, der Siegquote von {de(s['Siegquote'])} auf {de(m['Siegquote'])}. Der additive Ansatz verteilt "
                       "den Erfolg auf alle Spieler der Mannschaft, auch auf die, die nichts beitragen.")
