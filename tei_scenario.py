"""Szenario: Reglerwerte (Settings) und die daraus erzeugten Spiele."""

from dataclasses import dataclass

import tei_constants as C
import tei_games as G


@dataclass(frozen=True)
class Settings:
    teams: int = C.DEFAULT_TEAMS
    size: int = C.DEFAULT_SIZE
    delta: float = C.DEFAULT_DELTA
    games: int = C.DEFAULT_GAMES
    q: float = C.DEFAULT_Q
    use_wind: bool = False
    wind: float = C.DEFAULT_WIND              # nur mit use_wind
    aggregate: str = C.DEFAULT_AGGREGATE
    k: float = C.DEFAULT_K                    # Elo-Team, Punkte je Spieler
    lam: float = C.DEFAULT_LAM                # Ridge
    seed: int = C.DEFAULT_SEED

    @property
    def n(self) -> int:
        return self.teams * self.size

    @property
    def wind_eff(self) -> float:
        return self.wind if self.use_wind else 0.0

    @property
    def k_theta(self) -> float:
        return self.k / C.SCALE


def make_games(s: Settings) -> G.Games:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiche Spiele."""
    return G.simulate_games(s.teams, s.size, s.delta, s.games, s.q, s.wind_eff, s.aggregate, s.seed)
