"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter und Szenario."""

import numpy as np
import pytest

import tei_constants as C
import tei_presets as P
from tei_scenario import Settings, make_games

FIELDS = {"teams": "teams", "size": "size", "delta": "delta", "games": "games", "q": "q", "use_wind": "use_wind", "wind": "wind", "aggregate": "aggregate", "k": "k", "lam": "lam",
          "seed": "seed"}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Teilweise Rotation"]
    assert Settings(**{field: p[key] for key, field in FIELDS.items()}) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(FIELDS.values()) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_the_only_hidden_widget_is_the_windschatten_share():
    assert P.KEPT_KEYS == ("wind_slider",)


def test_bool_caster_reads_the_address_values():
    assert P._bool("True") and P._bool("1") and P._bool("true") and not P._bool("False") and not P._bool("0") and not P._bool("")


def test_the_presets_show_their_mechanism():
    assert C.PRESETS["Immer dieselben Trios"]["q"] == 0.0 and not C.PRESETS["Immer dieselben Trios"]["use_wind"]
    assert C.PRESETS["Vollständige Rotation"]["q"] == 1.0
    w = C.PRESETS["Windschatten"]
    assert w["use_wind"] and w["wind"] >= 0.9


def test_sweep_instances_are_fixed_and_distinct():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------

def test_settings_derive_player_count_and_windschatten_share():
    s = Settings(teams=5, size=2, use_wind=True, wind=0.4)
    assert s.n == 10 and s.wind_eff == 0.4 and Settings(use_wind=False, wind=0.4).wind_eff == 0.0
    assert Settings(k=C.SCALE).k_theta == pytest.approx(1.0)


def test_make_games_follows_the_settings_and_is_deterministic():
    s = Settings(teams=3, size=2, games=40, q=1.0, use_wind=True, wind=1.0, seed=4)
    a, b = make_games(s), make_games(s)
    assert a.n == 6 and a.a.shape == (40, 2) and (a.kind == 2).all() and np.array_equal(a.y, b.y)
    off = make_games(Settings(teams=3, size=2, games=40, q=1.0, use_wind=False, wind=1.0, seed=4))
    assert (off.kind == 1).all()
