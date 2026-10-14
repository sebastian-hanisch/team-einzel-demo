"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück."""

import os

from streamlit.testing.v1 import AppTest

import tei_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def keys(at):
    return {s.key for s in at.slider}


def test_the_windschatten_share_is_drawn_only_with_its_switch():
    at = fresh()
    assert "wind_slider" not in keys(at)
    at.checkbox(key="use_wind_check").set_value(True).run(timeout=TIMEOUT)
    assert "wind_slider" in keys(at)
    at.checkbox(key="use_wind_check").set_value(False).run(timeout=TIMEOUT)
    assert "wind_slider" not in keys(at) and not at.exception


def test_hidden_value_survives_and_is_not_written_into_the_hidden_widget():
    at = fresh()
    at.checkbox(key="use_wind_check").set_value(True).run(timeout=TIMEOUT)
    at.slider(key="wind_slider").set_value(0.35).run(timeout=TIMEOUT)
    at.checkbox(key="use_wind_check").set_value(False).run(timeout=TIMEOUT)
    assert "wind_slider" not in at.session_state
    at.checkbox(key="use_wind_check").set_value(True).run(timeout=TIMEOUT)
    assert abs(at.slider(key="wind_slider").value - 0.35) < 1e-9
    assert not at.exception


def test_preset_sets_values_and_the_hidden_widget_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Windschatten"]].click().run(timeout=TIMEOUT)
    assert at.checkbox(key="use_wind_check").value and abs(at.slider(key="wind_slider").value - 0.9) < 1e-9 and at.slider(key="q_slider").value == 1.0
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Immer dieselben Trios"]].click().run(timeout=TIMEOUT)
    assert not at.checkbox(key="use_wind_check").value and at.slider(key="q_slider").value == 0.0 and "wind_slider" not in at.session_state
    assert not at.exception


def test_extreme_team_sizes_and_counts_run():
    at = fresh()
    at.selectbox(key="size_select").set_value(4).run(timeout=TIMEOUT)
    at.slider(key="teams_slider").set_value(C.TEAMS_MAX).run(timeout=TIMEOUT)
    assert not at.exception
    at.selectbox(key="size_select").set_value(2).run(timeout=TIMEOUT)
    at.slider(key="teams_slider").set_value(C.TEAMS_MIN).run(timeout=TIMEOUT)
    at.checkbox(key="use_wind_check").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
