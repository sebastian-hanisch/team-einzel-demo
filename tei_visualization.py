"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import tei_constants as C
import tei_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scatter(a: ev.Analysis):
    """Geschätzte gegen wahre Stärke je Schätzer (Skala an die wahre Stärke angeglichen, Elo-Punkte um 1500); die Diagonale ist der perfekte Schätzer."""
    g = a.games
    truth = C.BASE + C.SCALE * (g.s - g.s.mean())
    fig = go.Figure()
    lo, hi = float(truth.min()) - 40, float(truth.max()) + 40
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(150,150,150,0.8)", dash="dash"), name="perfekt", hoverinfo="skip"))
    for name in C.EST_NAMES:
        e = a.estimates[name]
        y = C.BASE + C.SCALE * ev.align_scale(e, g.s) * (e - e.mean())
        fig.add_trace(go.Scatter(x=truth, y=y, mode="markers", name=name, marker=dict(color=C.COLORS[name], size=9, opacity=0.85),
                                 text=[f"Spieler {p + 1}, Mannschaft {1 + next(k for k, t in enumerate(g.teams) if p in t)}" for p in range(g.n)],
                                 hovertemplate="%{text}: wahr %{x:.0f}, geschätzt %{y:.0f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title_text="wahre Stärke (Elo-Punkte)")
    fig.update_yaxes(title_text="geschätzte Stärke (Skala angeglichen)")
    return _base(fig, height=420)


def build_accuracy(a: ev.Analysis):
    names = list(C.EST_NAMES)
    fig = go.Figure(go.Bar(x=names, y=[a.accuracy[k] for k in names], marker_color=[C.COLORS[k] for k in names], text=[ev.de(a.accuracy[k]) for k in names], textposition="auto",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="Zufall", annotation_position="bottom right")
    fig.update_yaxes(title_text="Anteil richtig geordneter Spielerpaare", range=[0.4, 1.02])
    return _base(fig, height=300)


def build_fixed(rows):
    """Paartreffer bei festen Mannschaften (q = 0) je Spielzahl, alle Schätzer."""
    sel = [r for r in rows if r["q"] == 0.0]
    fig = go.Figure()
    for name in C.EST_NAMES:
        fig.add_trace(go.Bar(x=[f"{r['games']} Spiele" for r in sel], y=[r["acc"][name] for r in sel], name=name, marker_color=C.COLORS[name],
                             hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(150,150,150,0.8)", annotation_text="Zufall", annotation_position="bottom right")
    fig.update_yaxes(title_text="Paartreffer", range=[0.4, 1.0])
    return _base(fig)


def build_rotation(rows, games):
    """Paartreffer je Rotation q bei fester Spielzahl, alle Schätzer."""
    sel = [r for r in rows if r["games"] == games]
    fig = go.Figure()
    for name in C.EST_NAMES:
        fig.add_trace(go.Scatter(x=[ev.de(r["q"], 1) for r in sel], y=[r["acc"][name] for r in sel], mode="lines+markers", name=name, line=dict(color=C.COLORS[name], width=3),
                                 hovertemplate=name + ", q = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Rotation q", type="category")
    fig.update_yaxes(title_text=f"Paartreffer nach {games} Spielen", range=[0.5, 1.0])
    return _base(fig)


def build_wind(rows):
    names = (*C.EST_NAMES, "Maximum Likelihood")
    colors = {**C.COLORS, "Maximum Likelihood": "#5f9d95"}
    fig = go.Figure()
    for name in names:
        fig.add_trace(go.Scatter(x=[f"{ev.de(100 * r['c'], 0)} %" for r in rows], y=[r["err"][name] for r in rows], mode="lines+markers", name=name,
                                 line=dict(color=colors[name], width=3), hovertemplate=name + ", c = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    fig.update_xaxes(title_text="Anteil der Windschatten-Spiele c", type="category")
    fig.update_yaxes(title_text="Fehler des Schwächsten (θ, + = zu hoch)")
    return _base(fig)


def build_lambda(rows):
    x = [ev.de(r["lam"], 1) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[r["rmse"] for r in rows], mode="lines+markers", name="RMSE (links)", line=dict(color=C.COLOR_RIDGE, width=3),
                             hovertemplate="λ = %{x}: RMSE %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=[r["acc"] for r in rows], mode="lines+markers", name="Paartreffer (rechts)", yaxis="y2", line=dict(color=C.COLOR_ELO, width=3, dash="dot"),
                             hovertemplate="λ = %{x}: Paartreffer %{y:.3f}<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Paartreffer", fixedrange=True))
    fig.update_xaxes(title_text="Ridge-λ (0 = kaum Regularisierung)", type="category")
    fig.update_yaxes(title_text="RMSE der Einzelstärken (Skala angeglichen)")
    return _base(fig)


def build_aggregate(rows):
    fig = go.Figure()
    for name in C.EST_NAMES:
        fig.add_trace(go.Bar(x=[r["aggregate"] for r in rows], y=[r["acc"][name] for r in rows], name=name, marker_color=C.COLORS[name],
                             hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    fig.update_xaxes(title_text="Wie entsteht die Mannschaftsstärke in der Wahrheit?")
    fig.update_yaxes(title_text="Paartreffer", range=[0.4, 1.0])
    return _base(fig)
