"""Von der Mannschaft zum Einzelnen – wer trägt das Team? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Ranking-Linie: Beobachtet werden nur Ergebnisse von Mannschaften, gesucht ist die Stärke jedes Einzelnen. Sechs Experimente messen, wann sich Spieler derselben
Mannschaft trennen lassen (Rotation der Aufstellungen), was die naive Siegquote gegen eine Regression mit Regularisierung kostet (Windschatten) und was passiert, wenn die
Mannschaftsstärke nicht die Summe ist. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import tei_constants as C
import tei_evaluation as ev
from tei_pdf_export import build_pdf
from tei_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from tei_scenario import Settings
from tei_visualization import (
    build_accuracy,
    build_aggregate,
    build_fixed,
    build_lambda,
    build_rotation,
    build_scatter,
    build_wind,
)

st.set_page_config(page_title="Von der Mannschaft zum Einzelnen – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _rotation():
    return ev.sweep_rotation()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _wind():
    return ev.sweep_wind()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _lambda():
    return ev.sweep_lambda()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _aggregate():
    return ev.sweep_aggregate()


st.title("🤝 Von der Mannschaft zum Einzelnen: wer trägt das Team?")
st.markdown(
    """
Beobachtet werden nur **Mannschaftsergebnisse**, gesucht ist die **Stärke jedes Einzelnen**. Diese Demo ist das **zehnte Stück der Ranking-Linie** und fragt, wann das gelingt:
Spieler, die immer **zusammen** spielen, sind nicht zu trennen, mehr Spiele ändern daran nichts. Erst wechselnde **Aufstellungen** machen die Einzelnen sichtbar, und die naive
**Siegquote** lässt den Mitspieler mitzählen. Gemessen werden vier Schätzer auf denselben Spielen: Siegquote, Elo für Teams, **Regression mit Regularisierung** und TrueSkill für Teams.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie aus Mannschaften Einzelne werden", expanded=True):
    st.markdown(
        """
1. **Annahme:** Die Stärke einer Mannschaft ist die **Summe** der Stärken ihrer Spieler; die Mannschaft A gewinnt mit der Wahrscheinlichkeit σ(Stärke A − Stärke B). Beobachtet wird nur der Ausgang.
2. **Designmatrix:** Jedes Spiel ist eine Zeile mit +1 für die Spieler der Mannschaft A und −1 für die der Mannschaft B. Aus ihr schätzt die Regression die Stärken.
3. **Kollinearität:** Spielen immer dieselben Spieler zusammen, haben ihre Spalten dieselbe Form: Die Differenz zwischen zwei Mitspielern taucht in keiner Zeile auf. Der Rang der
   Matrix sinkt, die Spieler sind nicht zu trennen.
4. **Ausweg:** wechselnde Aufstellungen (Rotation) und eine **Regularisierung**, die bei wenig Information die Schätzungen zur Mitte hin zieht. Sie hat einen Preis: Sie zieht
   bei wenig Information auch einen schwachen Dauer-Mitspieler zu seinem starken Partner (Experiment 4).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Mannschaften und Spiele**")
    teams = st.slider("Feste Mannschaften", *bounds("teams_slider"), key="teams_slider", help="Anzahl der festen Mannschaften; jede hat die unten gewählte Größe.")
    size = st.selectbox("Spieler je Mannschaft", C.SIZE_CHOICES, key="size_select")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier in der Stärke benachbarter Spieler (logistische Einheiten).")
    games = st.slider("Spiele", *bounds("games_slider"), step=C.GAMES_STEP, key="games_slider", help="Wie viele Mannschaftsspiele beobachtet werden.")
    st.markdown("**Aufstellungen**")
    q = st.slider("Rotation q", *bounds("q_slider"), step=C.Q_STEP, key="q_slider", format="%.2f",
                  help="Anteil der Spiele, in denen alle Spieler neu auf zwei Mannschaften verteilt werden. Sonst spielen zwei feste Mannschaften gegeneinander.")
    use_wind = st.checkbox("Windschatten: der Schwächste spielt neben dem Besten", key="use_wind_check",
                           help="In einem Teil der Spiele steht der schwächste Spieler in derselben Mannschaft wie der beste.")
    if use_wind:
        seed_widget("wind_slider")
        wind = st.slider("Anteil der Windschatten-Spiele", *bounds("wind_slider"), step=C.WIND_STEP, key="wind_slider", format="%.2f")
        remember("wind_slider", wind)
    aggregate = st.selectbox("Wie entsteht die Mannschaftsstärke?", C.AGGREGATES, key="aggregate_select",
                             help="Summe der Stärken (die Annahme der Schätzer) oder: der Beste trägt (das m-fache des Besten). Die Schätzer rechnen immer mit der Summe.")
    st.markdown("**Schätzer**")
    k = st.slider("Elo-Team: K in Punkten je Spieler", *bounds("k_slider"), key="k_slider", help="Wertungsänderung je Spieler und Spiel: K mal (Ergebnis minus Erwartung).")
    lam = st.slider("Ridge-Regularisierung λ", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider", format="%.1f",
                    help="Prior N(0, 1/λ) auf jeder Stärke. 0 = kaum Regularisierung. Es gibt kein für alle Lagen bestes λ: bei wenig Information hilft viel, beim Windschatten schadet es (Experimente 4 und 5).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

wind_v = round(float(current_value("wind_slider")), 2)
sync_query_params({"teams_slider": int(teams), "size_select": int(size), "delta_slider": round(float(delta), 2), "games_slider": int(games), "q_slider": round(float(q), 2),
                   "use_wind_check": bool(use_wind), "wind_slider": wind_v, "aggregate_select": aggregate, "k_slider": int(k), "lam_slider": round(float(lam), 1),
                   "seed_input": int(seed)})

settings = Settings(int(teams), int(size), round(float(delta), 2), int(games), round(float(q), 2), bool(use_wind), wind_v, aggregate, float(k), round(float(lam), 1), int(seed))
a = _analysis(settings)
g = a.games

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Schätzungen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Paartreffer: Siegquote", ev.de(a.accuracy["Siegquote"]), delta=f"{ev.de(100 * a.ties['Siegquote'], 0)} % Gleichstände", delta_color="off",
          help="Anteil richtig geordneter Spielerpaare; Gleichstände zählen halb, 0,5 ist Zufall.")
m2.metric("Paartreffer: Elo-Team", ev.de(a.accuracy["Elo-Team"]))
m3.metric("Paartreffer: Ridge", ev.de(a.accuracy["Ridge"]), delta=f"Rang {a.rank} von {settings.n - 1}", delta_color="off",
          help="Rang der Designmatrix: wie viele Richtungen im Raum der Stärken die Spiele überhaupt unterscheiden. Weniger als n − 1 heißt: Spieler sind nicht zu trennen.")
m4.metric("Paartreffer: TrueSkill-Teams", ev.de(a.accuracy["TrueSkill-Teams"]))

st.plotly_chart(build_scatter(a), width="stretch", key="scatter_chart")
st.caption("Geschätzte gegen wahre Stärke je Schätzer (Skala an die wahre Stärke angeglichen, Elo-Punkte um 1500). Spieler derselben Mannschaft liegen bei festen Aufstellungen "
           "übereinander: Die Schätzer können sie nicht trennen.")
st.plotly_chart(build_accuracy(a), width="stretch", key="accuracy_chart")
st.caption(f"Anteil der Spiele: feste Mannschaften {ev.de(100 * a.kind_share[0], 0)} %, Rotation {ev.de(100 * a.kind_share[1], 0)} %, Windschatten {ev.de(100 * a.kind_share[2], 0)} %.")

st.dataframe(pd.DataFrame(ev.table(a)), hide_index=True, width="stretch")
st.download_button("📄 PDF der aktuellen Schätzungen", data=build_pdf(a), file_name="mannschaft-einzel.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wer trägt das Team?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Feste Mannschaften – mehr Spiele helfen nicht")
st.caption("12 Spieler in 4 festen Mannschaften zu 3 (feste Instanz, unabhängig vom Regler), nie neue Aufstellungen; 60, 200 und 600 Spiele, 60 feste Läufe. Gleichstände des "
           "Schätzers zählen halb.")
rows_rot = _rotation()
st.plotly_chart(build_fixed(rows_rot), width="stretch", key="exp1_chart")
verdict(*ev.verdict_fixed(rows_rot))
st.dataframe(pd.DataFrame([{"Spiele": r["games"], "Rang der Designmatrix": round(r["rank"], 1), "Gleichstände der Siegquote": f"{ev.de(100 * r['ties_rate'], 0)} %",
                            **{k: round(r["acc"][k], 3) for k in C.EST_NAMES}} for r in rows_rot if r["q"] == 0.0]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Wie viel Rotation reicht?")
st.caption("Anteil q der Spiele mit neu gewürfelter Aufstellung (0 bis 1), Spielzahl wählbar; Paartreffer der Schätzer, 60 feste Läufe.")
games_choice = st.radio("Spiele", C.GAMES_GRID, horizontal=True, index=len(C.GAMES_GRID) - 1, key="rotation_games_choice")
st.plotly_chart(build_rotation(rows_rot, games_choice), width="stretch", key="exp2_chart")
verdict(*ev.verdict_rotation(rows_rot))

st.subheader("🔬 Experiment 3: Siegquote gegen Regression")
st.caption("Dieselben Läufe: bei teilweiser Rotation (q = 0,1) und 200 Spielen gegen vollständige Rotation. Die Siegquote lässt den Mitspieler mitzählen.")
verdict(*ev.verdict_methods(rows_rot))
st.dataframe(pd.DataFrame([{"q": ev.de(r["q"], 1), "Spiele": r["games"], **{k: round(r["acc"][k], 3) for k in (*C.EST_NAMES, "Maximum Likelihood")}} for r in rows_rot]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Windschatten")
st.caption("Der Schwächste steht in einem Anteil c der Spiele neben dem Besten (sonst vollständige Rotation), 300 Spiele; Fehler seiner Einzelstärke nach Angleichen der Skala, "
           "60 feste Läufe. „Maximum Likelihood“ ist die Regression mit λ ≈ 0.")
rows_w = _wind()
st.plotly_chart(build_wind(rows_w), width="stretch", key="exp4_chart")
verdict(*ev.verdict_wind(rows_w))
st.dataframe(pd.DataFrame([{"Windschatten c": f"{ev.de(100 * r['c'], 0)} %", **{f"{k}: Fehler": round(r["err"][k], 2) for k in (*C.EST_NAMES, "Maximum Likelihood")},
                            "Siegquote: Plätze zu hoch": round(r["places"]["Siegquote"], 1), "Ridge: Plätze zu hoch": round(r["places"]["Ridge"], 1)} for r in rows_w]),
             hide_index=True, width="stretch")

st.subheader("🔬 Experiment 5: Wie viel Regularisierung?")
st.caption("Teilweise Rotation (q = 0,1), 200 Spiele; Ridge-λ von kaum bis stark; RMSE der Einzelstärken (Skala angeglichen) und Paartreffer, 60 feste Läufe.")
rows_l = _lambda()
st.plotly_chart(build_lambda(rows_l), width="stretch", key="exp5_chart")
verdict(*ev.verdict_lambda(rows_l))

st.subheader("🔬 Experiment 6: Wenn der Beste trägt")
st.caption("Wahrheit: die Mannschaftsstärke ist die Summe der Stärken, oder der Beste trägt (das Dreifache seiner Stärke); die Schätzer rechnen immer mit der Summe. Rotation "
           "q = 0,3, 300 Spiele, 60 feste Läufe.")
rows_a = _aggregate()
st.plotly_chart(build_aggregate(rows_a), width="stretch", key="exp6_chart")
verdict(*ev.verdict_aggregate(rows_a))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Aufstellungen wechseln** | Bei festen Mannschaften sind die Spieler nicht zu trennen, egal wie viele Spiele es gibt (Experiment 1). | – |
| **Die Aufstellungen sind zufällig** | Stellt ein Trainer die Besten zusammen, steht der Mitspieler im Ergebnis: der Windschatten (Experiment 4). Hier nur als Anteil c nachgestellt. | – |
| **Die Mannschaftsstärke ist die Summe** | Trägt der Beste, verteilt der additive Ansatz den Erfolg auf alle (Experiment 6); eine andere Aggregatfunktion wird nicht geschätzt. | – |
| **Ein Spiel ist Sieg oder Niederlage** | Remis, Tore und Heimvorteil haben ein eigenes Modell; hier nur der Ausgang. | Tore, Remis, Heimvorteil |
| **Die Stärken sind fest** | Eine Zeitachse (Form, Verletzung, Wechsel) gehört nicht zum Modell. | WHR und TrueSkill Through Time |
| **Alle Spieler spielen die gleiche Rolle** | Positionen und Rollen (Torwart, Stürmer) sind nicht modelliert. | – |
| **Das λ ist bekannt** | Es gibt kein für alle Lagen bestes λ: bei wenig Information hilft viel, beim Windschatten (300 Spiele) schadet es (Experimente 4 und 5). | – |
| **Alles ist simuliert** | Stärken, Aufstellungen und Ausgänge sind Konstruktionen; keine Aussage über einen Sport oder über echte Plus-Minus-Statistiken. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Spieler $i = 1, \dots, n$ mit Stärken $s_i$ (logistische Einheiten, $1$ Einheit $= 400/\ln 10 \approx 173{,}7$ Elo-Punkte). Ein Spiel $g$ hat die Mannschaften $A_g$,
$B_g$ gleicher Größe $m$; es gewinnt $A_g$ mit $P = \sigma\bigl(\sum_{i \in A_g} s_i - \sum_{j \in B_g} s_j\bigr)$. Variante: die Stärke der Mannschaft ist $m \cdot \max_i s_i$.

**Designmatrix.** $X \in \{-1, 0, 1\}^{G \times n}$ mit $X_{gi} = +1$ für $i \in A_g$, $-1$ für $i \in B_g$, sonst $0$; es gilt $P(y_g = 1) = \sigma\bigl((X s)_g\bigr)$. Jede Zeile
summiert zu null, der Rang ist höchstens $n - 1$. Bei festen Mannschaften sind die Spalten der Spieler einer Mannschaft gleich: der Rang ist (Mannschaften $- 1$).

**Ridge (MAP).** $\hat s = \arg\max_s \sum_g \ln \sigma\bigl(\pm (X s)_g\bigr) - \tfrac{\lambda}{2}\lVert s \rVert^2$; strikt konkav, Newton-Verfahren mit
$H = X^\top W X + \lambda I$, $W = \mathrm{diag}\bigl(p_g(1 - p_g)\bigr)$.

**Elo-Team.** $r_i \leftarrow r_i + k\,(y_g - E_g)$ für $i \in A_g$, $r_j \leftarrow r_j - k\,(y_g - E_g)$ für $j \in B_g$, mit $E_g = \sigma\bigl(\sum_{A_g} r - \sum_{B_g} r\bigr)$.

**TrueSkill-Teams** (1 Durchlauf, ohne Remis). Mit $c^2 = 2 m \beta^2 + \sum_{i \in A_g \cup B_g} \sigma_i^2$, $t = (\sum_w \mu - \sum_l \mu)/c$, $v = \phi(t)/\Phi(t)$,
$w = v\,(v + t)$: $\mu_i \leftarrow \mu_i \pm \sigma_i^2 v/c$, $\sigma_i^2 \leftarrow \sigma_i^2\,(1 - \sigma_i^2 w/c^2)$; $\beta = 1{,}702/\sqrt 2$.

**Paartreffer.** Anteil der Spielerpaare $(i, j)$, die der Schätzer wie die Wahrheit ordnet; ein Gleichstand des Schätzers zählt $\tfrac12$. **RMSE:** $\sqrt{\tfrac1n \sum (c\,\hat s_i - s_i)^2}$
nach Zentrieren und mit dem Faktor $c$ der Kleinste-Quadrate-Angleichung.

**Zufall.** Aufstellungen und Ausgänge aus der Gleichverteilten des Generators (SplitMix64); feste Mannschaften zufällig zugeordnet.

**Literatur.**
- Herbrich, R., Minka, T., Graepel, T. (2006): TrueSkill™: A Bayesian skill rating system. *Advances in Neural Information Processing Systems 19*, 569–576 (im Volltext eingesehen:
  Mannschaftsleistung als Summe der Spielerleistungen, Erwartungsabgleich mit V und W; die ausgeschriebene Aktualisierung µ_i += σ_i² v/c steht im Papier nicht, sie folgt aus dem Modell
  und ist hier nach der Dokumentation des Pakets `trueskill` angesetzt).
- Hoerl, A. E., Kennard, R. W. (1970): Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics* 12 (1), 55–67 (im Scan des Technometrics-Archivs eingesehen:
  Schätzer [X'X + kI]⁻¹ X'Y, Gleichung (2.1); die logistische Fassung der Demo steht dort nicht).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
