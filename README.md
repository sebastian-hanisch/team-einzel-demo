# Von der Mannschaft zum Einzelnen: wer trägt das Team? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-team-einzel-demo.streamlit.app/)**

Zehntes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `whr-demo` und `glicko-trueskill-demo`: Beobachtet werden nur **Mannschaftsergebnisse**, gesucht ist die
**Stärke jedes Einzelnen**. Gemessen wird, wann sich Spieler derselben Mannschaft trennen lassen, was die naive Siegquote gegen eine Regression mit Regularisierung kostet und was
passiert, wenn die Mannschaftsstärke nicht die Summe ist. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Stärken aus dem Generator Version 1, wortgleich; Aufstellungen
und Ausgänge lokal); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Wann lässt sich aus Mannschaftsergebnissen die Einzelstärke schätzen? **Nur soweit die Aufstellungen wechseln.** Spielen die Spieler immer in denselben Trios, sind sie innerhalb der
Mannschaft nicht zu trennen, und **mehr Spiele ändern daran nichts:** Anteil richtig geordneter Paare 0,66 bei 60, 200 und 600 Spielen (Zufall wäre 0,5; Rang der Designmatrix 3 von
11). **Schon jedes zehnte Spiel mit neuer Aufstellung** trennt sie, langsam: Regression 0,74 / 0,83 / 0,91 nach 60 / 200 / 600 Spielen; die **naive Siegquote** lässt den Mitspieler
mitzählen und bleibt bei 0,69 / 0,72 / 0,73 hängen. Steht der Schwächste in 90 % der Spiele neben dem Besten („Windschatten“), liegt die Siegquote ihn um **1,6 Einheiten** zu hoch
(rund fünf Plätze), Elo für Teams um 1,25, TrueSkill-Teams um 1,4, die Regression um 0,5 und die Regression ohne Regularisierung um 0,13. **Die Regularisierung hilft bei wenig
Information** (Fehler 0,57 bei λ = 0,5 gegen 0,73 ohne), **zieht aber beim Windschatten zum Mitspieler hin.** Und wenn **der Beste trägt** statt der Summe, fällt der Paartreffer von
0,93 auf 0,80.

## Modell

- **Spieler und Stärken:** n = Mannschaften × Größe Spieler mit gleichabständigen Stärken (`make_strengths`, Abstand δ, Mittel 0); Spieler 1 ist der Schwächste. Die festen Mannschaften
  sind zufällig zugeordnet, nicht nach Stärke gestapelt.
- **Spiele:** Zwei Mannschaften gleicher Größe m; A gewinnt mit σ(Stärke A − Stärke B). Die Mannschaftsstärke ist die **Summe** der Spielerstärken, oder (Fehlspezifikation) das
  m-fache des Besten. **Aufstellung je Spiel:** mit Wahrscheinlichkeit c *Windschatten* (der Schwächste und der Beste in Mannschaft A, die übrigen Plätze zufällig), sonst mit
  Wahrscheinlichkeit q *Rotation* (alle Spieler neu gewürfelt), sonst zwei feste Mannschaften.
- **Schätzer** (alle auf denselben Spielen): *Siegquote* (Anteil gewonnener Spiele eines Spielers); *Elo-Team* (Wertung der Mannschaft = Summe, K je Spieler); *Ridge* (logistische
  Regression auf die Designmatrix ±1 mit Prior N(0, 1/λ) auf jeder Stärke, MAP, Newton-Verfahren); *TrueSkill-Teams* (Erwartungsabgleich in einem Durchlauf, c² = 2 m β² + Σσ²). Als
  Vergleich die Regression ohne Regularisierung (λ = 10⁻⁶, nur das Niveau fixiert).
- **Maße:** *Paartreffer* (Anteil richtig geordneter Spielerpaare; Gleichstände des Schätzers zählen halb, 0,5 = Zufall), *RMSE nach Angleichen der Skala* (die Schätzer haben
  verschiedene Skalen), *Fehler eines Spielers* (Windschatten), *Rang der Designmatrix*.
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 12 Spielern in 4 Mannschaften zu 3 und
  Stärkeabstand 0,3, unabhängig vom Regler; K = 17 Punkte je Spieler, λ = 1 wenn nicht anders genannt.
- **Gleichstände halb:** Ein Schätzer, der Mitspieler gleichsetzt, sähe nach dem Kendall-τ über die „unterscheidbaren“ Paare perfekt aus (1,000 bei festen Mannschaften im ersten
  Messlauf); die Demo zählt Gleichstände als halbe Treffer.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); die Zufallszahlen eines Spiels haben eine feste Reihenfolge und Größe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 12 Spieler in 4 Mannschaften zu 3, δ = 0,3, Paartreffer, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Feste Mannschaften (q = 0), 60 / 200 / 600 Spiele | Ridge **0,656 / 0,662 / 0,663**, alle Schätzer innerhalb 0,01; Rang der Designmatrix 3 von 11; 18 % Gleichstände der Siegquote. | `test_with_fixed_teams_more_games_do_not_help` |
| Rotation q = 0,1: Ridge / Siegquote | **0,735 / 0,828 / 0,910** gegen **0,692 / 0,721 / 0,734**; Rang 8,7 / 11 / 11. | `test_a_tenth_of_rotating_games_separates_the_players_slowly` |
| q = 0,1, 200 Spiele: weitere Schätzer | Elo-Team 0,757, TrueSkill-Teams 0,757, Regression ohne Regularisierung 0,750; bei 60 Spielen ohne Regularisierung 0,664 gegen Ridge 0,735. | `test_intermediate_rotation_and_other_estimators` |
| q = 0,3: Ridge | 0,804 / 0,900 / 0,960. | `test_intermediate_rotation_and_other_estimators` |
| Vollständige Rotation (q = 1): Ridge / Siegquote | **0,890 / 0,957 / 0,986** gegen 0,866 / 0,928 / 0,974: rücken zusammen (Abstand bei 200 Spielen unter 0,04). | `test_the_regression_and_the_win_rate_come_together_under_full_rotation` |
| Windschatten c = 0 / 0,3 / 0,6 / 0,9: Fehler des Schwächsten, Siegquote | **+0,12 / +0,78 / +1,30 / +1,60** Einheiten; bei c = 0,9: Elo-Team +1,25, TrueSkill-Teams +1,41, Ridge +0,52, Maximum Likelihood +0,13. | `test_the_win_rate_lifts_the_weakest_player_next_to_the_best` |
| Windschatten c = 0,9: Spieler, die der Schätzer unter den Schwächsten setzt | Siegquote **4,9**, Ridge 1,5, Maximum Likelihood 0,6 (ohne Windschatten 0,2 bis 0,4). | `test_the_regularisation_pulls_the_weakest_towards_the_partner` |
| Ridge-λ (q = 0,1, 200 Spiele), λ = 0 / 0,1 / 0,5 / 1 / 3 / 10 | RMSE **0,731 / 0,611 / 0,573 / 0,581 / 0,638 / 0,736**; Paartreffer 0,750 / 0,810 / 0,831 / 0,828 / 0,802 / 0,766. | `test_a_moderate_regularisation_is_best_and_too_much_levels_everyone` |
| Wahrheit „Der Beste trägt“ statt Summe (q = 0,3, 300 Spiele) | Ridge **0,933 → 0,796**, TrueSkill-Teams 0,869 → 0,751, Elo-Team 0,877 → 0,751, Siegquote 0,763 → 0,681. | `test_a_wrong_aggregate_costs_accuracy` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Vorab-Zahlen verschieben sich um wenige Hundertstel** (Ridge bei q = 0,1: 0,75 / 0,82 / 0,90 gegen **0,735 / 0,828 / 0,910**; Windschatten +1,61 / +0,46 gegen **+1,60 / +0,52**):
  anderer Zufallsstrom (SplitMix64 statt numpy), 12 Spieler wie geplant. Die Aussagen sind dieselben.
- **Teams: 3 bis 6 Mannschaften zu 2, 3 oder 4** (6 bis 24 Spieler) statt „6 bis 18 Spieler“: die Anzahl der festen Mannschaften ist der Regler, die Spielerzahl folgt daraus.
- **Der Rang der Designmatrix ist neu im Hauptergebnis:** Er zeigt die Kollinearität direkt (3 von 11 bei festen Mannschaften, 8,7 bei q = 0,1 und 60 Spielen, 11 ab 200 Spielen); der
  Plan hatte die Bedingungszahl vorgesehen, die bei Rangverlust unendlich wäre.
- **Der Ridge ist beim Windschatten die schwächere Wahl als Maximum Likelihood** (+0,52 gegen +0,13); bei wenig Information ist er dafür deutlich stabiler (q = 0,1, 60 Spiele: 0,735
  gegen 0,664). Die Demo zeigt beides und verspricht kein bestes λ.
- **Mittelwert statt Summe wäre nur eine Skala:** Bei gleicher Mannschaftsgröße ist „Mittel × m“ dasselbe wie die Summe; als Fehlspezifikation taugt nur das Maximum.

## Abgrenzung

- **Glicko und TrueSkill** (Stück 8): TrueSkill 1 gegen 1; die Teams-Fassung kommt hier, die Formeln (v, w, β) sind dort geprüft.
- **Bradley-Terry** (Stück 5): derselbe Schätzer für Einzelne; hier mit einer Designmatrix aus Mannschaften und Ridge.
- **WHR und TrueSkill Through Time** (Stück 9): Zeitachse; hier statisch.
- **Nachfolger (nur beim Namen):** Nicht-Transitivität.

## Ehrliche Grenzen

- **Zufällige Aufstellungen:** Echte Mannschaften sind nicht zufällig zusammengesetzt; der Trainer, der die Besten zusammenstellt, ist der Windschatten-Fall, nicht das Modell.
- **Additivität:** Die Summe ist eine Annahme; Experiment 6 zeigt, was es kostet, wenn sie nicht gilt, aber keine andere Aggregatfunktion wird geschätzt.
- **Kein bestes λ:** bei wenig Information hilft viel, beim Windschatten schadet es; die Demo zeigt eine Lage je Experiment.
- **Kleine Felder:** 6 bis 24 Spieler, Mannschaften zu 2 bis 4, ein Generator, 60 Läufe je Zelle; Unterschiede unter etwa 0,01 im Paartreffer sind Rauschen.
- **Keine Rollen, kein Heimvorteil, keine Zeitachse,** nur Sieg und Niederlage.
- **Alles simuliert:** keine Aussage über einen Sport oder über echte Plus-Minus-Statistiken.

## Quellen

- Herbrich, R., Minka, T., Graepel, T. (2006): TrueSkill™: A Bayesian skill rating system. *Advances in Neural Information Processing Systems 19*, 569–576 (eingesehen
  2026-10-17 im Volltext der NIPS-Fassung: Mannschaftsleistung als Summe der Spielerleistungen p_i ~ N(s_i, β²) (Abschnitt 2), Erwartungsabgleich mit V und W (Tabelle 1, Abschnitt 3); die
  ausgeschriebene Aktualisierung µ_i += σ_i² v/c mit c² = 2 m β² + Σσ² steht im Papier nicht, sie folgt aus Modell und Tabelle 1 und ist hier nach der Dokumentation des Pakets `trueskill`
  angesetzt). Die Aktualisierung für Teams ist gegen `trueskill` 0.4.5 geprüft.
- Hoerl, A. E., Kennard, R. W. (1970): Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics* 12 (1), 55–67 (eingesehen 2026-10-17 im Scan des Technometrics-Archivs, stat.cmu.edu: Schätzer [X'X + kI]⁻¹ X'Y, Gleichung (2.1), und Existenzsatz in Abschnitt 4b (immer ein k > 0 mit kleinerem mittlerem
  quadratischem Fehler), bayessche Deutung als Prior mit Mittel 0 in Abschnitt 6; die logistische Fassung der Demo steht dort nicht).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** Arbeiten zu verallgemeinerten Bradley-Terry-Modellen für Teams und zu regularisierten Plus-Minus-Statistiken im Sport; die Zahlen dieser Demo sind Simulation.

## Tests

107 Tests, rund eine Minute: Einheiten gegen Handwerte (`test_games.py`: Aufstellungen, Windschatten, Aggregat, Ausgänge; `test_estimators.py`: Designmatrix, Siegquote, Elo-Team,
Ridge-Regression, TrueSkill-Teams), Generator (`test_generator.py`, wortgleich), **Orakel** (`test_oracle_team_einzel.py`: Ridge-Regression gegen den Maximierer von scipy; Rang der
Designmatrix gegen `numpy.linalg.matrix_rank` für feste und rotierende Aufstellungen; **TrueSkill für Teams gegen das Paket `trueskill`** auf 25 Spielen; Elo-Team gegen die
Lehrbuchformel in Punkten; Paartreffer und Skalenangleichung von Hand; Windschatten mit starkem Signal), Maße, Tabelle und Verdikte (`test_evaluation.py`), alle Zahlen dieses README
(`test_claims.py`), Presets und Szenario (`test_presets.py`), ausblendbarer Regler (`test_hidden_widget_state.py`), PDF, Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `tei_`-Module, gegen die Einheitentests ohne App-Tests): Der erste Lauf (126 Stellen; durch den
Neustart der Sitzung bei 116 unterbrochen) ließ 25 Stellen überleben. Davon waren **zwei tote Codestellen** (eine ungenutzte Eigenschaft und eine nie greifende Prüfung der
Spielerzahl; entfernt, es bleiben 116 Stellen) und **echte Lücken**: die Mindestgröße des Feldes, die Siegquote bei genau einem Spiel, der Name des besten Schätzers im Verdikt,
die Nummerierung und die Mannschaftszuordnung in der Tabelle, die Zentrierung der Schätzungen und die Grenzen der Hinweise auf fehlenden Rang und auf den Windschatten. Sie sind mit
Tests geschlossen. Zweiter Lauf über alle 116 Stellen: **102 getötet, 14 überlebt;** 9 davon sind mit den neuen Tests einzeln auf einer Kopie getötet, **5 sind äquivalent:**
Vergleiche bei stetig verteilten Zufallszahlen (`u < p` gegen `u <= p`, dreimal), die Schleifengrenze des Newton-Verfahrens (es bricht vorher ab) und das Vorzeichen der Zentrierung
der Wahrheit in der Skalenangleichung (die Schätzung ist zentriert, das Mittel der Wahrheit fällt heraus).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; liefert die Stärken) |
| `tei_games.py` | Aufstellungen (feste Mannschaften, Rotation, Windschatten) und Ausgänge |
| `tei_estimators.py` | Siegquote, Elo-Team, Ridge-Regression, TrueSkill-Teams, Designmatrix |
| `tei_scenario.py` | Reglerwerte (`Settings`) und die Spiele |
| `tei_evaluation.py` | Maße, Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `tei_presets.py`, `tei_constants.py` | Permalink, Presets, ausblendbarer Regler, Grenzen |
| `tei_visualization.py`, `tei_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Andere Aggregatfunktionen schätzen** (Maximum, Rollen, Synergie): würde eine Modellwahl aus den Daten verlangen.
- **Trainer-Aufstellungen** mit Absicht (die Besten zusammen) jenseits des Windschatten-Anteils.
- **Zeitachse** (Form, Verletzung): Stück „WHR und TrueSkill Through Time“.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy und das Paket `trueskill` für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
