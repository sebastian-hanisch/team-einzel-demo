"""PDF-Export der aktuellen Konfiguration und der Schätzungen (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import tei_constants as C
import tei_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("λ", "lambda"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Von der Mannschaft zum Einzelnen: wer trägt das Team?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    wind = f", Windschatten {ev.de(100 * s.wind_eff, 0)} %" if s.use_wind else ""
    pdf.multi_cell(0, 5.5, _latin(f"{s.teams} feste Mannschaften zu {s.size} ({s.n} Spieler), {s.games} Spiele, Rotation q = {ev.de(s.q)}{wind}, Wahrheit: {s.aggregate}, "
                                  f"Elo-K = {ev.de(s.k, 0)} Punkte, Ridge lambda = {ev.de(s.lam, 1)}, Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Spieler", "Team", "wahr", "Siegquote", "Elo-Team", "Ridge", "TrueSkill", "+- TS")
    widths = (20, 16, 20, 26, 24, 20, 24, 20)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    keys = ("Spieler", "Mannschaft", "wahre Stärke", "Siegquote", "Elo-Team", "Ridge", "TrueSkill-Teams", "± TrueSkill")
    for row in ev.table(a):
        for key, w in zip(keys, widths):
            pdf.cell(w, 6, _latin(str(row[key])), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Stärken aus dem Generator der Ranking-Linie, Aufstellungen und Ausgänge zufällig); die wahre Stärke ist nur in der Simulation bekannt. Die Schätzungen sind "
        "in Elo-Punkten um 1500 dargestellt, ihre Skala an die wahre Stärke angeglichen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
