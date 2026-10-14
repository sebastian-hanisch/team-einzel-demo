"""PDF-Export: erzeugt für jede Konfiguration gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import tei_constants as C
import tei_evaluation as ev
import tei_pdf_export as X
from tei_scenario import Settings


def test_pdf_is_built_for_every_configuration():
    for s in (Settings(games=60), Settings(games=60, q=0.0), Settings(games=60, q=1.0, use_wind=True, wind=0.8), Settings(teams=3, size=2, games=40, aggregate=C.AGGREGATES[1]),
              Settings(teams=6, size=4, games=60)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("λ ≈ 0,5; σ → τ µ θ − **fett** €")
    assert out == "lambda ca. 0,5; sigma -> Tau mu theta - fett EUR"
    assert out.encode("latin-1")
