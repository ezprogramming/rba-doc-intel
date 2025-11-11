from app.pdf.chunker import _extract_section_hint


def test_extract_section_hint_detects_enumerated_heading():
    text = "3.2 Financial Conditions\nInflation remained elevated ..."
    assert _extract_section_hint(text) == "3.2 Financial Conditions"


def test_extract_section_hint_detects_uppercase_heading():
    text = "SERVICES INFLATION OUTLOOK\nThe services sector ..."
    assert _extract_section_hint(text) == "Services Inflation Outlook"
