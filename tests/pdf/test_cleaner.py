from app.pdf import cleaner


def test_clean_text_drops_header_and_normalizes_space():
    text = "1 Reserve Bank of Australia\n\nData point\n\nAnother line"
    result = cleaner.clean_text(text)
    assert result == "Data point Another line"
