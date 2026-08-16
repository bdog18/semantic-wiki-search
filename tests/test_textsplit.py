from swsearch.common.textsplit import split_paragraphs


def test_split_paragraphs_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    assert split_paragraphs(text) == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_split_paragraphs_tolerates_extra_blank_lines():
    text = "First.\n\n\n\nSecond."
    assert split_paragraphs(text) == ["First.", "Second."]


def test_split_paragraphs_empty_input():
    assert split_paragraphs("") == []
    assert split_paragraphs("\n\n") == []
