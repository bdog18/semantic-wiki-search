from swsearch.extract.wikitext import clean_paragraph, is_furniture, strip_markup

# The five markup shapes actually observed in a 50k-paragraph sample of the
# baseline index, kept verbatim so the tests fail if a rewrite stops covering
# what the corpus really contains.
FILE_CAPTION = "[[File:Ernst Middendorp 20180316.jpg|thumb|right|200px|[[Ernst Middendorp]] became the manager of Asante Kotoko]]"
TEMPLATE = "{{#ifeq:{{{transcludesection|ESresultInvercargill}}}|ESresultInvercargill|"
TABLE_ROW = '! colspan="3" style="text-align:right;font-weight:normal" {{!}} Turnout'
HEADING = "== Early life =="
PROSE = "The obliquity of the orbit of the brown dwarf with respect to the axis of rotation of the star has been measured."


def test_drops_file_caption():
    assert clean_paragraph(FILE_CAPTION) is None


def test_drops_file_caption_before_unwrapping_nested_links():
    # The regression this module exists to prevent: unwrapping the nested
    # [[Ernst Middendorp]] first would yield "Ernst Middendorp became the
    # manager of Asante Kotoko", which reads as prose and would survive.
    assert is_furniture(FILE_CAPTION)


def test_drops_image_and_media_variants():
    assert clean_paragraph("[[Image:x.jpg|thumb|caption]]") is None
    assert clean_paragraph("[[media:x.ogg|listen]]") is None


def test_drops_template_fragment():
    assert clean_paragraph(TEMPLATE) is None


def test_drops_table_markup():
    assert clean_paragraph(TABLE_ROW) is None
    assert clean_paragraph("| Turnout") is None
    assert clean_paragraph('{| class="wikitable"') is None


def test_drops_bare_heading():
    assert clean_paragraph(HEADING) is None


def test_keeps_prose_unchanged():
    assert clean_paragraph(PROSE) == PROSE


def test_unwraps_inline_links_in_prose():
    text = "The [[Roman Empire|Romans]] built [[aqueduct]]s across the province."
    assert clean_paragraph(text) == "The Romans built aqueducts across the province."


def test_unwraps_external_links():
    assert clean_paragraph("See [https://example.com the report] for details.") == "See the report for details."
    # A bare external link has no display text and leaves nothing behind.
    assert clean_paragraph("Nothing here [https://example.com] either.") == "Nothing here either."


def test_strips_bold_and_italic_quotes():
    assert clean_paragraph("'''Hamlet''' is a ''tragedy'' by Shakespeare.") == "Hamlet is a tragedy by Shakespeare."


def test_keeps_prose_containing_inline_template():
    # Real prose carrying an inline {{convert}} is prose. Dropping it was a
    # false positive that cost 3 paragraphs per 50,000 of genuine content.
    text = "The obliquity of the orbit was measured at {{convert|12|deg}} by the survey team."
    assert clean_paragraph(text) == "The obliquity of the orbit was measured at by the survey team."


def test_keeps_prose_containing_embedded_caption():
    text = "Operations began in 1918. [[File:Plant.jpg|thumb|The plant in 1920]] Output rose sharply thereafter."
    cleaned = clean_paragraph(text)
    assert cleaned is not None
    assert "Output rose sharply thereafter." in cleaned
    assert "thumb" not in cleaned and "The plant in 1920" not in cleaned


def test_keeps_prose_with_empty_link_display_text():
    # [[stty|]] has empty display text and must fall back to the target
    # rather than stranding brackets and getting the paragraph dropped.
    text = "[[Unix-like]] operating systems can still use it as the [[stty|]] erase character."
    assert clean_paragraph(text) == "Unix-like operating systems can still use it as the stty erase character."


def test_nested_media_caption_does_not_leak_tail_as_prose():
    # Non-greedy matching would stop at the inner link's ]] and leave
    # "became the manager of Asante Kotoko]]" behind looking like a sentence.
    assert "became the manager" not in strip_markup(FILE_CAPTION)


def test_paragraph_hollowed_out_by_stripping_fails_min_chars():
    # Mostly-template text that doesn't *lead* with a template survives
    # is_furniture, strips to almost nothing, and is then rejected by the
    # caller's length threshold rather than by a floor baked in here.
    text = "x {{a}} {{b}} {{c}}"
    assert clean_paragraph(text) == "x"
    assert clean_paragraph(text, min_chars=70) is None


def test_stripped_to_nothing_is_always_dropped():
    assert clean_paragraph("{{a}}{{b}}".replace("{{a}}", " {{a}}")) is None


def test_drops_empty_and_whitespace():
    assert clean_paragraph("") is None
    assert clean_paragraph("   \n  ") is None


def test_min_chars_is_opt_in():
    short = "Too short."
    assert clean_paragraph(short) == short
    assert clean_paragraph(short, min_chars=70) is None
