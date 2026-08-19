"""Residual MediaWiki markup handling, at paragraph granularity.

WikiExtractor strips the bulk of wikitext, but a small tail survives into the
extracted corpus. Measured over a 50,000-paragraph random sample of the
41.95M-paragraph baseline index: ~0.024% of paragraphs still carry `[[...]]`
markup, ~0.008% carry `{{...}}`, and ~0.002% are file captions.

Those rates understate the damage. The survivors are overwhelmingly short and
dense with proper nouns -- exactly the shape that scores well against a
name-like query -- so they surface far above their corpus share: across 20
representative queries at k=5, 1% of returned results were raw markup, a ~40x
over-representation. `[[File:Hamlet Winstanley Self-Portrait.jpg|thumb|140px|
[[Hamlet Winstanley]], 1730, Self-Portrait]]` ranked #2 for "Who wrote
Hamlet?" purely on the strength of the token "Hamlet".

The central distinction is between a paragraph that *is* markup and prose that
merely *contains* it. Only the former is dropped. Treating them alike costs
real content: an article body carrying an inline {{convert}} or a malformed
[[stty|]] is still an article body, and an early draft of this module threw
away three such paragraphs per fifty thousand for that reason.

Kept separate from wikidump.py so callers that only need to clean text --
notably a metadata migration reading rows back out of an already-built index
-- don't pull in lxml and the dump-parsing machinery to do it.
"""

import re

# A paragraph *leading* with any of these is furniture, not prose: a file
# caption, a table row/header/opener, an unexpanded template block, or a bare
# section heading. Anchored deliberately -- the same constructs appearing mid
# paragraph are inline noise inside real text and get stripped instead.
_LEADING_FURNITURE_RE = re.compile(
    r"""^\s*(?:
          \[\[\s*(?:File|Image|Media)\s*:   # [[File:...]] caption
        | [!|]                              # ! header cell / | table cell
        | \{\|                              # {| table opener
        | \{\{                              # {{template}} block
        | ={2,}                             # == heading ==
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_MEDIA_START_RE = re.compile(r"\[\[\s*(?:File|Image|Media)\s*:", re.IGNORECASE)
_TEMPLATE_START_RE = re.compile(r"\{\{")

# Display text is optional and may be empty: [[stty|]] must resolve to "stty"
# rather than failing to match and stranding brackets in the output.
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]*?)(?:\|([^\[\]]*?))?\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]*(?:\s+([^\]]*))?\]")
_QUOTE_MARKUP_RE = re.compile(r"'{2,5}")
_STRAY_DELIMITER_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|\{\||\|\}")


def _remove_nested(text: str, start_re: re.Pattern, open_tok: str, close_tok: str) -> str:
    """Remove spans opened by start_re and closed by their *matching* close
    token, respecting nesting.

    A plain non-greedy regex stops at the first close token, which for
    `[[File:x.jpg|thumb|[[Ernst Middendorp]] became the manager]]` is the
    inner link's -- leaving `became the manager]]` behind as fake prose. An
    unbalanced span (truncated markup) is dropped through to end of string.
    """
    out: list[str] = []
    i = 0
    while (match := start_re.search(text, i)) is not None:
        out.append(text[i:match.start()])
        depth, j = 0, match.start()
        while j < len(text):
            if text.startswith(open_tok, j):
                depth += 1
                j += len(open_tok)
            elif text.startswith(close_tok, j):
                depth -= 1
                j += len(close_tok)
                if depth == 0:
                    break
            else:
                j += 1
        if depth != 0:
            return "".join(out)
        i = j
    out.append(text[i:])
    return "".join(out)


def is_furniture(text: str) -> bool:
    """True if the paragraph is MediaWiki furniture rather than prose.

    A drop test, not a strip test. A file caption with its brackets removed
    is still a caption and a table row is still a table row -- stripping them
    would leave short, proper-noun-dense fragments that keep the ranking
    problem while hiding its cause.
    """
    return bool(_LEADING_FURNITURE_RE.match(text))


def strip_markup(text: str) -> str:
    """Remove the markup that appears *inside* otherwise-real prose.

    Embedded captions and templates are removed outright (their content isn't
    part of the sentence they interrupt); links are unwrapped to their display
    text; quote runs and any stray unmatched delimiters are dropped.
    """
    text = _remove_nested(text, _MEDIA_START_RE, "[[", "]]")
    text = _remove_nested(text, _TEMPLATE_START_RE, "{{", "}}")

    previous = None
    # Innermost-first: the link pattern can't match across a nested pair, and
    # each pass either removes a pair or changes nothing, so this terminates.
    while previous != text:
        previous = text
        text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)

    text = _EXTERNAL_LINK_RE.sub(lambda m: m.group(1) or "", text)
    text = _QUOTE_MARKUP_RE.sub("", text)
    text = _STRAY_DELIMITER_RE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_paragraph(text: str, min_chars: int = 0) -> str | None:
    """Return display-ready prose, or None if the paragraph should be dropped.

    min_chars defaults to 0 (no length filtering) so callers keep ownership of
    that threshold -- extraction applies wikidump.MIN_PARAGRAPH_CHARS, while a
    migration over already-indexed rows may want to preserve whatever was
    indexed regardless of length.
    """
    if not text or not text.strip():
        return None
    if is_furniture(text):
        return None

    cleaned = strip_markup(text)

    # Stripping can hollow out a paragraph that was mostly markup without
    # leading with it. Length is the caller's call, not this function's --
    # an absolute floor here silently overrode min_chars and dropped short
    # real sentences -- so only a paragraph stripped to nothing is refused
    # unconditionally; anything shorter than the caller cares about is
    # theirs to reject via min_chars.
    if not cleaned:
        return None
    if len(cleaned) < min_chars:
        return None
    return cleaned
