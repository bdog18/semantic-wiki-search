import re

# One paragraph-split pattern shared by every producer/consumer of article
# text: data_prep/wikidump.py writes paragraphs separated by a blank line,
# and this is the single place both plain-Python splitting (mining, eval)
# and Spark's regexp-based split() column function read that convention from.
PARAGRAPH_SPLIT_PATTERN = r"\n\s*\n"

_PARAGRAPH_SPLIT_RE = re.compile(PARAGRAPH_SPLIT_PATTERN)


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
