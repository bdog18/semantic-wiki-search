import html
import json
import os
import re

from lxml import etree
from tqdm import tqdm

from swsearch.extract.wikitext import clean_paragraph
from swsearch.logutil import get_logger

logger = get_logger(__name__)

# Below this, a "paragraph" is almost always a bare MediaWiki section header
# ("History.", "Career.", "Personal life.") rather than real content -- a
# 200-file corpus sample found 31% of paragraphs under 40 chars, dominated by
# a few hundred repeated header strings, while the 70-100 char range was
# already ~90%+ genuine sentences. Filtered here so these never get embedded,
# indexed, or mined as triplets in the first place.
MIN_PARAGRAPH_CHARS = 70


def clean_text(text: str) -> str:
    """Decode HTML entities, drop citation/markup noise, and normalize
    whitespace down to one line per paragraph."""
    text = html.unescape(text)
    text = re.sub(r'<ref\b[^>]*>.*?</ref>', ' ', text, flags=re.IGNORECASE | re.DOTALL)  # cited/reference content, not article prose
    text = re.sub(r'<[^>]+>', ' ', text)  # any other leftover HTML/XML tags (<br>, <section>, stray dump metadata, ...)
    text = re.sub(r'\s*\n\s*\n\s*', '\n', text)  # normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # normalize spaces
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text.strip()


def parse_docstring(file_path: str) -> list[dict]:
    """
    Parse a single XML fragment file containing <doc> elements from Wikipedia.
    Wraps the content in a root tag to handle malformed XML and extracts entries
    with at least 100 words.
    """
    try:
        with open(file_path, 'rb') as file:
            file_content = file.read()

        wrapped_content = b"<root>" + file_content + b"</root>"
        tree = etree.fromstring(wrapped_content)

        doc_info = []
        for doc in tree.findall('doc'):
            title = doc.get('title')
            raw = doc.text.strip() if doc.text else ""
            lines = [line for line in clean_text(raw).split("\n") if line]

            # WikiExtractor emits the article title as its own leading line
            # before a blank line and the body. Left in, it survives
            # paragraph-splitting downstream (embed_paragraphs, mine_triplets)
            # as its own "paragraph" that's a near-perfect lexical match for
            # title-like queries, so it wins the per-article dedup in search
            # over real content. Strip it here, once, at the shared source.
            if lines and title and lines[0] == title.strip():
                lines = lines[1:]

            # Drops the residual MediaWiki markup WikiExtractor leaves behind
            # -- file captions, unexpanded templates, table rows -- and
            # unwraps inline links in the prose that survives, folding in the
            # length filter. Applied here so a rebuild from raw never indexes
            # markup in the first place; extract.wikitext documents how much
            # of it there is and why so little of it does so much damage.
            cleaned = (clean_paragraph(line, min_chars=MIN_PARAGRAPH_CHARS) for line in lines)
            lines = [line for line in cleaned if line is not None]
            content = "\n\n".join(lines)
            word_count = len(content.split())

            if word_count >= 100:
                doc_info.append({
                    'id': doc.get('id'),
                    'url': doc.get('url'),
                    'title': title,
                    'content': content,
                })

        return doc_info

    except Exception:
        logger.exception("Error processing %s", file_path)
        return []


def save_json(file_path: str, data: list[dict]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def traverse_directory(input_dir: str, output_dir: str) -> None:
    """
    Recursively traverse a directory tree of XML fragment files and write
    cleaned JSON files to a corresponding output structure.
    """
    if not os.path.exists(input_dir):
        logger.error("Input directory does not exist: %s", input_dir)
        return

    all_files = []
    for root_dir, _, files in os.walk(input_dir):
        for file in files:
            all_files.append(os.path.join(root_dir, file))

    for file_path in tqdm(all_files, desc="Processing XML files", unit="file"):
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        file_name = os.path.basename(file_path)

        doc_data = parse_docstring(file_path)

        if relative_path == ".":
            output_file_path = os.path.join(output_dir, file_name + '.json')
        else:
            output_file_path = os.path.join(output_dir, relative_path, file_name + '.json')

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        save_json(output_file_path, doc_data)


def convert_json_array_to_jsonl(input_dir: str, output_dir: str) -> None:
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))

    for input_path in tqdm(all_files, desc="Converting JSON to JSONL", unit="file"):
        rel_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, os.path.basename(input_path) + "l")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    with open(output_path, "w", encoding="utf-8") as out:
                        for obj in data:
                            json.dump(obj, out, ensure_ascii=False)
                            out.write("\n")
        except Exception:
            logger.exception("Skipped %s", input_path)


def save_article_titles(data_dir: str, output_path: str) -> None:
    article_titles = {}
    files = [
        os.path.join(root, file)
        for root, _, filenames in os.walk(data_dir)
        for file in filenames if file.endswith(".jsonl")
    ]
    for file_path in tqdm(files, desc="Loading Article Titles From Files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line)
                    title = article.get("title")
                    if title:
                        article_titles[title] = article.get("url", "")
                except json.JSONDecodeError:
                    logger.exception("Skipping malformed JSONL line in %s", file_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(article_titles, f)
