# Semantic Wikipedia Search Engine (In Progress)

**Project Status**: In Progress
**Goal**: Build a semantic search engine over a Wikipedia dump: extract and clean
articles, build a link graph, mine hard-negative triplets, and serve semantic
search over a FAISS index.

This project extracts, cleans, and embeds Wikipedia content; builds a link graph
from a MySQL dump for use as positive/negative signal; mines hard-negative
triplets by combining the link graph with FAISS nearest-neighbor search; and
serves semantic search over paragraph-level embeddings with cosine reranking.
Training a custom encoder was explored but is deprioritized in favor of the
working off-the-shelf embedding pipeline (see **Research** below).

---

## Key Features (Completed)

- Extract and clean a Wikipedia XML dump into paragraph-level JSON/JSONL
- Build a link graph from MySQL SQL dumps, stored in SQLite for fast lookup
- Embed paragraphs (SentenceTransformer) and build a flat FAISS index, with
  metadata + a manifest written together so the index can never silently drift
  out of alignment with its metadata (previously a real bug: a directory-write
  vs. single-file mismatch, and no guaranteed correspondence between `.npy`
  batch files and metadata order)
- Mine hard-negative triplets in parallel, using the link graph for positives
  and FAISS nearest-neighbors (excluding linked/same-article paragraphs) for
  negatives
- Real semantic search: embed query -> FAISS candidates -> cosine rerank ->
  dedupe to one best-scoring result per article
- A real evaluation harness (Top-K Accuracy, Precision@K, Recall@K, MRR)
  wired to the actual search engine (previously a hardcoded stub that ignored
  its query argument entirely)
- An installable `src/swsearch` package with a `swsearch` CLI covering the
  whole pipeline (see **CLI** below), replacing the old `utils/` + notebook
  orchestration and its `sys.path.append` hacks
- End-to-end pipeline (extract -> embed -> build-index -> search -> evaluate)
  verified at small scale on an isolated scratch corpus (~66 articles), with
  the index/metadata alignment assertion passing and sensible search results

---

## Research (Deprioritized)

- `research/custom_encoder.py` (archived from `utils/custom_embedder.py`): a
  custom transformer encoder trained with triplet loss, meant to eventually
  replace the off-the-shelf SentenceTransformer model. Training was never
  completed (no saved weights or vectorizer exist), and it is **not** wired
  into the `swsearch` CLI. Kept outside `src/swsearch/` so TensorFlow/Keras
  stay an opt-in research dependency (`requirements-research.txt`) rather than
  a hard dependency of the installable package.

---

## In Progress / Next Steps

- Rebuild the production FAISS index + metadata store with the new
  manifest-based pipeline (the old `data/processed/faiss_index/` artifacts
  predate this refactor and are being kept until that's a deliberate,
  separate action -- see note below)
- Reprocess the full ~10M-article corpus through the new pipeline
- Improve reranking using article-level metadata beyond cosine similarity
- Expand the Gradio demo beyond a functional stub
- Revisit the custom encoder once the off-the-shelf pipeline's results are
  well understood

> **Note on existing data**: `data/processed/faiss_index/` currently holds
> ~120GB of output from the *pre-refactor* pipeline, including the
> `EMBEDDING_DIR`/`EMBEDDINGS_DIR` path-drift bug that produced 65,250
> duplicate `.npy` files, and `data/custom_model/` holds another ~97GB of
> stale pre-refactor artifacts nothing in the new pipeline reads or writes.
> `swsearch pipeline` (see **CLI** below) can now regenerate everything under
> `data/processed/` from `data/raw/` (which stays untouched) in one command,
> so both directories are safe to delete once a fresh pipeline run is
> verified good -- this isn't done automatically, it's a deliberate,
> separate, full-scale action for whenever that verification has happened.
> Always point `SWSEARCH_PATHS__DATA_ROOT` (or per-command path flags) at a
> scratch location when testing against a subset of data.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Main dependencies are in `pyproject.toml` (installed automatically). TensorFlow/
Keras (only needed for the archived `research/custom_encoder.py`) are kept out
of the base install; add them with:

```bash
pip install -r requirements-research.txt
```

**GPU**: `pip install -e .` pulls in the default (CPU-only) `torch` build even
on a machine with an NVIDIA GPU. `swsearch.config` auto-detects
`torch.cuda.is_available()` and only resolves `device` to `cuda` if a
CUDA-enabled `torch` build is actually installed -- otherwise `embed`/`search`/
`mine-triplets` silently run on CPU. To use a GPU, reinstall `torch` from the
CUDA wheel index matching your driver (check `nvidia-smi` for the supported
CUDA version, then find the newest matching tag at
https://download.pytorch.org/whl/ -- `cu126` matched an RTX 3070 Ti / driver
supporting CUDA 13.3 as of this writing):

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

Don't add `--no-deps` here -- torch's CUDA build depends on separate
`nvidia-*-cu12` packages that provide the actual `.so` libraries; skipping
them breaks `import torch` entirely, not just GPU detection.

---

## Configuration

Settings are managed by `pydantic-settings` (`src/swsearch/config.py`), overridable
via environment variables or a `.env` file (see `.env.example`). All variables use
the `SWSEARCH_` prefix with `__` separating nested groups, e.g.:

```bash
SWSEARCH_PATHS__DATA_ROOT=/mnt/E/Repos/semantic-wiki-search/data
SWSEARCH_MODEL__EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
SWSEARCH_MODEL__USE_GPU_FAISS=false
SWSEARCH_SPARK__DRIVER_MEMORY=50g
SWSEARCH_MINING__MAX_TRIPLETS_PER_ARTICLE=10
```

All paths resolve to absolute paths (anchored to the repo root by default), so
`swsearch` commands work from any working directory.

---

## CLI

```
swsearch pipeline         [--with-triplets] [--skip-fetch] [--index-type flat|ivfpq]
swsearch extract          [--input-dir] [--output-json-dir] [--output-jsonl-dir] [--article-titles-path]
swsearch build-linkgraph  [--page-sql] [--pagelinks-sql] [--linktarget-sql] [--jsonl-out] [--db-out]
swsearch embed            [--input-dir] [--output-dir] [--meta-db] [--model-name] [--batch-size]
swsearch build-index      [--embeddings-dir] [--index-path] [--meta-db] [--index-type flat|ivfpq]
swsearch mine-triplets    [--jsonl-dir] [--index-path] [--meta-db] [--link-db] [--out-dir] [--num-workers]
swsearch search QUERY     [--k] [--index-path] [--meta-db]
swsearch evaluate         [--test-queries] [--k-values "1,3,5,10"]
swsearch tools inspect-index INDEX_PATH
swsearch tools convert-faiss-meta [--json-path] [--db-path] [--yes]
```

Every command has sensible defaults derived from `Settings` (all under
`data/` by default) but every path can be overridden per-command, so the
pipeline can be run against an isolated scratch corpus for testing without
touching real data:

```bash
SWSEARCH_PATHS__DATA_ROOT=/tmp/scratch-corpus swsearch extract
SWSEARCH_PATHS__DATA_ROOT=/tmp/scratch-corpus swsearch embed
SWSEARCH_PATHS__DATA_ROOT=/tmp/scratch-corpus swsearch build-index
SWSEARCH_PATHS__DATA_ROOT=/tmp/scratch-corpus swsearch search "who invented science?"
```

### `swsearch pipeline`: run everything end to end

`swsearch pipeline` chains every stage above into one command: fetch raw data
if it isn't already on disk (downloads the XML dump + the 3 SQL link-graph
dumps from dumps.wikimedia.org and runs `wikiextractor`, skipping anything
already present rather than redoing it) -> build the link graph -> extract ->
embed -> build the index. Triplet mining is opt-in (`--with-triplets`) since
nothing in the search/evaluate path consumes it -- it only feeds the archived
custom-encoder research.

```bash
swsearch pipeline                 # fetch-if-missing -> linkgraph -> extract -> embed -> index
swsearch pipeline --skip-fetch    # assume data/raw is already fully populated
swsearch pipeline --with-triplets # also mine triplets at the end
```

Because every raw-data step (`fetch.py`) checks for its expected output
first, deleting `data/processed/` and `data/custom_model/` and rerunning
`swsearch pipeline` regenerates everything from `data/raw/` (which is left
untouched) without re-downloading anything that's already there. This is the
safe way to reclaim disk space from stale pre-refactor artifacts.

The pipeline can run for hours end to end (a full-corpus embed/mine pass
especially), so each stage logs a start banner with total elapsed time and a
finish banner with that stage's own duration and how many stages remain --
e.g. `=== Stage 4/6: embed paragraphs [pipeline elapsed: 42m10s] ===` --
independent of the item-by-item `tqdm` progress bars each stage already
prints for its own work.

---

## Example Use Case

**Query**: "Who was involved in World War 2?"
The system returns articles that are contextually relevant to the query, even if
the wording differs from the Wikipedia article's title.

---

## Project Structure

```
semantic-wiki-search/
├── pyproject.toml                # installable package, `swsearch` console script
├── requirements.txt               # main runtime dependencies
├── requirements-research.txt      # TensorFlow/Keras, for research/custom_encoder.py only
├── .env.example
├── src/swsearch/
│   ├── config.py                  # pydantic-settings singleton (SWSEARCH_ env prefix)
│   ├── cli.py                     # typer app: swsearch <command>
│   ├── pipeline.py                # `swsearch pipeline`: chains every stage, with progress/timing
│   ├── fetch.py                   # idempotent raw-data fetch (dump/SQL downloads, wikiextractor)
│   ├── logutil.py                 # logging setup
│   ├── common/                    # shared Spark session + paragraph-split helpers
│   ├── extract/wikidump.py        # XML dump -> cleaned JSON/JSONL + title->url lookup
│   ├── linkgraph/                 # build.py (SQL dump -> link graph), store.py (SQLite lookup)
│   ├── embed/paragraphs.py        # paragraph embedding + metadata/manifest writer
│   ├── metadata/store.py          # FAISS metadata SQLite store
│   ├── index/faiss_store.py       # FAISS index build/load/query (manifest-driven, alignment-checked)
│   ├── mining/triplets.py         # parallel hard-negative triplet mining
│   ├── search/engine.py           # SearchEngine: embed -> FAISS -> cosine rerank -> dedupe
│   ├── eval/metrics.py            # Top-K/Precision/Recall/MRR
│   └── tools/inspect_index.py     # FAISS index diagnostics
├── research/custom_encoder.py     # archived, deprioritized custom encoder training
├── tests/                         # minimal smoke suite (imports + alignment invariant)
├── notebooks/
│   ├── search_demo.ipynb          # working search demo, imports swsearch directly
│   └── archive/main.ipynb         # old 14-step orchestrator, superseded by the CLI
├── gradio/gradio_app.py           # functional Gradio demo wired to SearchEngine
├── data/                          # Wikipedia dump files and processed artifacts (gitignored)
└── wikiextractor-master/          # vendored wikiextractor
```

---

## Technologies Used

- SentenceTransformers (paragraph/query embeddings)
- FAISS (vector search and indexing)
- PySpark (dump parsing, link graph, paragraph splitting)
- SQLite (link graph and FAISS metadata lookup)
- pydantic-settings + typer (configuration and CLI)
- TensorFlow / Keras (archived custom encoder research only)

---

## Evaluation Metrics

- Top-K Accuracy
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

Verified against a real `SearchEngine` at small scale (see **Key Features**
above); full-corpus evaluation numbers are not yet published.

---

## Future Work

- Reprocess the full enwiki dataset (10M+ articles) through the new pipeline
- Refine negative sampling for better contrastive training
- Support multilingual Wikipedia input
- Deploy the index behind an API for public search

---

## Notes

This is an independent project exploring end-to-end semantic search using
Wikipedia as a dataset. It is actively in development and intended for
demonstration, experimentation, and research purposes.
