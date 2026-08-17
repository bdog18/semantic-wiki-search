from pathlib import Path
from typing import Optional

import typer

from swsearch import __version__
from swsearch.config import settings
from swsearch.logutil import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Semantic Wikipedia search: extract -> link graph -> embed -> index -> search.")
tools_app = typer.Typer(help="Diagnostic and maintenance tools.")
app.add_typer(tools_app, name="tools")


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show the swsearch version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def pipeline(
    with_triplets: bool = typer.Option(
        False, "--with-triplets", help="Also mine hard-negative triplets after building the index "
        "(slow; only needed for the archived custom-encoder research -- search/evaluate don't use it)."
    ),
    skip_fetch: bool = typer.Option(
        False, "--skip-fetch", help="Skip checking/downloading raw dumps and running wikiextractor; "
        "assume data/raw is already fully populated."
    ),
    index_type: str = typer.Option("flat", help="Index type: 'flat' (default, proven path) or 'ivfpq'."),
) -> None:
    """Run the full pipeline end to end: fetch raw data if missing (download
    the XML/SQL dumps and run wikiextractor, skipping anything already on
    disk) -> build link graph -> extract -> embed -> build index ->
    optionally mine triplets."""
    from swsearch.pipeline import run_full_pipeline

    run_full_pipeline(with_triplets=with_triplets, skip_fetch=skip_fetch, index_type=index_type)
    typer.echo("Full pipeline complete.")


@app.command()
def extract(
    input_dir: Path = typer.Option(settings.paths.extracted_dir, help="Directory of raw wikiextractor XML fragments."),
    output_json_dir: Path = typer.Option(settings.paths.json_dir, help="Where to write cleaned per-article JSON."),
    output_jsonl_dir: Path = typer.Option(settings.paths.jsonl_dir, help="Where to write JSONL converted from the JSON output."),
    article_titles_path: Path = typer.Option(settings.paths.article_titles_path, help="Where to write the title -> url lookup."),
) -> None:
    """Clean raw wikiextractor XML fragments into JSON, then JSONL, then a title->url lookup."""
    from swsearch.extract.wikidump import convert_json_array_to_jsonl, save_article_titles, traverse_directory

    traverse_directory(str(input_dir), str(output_json_dir))
    convert_json_array_to_jsonl(str(output_json_dir), str(output_jsonl_dir))
    save_article_titles(str(output_jsonl_dir), str(article_titles_path))
    typer.echo(f"Extracted articles to {output_json_dir} and {output_jsonl_dir}")


@app.command()
def build_linkgraph(
    page_sql: Path = typer.Option(settings.paths.page_sql_path, help="enwiki-latest-page.sql dump."),
    pagelinks_sql: Path = typer.Option(settings.paths.pagelinks_sql_path, help="enwiki-latest-pagelinks.sql dump."),
    linktarget_sql: Path = typer.Option(settings.paths.linktarget_sql_path, help="enwiki-latest-linktarget.sql dump."),
    jsonl_out: Path = typer.Option(settings.paths.link_graph_jsonl_dir, help="Where to write the Spark JSONL link graph."),
    db_out: Path = typer.Option(settings.paths.link_graph_db_path, help="Where to write the SQLite link graph lookup."),
) -> None:
    """Parse MySQL dump tables into a from_title -> linked_titles link graph."""
    from swsearch.linkgraph.build import export_link_graph_to_jsonl
    from swsearch.linkgraph.store import build_linkgraph_sqlite

    export_link_graph_to_jsonl(str(page_sql), str(pagelinks_sql), str(linktarget_sql), str(jsonl_out))
    skipped = build_linkgraph_sqlite(str(jsonl_out), str(db_out))
    typer.echo(f"Link graph written to {db_out} ({skipped} malformed row(s) skipped)")


@app.command()
def build_backlinks(
    db_in: Path = typer.Option(settings.paths.link_graph_db_path, help="Existing link graph sqlite db."),
    db_out: Path = typer.Option(settings.paths.backlink_counts_db_path, help="Where to write the SQLite backlink count lookup."),
) -> None:
    """Build a SQLite database containing backlink counts for each page in the link graph."""
    from swsearch.linkgraph.backlinks import build_backlink_counts_sqlite

    build_backlink_counts_sqlite(str(db_in), str(db_out))
    typer.echo(f"Backlink counts written to {settings.paths.backlink_counts_db_path}")


@app.command()
def embed(
    input_dir: Path = typer.Option(settings.paths.json_dir, help="Directory of cleaned article JSON to embed."),
    output_dir: Path = typer.Option(settings.paths.embeddings_dir, help="Where to write per-batch embedding .npy files."),
    meta_db: Path = typer.Option(settings.paths.faiss_meta_db_path, help="Where to write the FAISS metadata SQLite store."),
    model_name: str = typer.Option(settings.model.embedding_model_name, help="SentenceTransformer model name."),
    batch_size: int = typer.Option(4096, help="Paragraphs accumulated (and length-sorted for padding-efficient GPU batching) before each encode/save/write cycle -- not the GPU batch size itself, see encode-batch-size."),
    encode_batch_size: int = typer.Option(256, help="Paragraphs encoded together in one GPU forward pass. Paired with batch_size=4096, this measured as the fastest full-corpus embed run on record (~3.4k paragraphs/sec) -- see embed_paragraphs's docstring for the comparison data."),
) -> None:
    """Split articles into paragraphs and embed them, writing metadata + a
    manifest alongside the embeddings so index building can never drift out
    of alignment with them."""
    from swsearch.embed.paragraphs import embed_paragraphs

    total = embed_paragraphs(
        data_dir=str(input_dir),
        embeddings_dir=str(output_dir),
        meta_db_path=str(meta_db),
        model_name=model_name,
        batch_size=batch_size,
        encode_batch_size=encode_batch_size,
        device=settings.model.device,
    )
    typer.echo(f"Embedded {total} paragraphs into {output_dir}")


@app.command()
def build_index(
    embeddings_dir: Path = typer.Option(settings.paths.embeddings_dir, help="Directory of .npy embedding batches from `swsearch embed`."),
    index_path: Path = typer.Option(settings.paths.faiss_index_path, help="Where to write the FAISS index."),
    meta_db: Path = typer.Option(settings.paths.faiss_meta_db_path, help="FAISS metadata SQLite store matching embeddings_dir (from `swsearch embed`)."),
    index_type: str = typer.Option("flat", help="Index type: 'flat' (default, proven path) or 'ivfpq'."),
) -> None:
    """Build a FAISS index from embed's output, in the exact order recorded
    in its manifest, and verify it stays aligned with the metadata store."""
    from swsearch.index.faiss_store import build_flat_index_from_manifest, build_ivfpq_index_from_manifest
    from swsearch.metadata.store import load_faiss_meta_sqlite

    meta_conn = load_faiss_meta_sqlite(str(meta_db))
    if index_type == "flat":
        build_flat_index_from_manifest(str(embeddings_dir), meta_conn, str(index_path))
    elif index_type == "ivfpq":
        build_ivfpq_index_from_manifest(str(embeddings_dir), meta_conn, str(index_path))
    else:
        raise typer.BadParameter("index_type must be 'flat' or 'ivfpq'")
    typer.echo(f"FAISS index ({index_type}) written to {index_path}")


@app.command()
def mine_triplets(
    jsonl_dir: Path = typer.Option(settings.paths.jsonl_dir, help="Directory of article JSONL to mine triplets from."),
    index_path: Path = typer.Option(settings.paths.faiss_index_path, help="FAISS index used to find hard negatives."),
    meta_db: Path = typer.Option(settings.paths.faiss_meta_db_path, help="FAISS metadata SQLite store matching index_path."),
    link_db: Path = typer.Option(settings.paths.link_graph_db_path, help="Link graph SQLite store."),
    out_dir: Path = typer.Option(settings.paths.triplets_dir, help="Where to write mined triplet JSONL files."),
    num_workers: Optional[int] = typer.Option(None, help="Worker processes (default: min(cpu_count // 2, 2))."),
) -> None:
    """Mine hard-negative (anchor, positive, negative) triplets using the link
    graph for positives and FAISS nearest-neighbors for negatives."""
    from swsearch.mining.triplets import mine_triplets as _mine_triplets

    total = _mine_triplets(
        jsonl_dir=str(jsonl_dir),
        index_path=str(index_path),
        meta_db_path=str(meta_db),
        link_db_path=str(link_db),
        article_titles_path=str(settings.paths.article_titles_path),
        out_dir=str(out_dir),
        num_workers=num_workers,
    )
    typer.echo(f"Mined {total} triplets into {out_dir}")


@app.command("train-transfer")
def train_transfer(
    triplets_dir: Path = typer.Option(settings.paths.triplets_dir, help="Directory of mined triplet JSONL files (from `swsearch mine-triplets`)."),
    output_dir: Path = typer.Option(settings.paths.transfer_model_dir, help="Where to save the fine-tuned model."),
    base_model_name: str = typer.Option(settings.model.embedding_model_name, help="Pretrained SentenceTransformer to fine-tune."),
    batch_size: int = typer.Option(16, help="Training micro-batch size (kept small -- training needs backward-pass activations, unlike inference; see gradient-accumulation-steps for effective batch size)."),
    gradient_accumulation_steps: int = typer.Option(4, help="Accumulate gradients over this many micro-batches (effective batch size = batch_size * this)."),
    max_steps: int = typer.Option(20000, help="Total training steps (triplets stream, so there's no fixed epoch count)."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate."),
    scale: float = typer.Option(20.0, help="MultipleNegativesRankingLoss softmax temperature (higher = sharper)."),
    warmup_ratio: float = typer.Option(0.1, help="Fraction of steps to ramp the learning rate up over before it hits full strength."),
    freeze_layers: int = typer.Option(3, help="Freeze the embeddings + this many bottom transformer layers (out of 6 for the default base model); 0 disables freezing."),
) -> None:
    """Fine-tune a pretrained SentenceTransformer on mined triplets
    (transfer learning) with MultipleNegativesRankingLoss, keeping whichever
    checkpoint scores best on a held-out triplet split. Reuses the triplets
    mined once against the baseline index -- no separate mining run is
    needed for this. The saved model can then be passed to
    `swsearch embed --model-name <output_dir>`."""
    from swsearch.train import train_transfer_model

    train_transfer_model(
        triplets_dir=str(triplets_dir),
        output_dir=str(output_dir),
        base_model_name=base_model_name,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        scale=scale,
        warmup_ratio=warmup_ratio,
        freeze_layers=freeze_layers,
    )
    typer.echo(f"Fine-tuned model saved to {output_dir}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(5, help="Number of results to return."),
    index_path: Path = typer.Option(settings.paths.faiss_index_path, help="FAISS index to search."),
    meta_db: Path = typer.Option(settings.paths.faiss_meta_db_path, help="FAISS metadata SQLite store matching index_path."),
    model_name: str = typer.Option(settings.model.embedding_model_name, help="Model to encode the query with -- must match whatever model produced index_path's embeddings (override for non-baseline indexes, e.g. a fine-tuned model's local directory)."),
    rerank: bool = typer.Option(settings.rerank.enabled, "--rerank/--no-rerank", help="Rerank by title-match + backlink count instead of plain cosine similarity."),
) -> None:
    """Search the index and print the top-k articles."""
    from swsearch.search.engine import SearchEngine

    engine = SearchEngine(index_path=str(index_path), meta_db_path=str(meta_db), model_name=model_name, rerank_enabled=rerank)
    for rank, result in enumerate(engine.search(query, k=k), start=1):
        typer.echo(f"{rank}. {result['title']}  (score={result['score']:.4f})  {result['url']}")


@app.command()
def evaluate(
    test_queries: Path = typer.Option(settings.paths.test_queries_path, help="JSON file of {query, relevant_articles} entries."),
    k_values: str = typer.Option("1,3,5,10", help="Comma-separated K values for Top-K/Precision/Recall."),
    index_path: Path = typer.Option(settings.paths.faiss_index_path, help="FAISS index to evaluate (override to compare a different model's index)."),
    meta_db: Path = typer.Option(settings.paths.faiss_meta_db_path, help="FAISS metadata SQLite store matching index_path."),
    model_name: str = typer.Option(settings.model.embedding_model_name, help="Model to encode queries with -- must match whatever model produced index_path's embeddings (override for non-baseline indexes, e.g. a fine-tuned model's local directory)."),
    rerank: bool = typer.Option(settings.rerank.enabled, "--rerank/--no-rerank", help="Rerank by title-match + backlink count instead of plain cosine similarity."),
) -> None:
    """Run Top-K Accuracy, Precision@K, Recall@K, and MRR against a real SearchEngine."""
    from swsearch.eval.metrics import evaluate_all_metrics, load_test_set
    from swsearch.search.engine import SearchEngine

    ks = tuple(int(k.strip()) for k in k_values.split(","))
    engine = SearchEngine(index_path=str(index_path), meta_db_path=str(meta_db), model_name=model_name, rerank_enabled=rerank)
    retrieval_function = lambda q: [r["title"] for r in engine.search(q, k=max(ks))]  # noqa: E731

    test_set = load_test_set(str(test_queries))
    results = evaluate_all_metrics(test_set, retrieval_function, ks=ks)
    for metric, value in results.items():
        typer.echo(f"{metric}: {value}")


@tools_app.command("inspect-index")
def tools_inspect_index(
    index_path: Path = typer.Argument(..., help="Path to a FAISS index file."),
) -> None:
    """Print type/size/GPU-memory diagnostics for a FAISS index file."""
    from swsearch.tools.inspect_index import inspect_index

    inspect_index(str(index_path))


@tools_app.command("convert-faiss-meta")
def tools_convert_faiss_meta(
    json_path: Path = typer.Option(..., help="Existing paragraphs.index.meta.json to convert."),
    db_path: Path = typer.Option(..., help="Output SQLite database path."),
    yes: bool = typer.Option(False, "--yes", help="Overwrite db_path without prompting."),
) -> None:
    """One-time conversion of a legacy FAISS meta.json file into the SQLite metadata store."""
    from swsearch.metadata.store import convert_faiss_meta_json_to_sqlite

    if db_path.exists() and not yes:
        if not typer.confirm(f"{db_path} already exists. Overwrite?"):
            typer.echo("Conversion cancelled.")
            raise typer.Exit()

    convert_faiss_meta_json_to_sqlite(str(json_path), str(db_path))
    typer.echo(f"Converted {json_path} -> {db_path}")
