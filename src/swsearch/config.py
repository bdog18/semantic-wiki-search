from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/swsearch/config.py -> repo root is two levels up from this file's package dir.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_device() -> str:
    """cuda if a GPU is visible to torch, else cpu. Checked once (Settings is
    a cached singleton via get_settings()), not hardcoded -- so `swsearch
    embed`/`swsearch pipeline` use a GPU automatically wherever one's
    available instead of defaulting to CPU. Still overridable via
    SWSEARCH_MODEL__DEVICE for anyone who wants to force one or the other."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SparkSettings(BaseModel):
    driver_memory: str = "8g"
    max_result_size: str = "4g"
    shuffle_partitions: int = 100
    local_dir: str = "spark-temp"


class ModelSettings(BaseModel):
    embedding_model_name: str = "all-MiniLM-L6-v2"
    device: str = Field(default_factory=_default_device)
    use_gpu_faiss: bool = False
    
    
class RerankSettings(BaseModel):
    enabled: bool = False
    title_match_weight: float = 0.5
    backlink_weight: float = 0.3


class MiningSettings(BaseModel):
    max_triplets_per_article: int = 10
    # Negatives are sampled from a *band* of the anchor's nearest neighbours
    # (ranks [negative_rank_min, negative_rank_max)), not from the top of the
    # list. Drawing from the top-10 -- the previous behaviour -- means the
    # "negatives" are literally the most relevant paragraphs in the corpus,
    # and the link graph is far too weak a filter to catch that.
    #
    # Measured over 250 anchors against the clean baseline index, with mean
    # cos(anchor, positive) = 0.4744 as the bar a negative must sit below:
    #   ranks      0-10   cos 0.5562   61.2% inverted   <- previous default
    #   ranks    200-600  cos 0.4194   42.0% inverted
    #   ranks   1000-2000 cos 0.3818   32.4% inverted   <- current default
    #   ranks  5000-10000 cos 0.3368   30.0% inverted
    # 1000-2000 is the knee: comfortably below the positive bar, but still
    # far harder than a random paragraph (~0.05), so the loss keeps a real
    # discrimination task. Going deeper buys little and costs hardness.
    negative_rank_min: int = 1000
    negative_rank_max: int = 2000
    # How many candidates to draw out of that band per anchor. Kept at the
    # old pool size so the per-batch SQLite lookup volume is unchanged --
    # only the ranks the candidates come from moved.
    negative_pool_size: int = 10
    # IVF cells probed when searching for negatives. The index ships with
    # nprobe=10, which is fine for top-10 retrieval but too narrow to rank a
    # 600-deep band reliably; 32 is what the rank-band measurements above
    # were taken at.
    negative_search_nprobe: int = 32
    batch_size: int = 128
    min_text_length: int = 100
    min_paragraph_length: int = 30
    min_paragraphs_for_triplets: int = 2
    min_positive_words: int = 15
    # 0.0 => always anchor on the article title. This was the *effective*
    # behaviour for every run that beat baseline: before extract/wikidump.py
    # stopped emitting the article title as its own leading paragraph,
    # paras[0] was that title, so `_first_sentence(paras[0])` returned the
    # title too and this probability changed nothing. Cleaning the corpus
    # silently activated real lead-sentence anchors -- which mining's own
    # docstring records as empirically worse than title anchors.
    sentence_anchor_probability: float = 0.0


class PathSettings(BaseModel):
    data_root: Path = REPO_ROOT / "data"

    @field_validator("data_root", mode="after")
    @classmethod
    def _make_absolute(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    # --- raw inputs ---
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def page_sql_path(self) -> Path:
        return self.raw_dir / "enwiki-latest-page.sql"

    @property
    def pagelinks_sql_path(self) -> Path:
        return self.raw_dir / "enwiki-latest-pagelinks.sql"

    @property
    def linktarget_sql_path(self) -> Path:
        return self.raw_dir / "enwiki-latest-linktarget.sql"

    @property
    def extracted_dir(self) -> Path:
        return self.raw_dir / "extracted_wikidata"

    # --- processed text ---
    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def json_dir(self) -> Path:
        return self.processed_dir / "wikidata_json"

    @property
    def jsonl_dir(self) -> Path:
        return self.processed_dir / "wikidata_jsonl"

    @property
    def article_titles_path(self) -> Path:
        return self.processed_dir / "article_titles.json"

    # --- link graph ---
    @property
    def link_graph_jsonl_dir(self) -> Path:
        return self.processed_dir / "wiki_link_graph_jsonl"

    @property
    def link_graph_db_path(self) -> Path:
        return self.processed_dir / "wiki_link_graph.db"
    
    # --- backlink counts ---
    @property
    def backlink_counts_db_path(self) -> Path:
        return self.processed_dir / "wiki_backlink_counts.db"

    # --- models ---
    @property
    def models_dir(self) -> Path:
        return self.processed_dir / "models"

    # --- embeddings / FAISS ---
    @property
    def faiss_index_dir(self) -> Path:
        """The active index, used by every CLI command that doesn't get an
        explicit --index-path/--meta-db.

        This deliberately points at the baseline run rather than the old
        processed/faiss_index location. That location held an index built
        from the pre-fix corpus -- back when extract/wikidump.py still
        emitted the article title and bare section headers as their own
        paragraphs -- and because `swsearch evaluate` and `swsearch search`
        silently fall back here when the flags are omitted, that stale index
        answered two separate evaluations with numbers from a corpus nobody
        had used in weeks. Baseline measures MRR 0.7017 against it versus
        0.7947 against the clean corpus, which reads as a regression that
        never happened. Pointing the default at the live index removes the
        trap.

        NOTE that `swsearch embed` and `swsearch build-index` also default
        their *outputs* here, and metadata.store.create_faiss_meta_db()
        deletes whatever is already at its target path. Running either
        without explicit --output-dir/--meta-db/--index-path will overwrite
        the baseline index in place.
        """
        return self.models_dir / "baseline" / "runs" / "all-MiniLM-L6"

    @property
    def embeddings_dir(self) -> Path:
        return self.faiss_index_dir / "embeddings"

    @property
    def faiss_index_path(self) -> Path:
        return self.faiss_index_dir / "paragraphs.index"

    @property
    def faiss_meta_db_path(self) -> Path:
        return self.faiss_index_dir / "paragraphs.index.meta.db"

    # --- triplets ---
    @property
    def triplets_dir(self) -> Path:
        return self.processed_dir / "triplets" / "parallel_parts"

    # --- eval ---
    @property
    def test_data_dir(self) -> Path:
        return self.data_root / "test_data"

    @property
    def test_queries_path(self) -> Path:
        return self.test_data_dir / "test_queries.json"

    # --- research (custom encoder, archived) ---
    @property
    def custom_model_dir(self) -> Path:
        return self.data_root / "custom_model"

    # --- transfer learning (fine-tuned SentenceTransformer) ---
    @property
    def transfer_model_dir(self) -> Path:
        return self.data_root / "transfer_model"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWSEARCH_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    spark: SparkSettings = SparkSettings()
    model: ModelSettings = ModelSettings()
    mining: MiningSettings = MiningSettings()
    paths: PathSettings = PathSettings()
    rerank: RerankSettings = RerankSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
