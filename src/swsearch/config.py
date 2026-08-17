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
    negative_pool_size: int = 10
    batch_size: int = 128
    min_text_length: int = 100
    min_paragraph_length: int = 30
    min_paragraphs_for_triplets: int = 2
    min_positive_words: int = 15
    sentence_anchor_probability: float = 0.3


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

    # --- embeddings / FAISS ---
    @property
    def faiss_index_dir(self) -> Path:
        return self.processed_dir / "faiss_index"

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
