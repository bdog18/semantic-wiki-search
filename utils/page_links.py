import os
import sys

# Add root directory (one level up from notebooks/)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
import shelve
import random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, trim
from glob import glob
from utils.sqlite_lookup import load_linkgraph_sqlite, get_links_for_title_sqlite


def build_paragraph_shelve(data_dir, shelve_path):
    print("Building paragraph shelve DB...")
    with shelve.open(shelve_path, writeback=False) as db:
        for article in tqdm(stream_article_lines(data_dir), desc="Indexing paragraphs"):
            title = article.get("title")
            content = article.get("content", "")
            paras = [p.strip() for p in content.split("\n\n") if p.strip()]
            if title and paras:
                db[title] = paras
    print("Paragraph shelve DB created.")


def load_articles_titles_only(data_dir):
    article_titles = {}
    files = []
    for root, _, filenames in os.walk(data_dir):
        for file in filenames:
            if file.endswith(".jsonl"):
                files.append(os.path.join(root, file))
    random.shuffle(files)

    for file_path in tqdm(files, desc="Reading titles"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line)
                    title = article.get("title")
                    if title:
                        article_titles[title] = article.get("url", "")
                except json.JSONDecodeError:
                    continue
    return article_titles


def save_embeddings_with_spark(data_dir, model, embedding_dir, meta_path):
    spark = SparkSession.builder \
        .appName("Capstone") \
        .master("local[*]") \
        .config("spark.driver.memory", "50g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.local.dir", "../spark-temp") \
        .config("spark.driver.maxResultSize", "10g") \
        .getOrCreate()

    os.makedirs(embedding_dir, exist_ok=True)
    print("Loading JSON into Spark DataFrame...")
    df = spark.read.option("multiLine", True) \
                    .option("recursiveFileLookup", "true") \
                    .json(data_dir)

    print("Splitting paragraphs...")
    df = df.withColumn("paragraphs", split(col("content"), r"\n\s*\n"))
    df = df.select("title", explode("paragraphs").alias("paragraph"))
    df = df.filter(trim(col("paragraph")) != "")

    all_texts = []
    text_to_meta = []

    print("Processing all paragraphs at once...")
    for row in tqdm(df.toLocalIterator(), desc="Encoding paragraphs", total=df.count()):
        paragraph = row["paragraph"]
        title = row["title"]
        all_texts.append(paragraph)
        text_to_meta.append(title)

    print("Encoding in batches and saving...")
    batch_size = 1024
    batch_number = 0
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Batching"):
        batch = all_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        np.save(os.path.join(embedding_dir, f"batch_{batch_number:05d}.npy"), batch_embeddings)
        batch_number += 1

    print("Saving metadata...")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"all_texts": all_texts, "text_to_meta": text_to_meta}, f)

    print("Done embedding and saving.")



def load_faiss_index(index_path, meta_path):
    print("loading faiss")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["all_texts"], meta["text_to_meta"]


def mine_triplets_streaming(para_db_path, article_titles, model, index, all_texts, text_to_meta, output_path):
    print("Streaming triplet mining with batching (anchor + positive)...")
    triplets_written = 0
    BATCH_SIZE = 2048  # Tune this to fit 3070 Ti memory

    files = []
    for root, _, filenames in os.walk(WIKI_DATA_DIR_JSONL):
        for file in filenames:
            if file.endswith(".jsonl"):
                files.append(os.path.join(root, file))
    random.shuffle(files)

    link_conn = load_linkgraph_sqlite(LINK_GRAPH_PATH)
    with shelve.open(para_db_path, flag="r") as para_db, open(output_path, "w", encoding="utf-8") as out_f:
        anchor_batch = []
        positive_batch = []
        meta_batch = []

        def process_batch():
            nonlocal triplets_written
            if not anchor_batch:
                return

            anchor_embeddings = model.encode(anchor_batch, 
                                             batch_size=BATCH_SIZE, 
                                             convert_to_numpy=True, 
                                             normalize_embeddings=True,
                                             device='cuda')
            D, I = index.search(anchor_embeddings, NEGATIVE_POOL_SIZE + 20)

            for idx, (anchor, positive, title, linked_titles) in enumerate(meta_batch):
                negative = None
                for j in I[idx]:
                    neg_title = text_to_meta[j]
                    neg_para = all_texts[j]
                    if (
                        neg_title != title and
                        neg_title not in linked_titles and
                        neg_para != anchor and
                        neg_para != positive and
                        len(neg_para) > 20
                    ):
                        negative = neg_para
                        break
                if not negative:
                    continue

                triplet = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "source": title,
                    "url": article_titles.get(title, "")
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                triplets_written += 1

        for file_path in tqdm(files, desc="Mining files"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        title = article.get("title")
                        raw_text = article.get("content", "")
                        paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
                        linked_titles = get_links_for_title_sqlite(link_conn, title)

                        if not linked_titles:
                            continue

                        triplets_for_article = 0
                        for i, anchor in enumerate(paras[1:], 1):
                            if triplets_for_article >= MAX_TRIPLETS_PER_ARTICLE:
                                break
                            if not anchor.strip() or len(anchor) <= 20:
                                continue

                            # Batchable positive selection from links
                            positive = None
                            for linked_title in linked_titles:
                                linked_paras = para_db.get(linked_title, [])
                                positive = next(
                                    (p for p in linked_paras if p.strip() and p.strip() != linked_title and linked_title.lower() not in p.lower() and len(p) > 20),
                                    None
                                )
                                if positive:
                                    break
                            if not positive and len(paras) > i + 1:
                                positive = paras[i + 1]
                            if not positive:
                                continue

                            anchor_batch.append(anchor)
                            positive_batch.append(positive)
                            meta_batch.append((anchor, positive, title, linked_titles))
                            triplets_for_article += 1

                            if len(anchor_batch) >= BATCH_SIZE:
                                process_batch()
                                anchor_batch.clear()
                                positive_batch.clear()
                                meta_batch.clear()

                            if triplets_written >= MAX_TRIPLETS:
                                print("Reached triplet limit.")
                                return
                    except json.JSONDecodeError:
                        continue

        process_batch()

    print(f"Finished mining {triplets_written} triplets.")


def stream_article_lines(data_dir):
    files = []
    for root, _, filenames in os.walk(data_dir):
        for file in filenames:
            if file.endswith(".json") or file.endswith(".jsonl"):
                files.append(os.path.join(root, file))
    random.shuffle(files)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def create_ivfpq_faiss_index(embedding_dir_path, faiss_idx_path, nlist=100, nprobe=10, train_size=100000, batch_size=1000, m=16, nbits=8):
    os.makedirs(os.path.dirname(faiss_idx_path), exist_ok=True)
    if os.path.exists(faiss_idx_path):
        print("FAISS index already exists.")
        return

    embedding_files = sorted(glob(os.path.join(embedding_dir_path, '*.npy')))
    if not embedding_files:
        raise ValueError("No .npy embedding files found in the directory.")

    print("Sampling vectors for training the IVF-PQ index...")
    sampled = []
    for file in embedding_files:
        data = np.load(file, mmap_mode='r')
        for vec in data:
            sampled.append(vec.astype(np.float32))
            if len(sampled) >= train_size:
                break
        if len(sampled) >= train_size:
            break
    train_data = np.vstack(sampled)

    dim = train_data.shape[1]
    if dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m} for PQ indexing.")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(train_data)
    index.nprobe = nprobe

    print(f"Adding embeddings to single IVF-PQ index from {len(embedding_files)} files...")
    for file_path in tqdm(embedding_files, desc="Adding to IVF-PQ index"):
        vectors = np.load(file_path, mmap_mode='r')
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size].astype(np.float32)
            index.add(batch)

    faiss.write_index(index, faiss_idx_path)
    print("IVF-PQ FAISS index saved.")

# CONFIGURATION
WIKI_DATA_DIR_JSON = "../data/processed/wikidata_json"
WIKI_DATA_DIR_JSONL = "../data/processed/wikidata_jsonl"
LINK_GRAPH_PATH = "../data/processed/wiki_link_graph.db"
PARAGRAPH_DB_PATH = "../data/processed/paragraphs_shelve.db"
EMBEDDING_DIR = "../data/processed/faiss_index/embeddings"
FAISS_INDEX_PATH = "../data/processed/faiss_index/paragraphs.index"
FAISS_META_PATH = FAISS_INDEX_PATH + ".meta.json"
TRIPLET_OUTPUT_PATH = "../data/processed/triplets/advanced_triplets.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TRIPLETS = 50000000
NEGATIVE_POOL_SIZE = 5
MAX_TRIPLETS_PER_ARTICLE = 5


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    model.to('cuda')

    # if not os.path.exists(LINK_GRAPH_PATH + ".db"):
    #     convert_link_graph_to_shelve("../data/processed/wiki_link_graph.json", LINK_GRAPH_PATH)

    # if not os.path.exists(PARAGRAPH_DB_PATH + ".db"):
    # build_paragraph_shelve(WIKI_DATA_DIR_JSONL, PARAGRAPH_DB_PATH)

    # if not os.path.exists(FAISS_INDEX_PATH):
    save_embeddings_with_spark(WIKI_DATA_DIR_JSON, model, EMBEDDING_DIR, FAISS_META_PATH) # Regular json
    # create_ivfpq_faiss_index(EMBEDDING_DIR, FAISS_INDEX_PATH)

    # article_titles = load_articles_titles_only(WIKI_DATA_DIR_JSONL) # jsonl
    # index, all_texts, text_to_meta = load_faiss_index(FAISS_INDEX_PATH, FAISS_META_PATH)

    # mine_triplets_streaming(
    #     para_db_path=PARAGRAPH_DB_PATH,
    #     article_titles=article_titles,
    #     model=model,
    #     index=index,
    #     all_texts=all_texts,
    #     text_to_meta=text_to_meta,
    #     output_path=TRIPLET_OUTPUT_PATH
    # )

