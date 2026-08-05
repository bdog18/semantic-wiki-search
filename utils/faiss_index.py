import os
import numpy as np
import faiss
from glob import glob
from tqdm import tqdm
import json

def create_faiss_index(faiss_idx_path, article_embedding_path):
    if not os.path.isfile(faiss_idx_path):
        # Load the embeddings and Initialize the FAISS index
        embeddings = np.load(article_embedding_path)
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add embeddings in batches
        batch_size = 1000
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to FAISS index"):
            batch = embeddings[i:i+batch_size]
            index.add(batch)

        faiss.write_index(index, faiss_idx_path)
        print("FAISS index saved.")
    else:
        print("FAISS index already exists.")


def create_faiss_index_from_dir(embedding_dir_path, faiss_idx_path, batch_size=1000):
    os.makedirs(os.path.dirname(faiss_idx_path), exist_ok=True)
    if os.path.exists(faiss_idx_path):
        print("FAISS index already exists.")
        return

    # Find all .npy embedding files
    embedding_files = sorted(glob(os.path.join(embedding_dir_path, '*.npy')))
    if not embedding_files:
        raise ValueError("No .npy embedding files found in the directory.")

    # Load first file header to get dimensionality
    first_batch = np.load(embedding_files[0], mmap_mode='r')
    dim = first_batch.shape[1]
    index = faiss.IndexFlatL2(dim)

    print(f"Found {len(embedding_files)} embedding files. Building FAISS index...")

    for file_path in tqdm(embedding_files, desc="Processing embedding files"):
        embeddings = np.load(file_path, mmap_mode='r')  # memory-mapped
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            index.add(batch.astype(np.float32))  # ensure correct type

    faiss.write_index(index, faiss_idx_path)
    print("FAISS index saved.")

def query_faiss(faiss_idx_path, query_embedding, k):
    # Load FAISS index
    index = faiss.read_index(faiss_idx_path)

    # Search FAISS (returns distances and indices)
    _, indices = index.search(query_embedding, k)
    return indices

def load_faiss_index(index_path, meta_path):
    print("Loading FAISS index...")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["all_texts"], meta["text_to_meta"]