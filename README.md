# Semantic Wikipedia Search Engine (In Progress)

**Project Status**: In Progress  
**Goal**: Build a semantic search engine over a Wikipedia dump using deep learning-based embeddings, FAISS indexing, and custom model training.

This project implements a full-text semantic retrieval system. It extracts, cleans, and embeds Wikipedia content, mines hard triplets using graph structure and FAISS similarity, and trains a transformer encoder using triplet loss. The goal is to support high-quality semantic search over a large corpus without relying on keyword matching.

---

## Key Features (Completed)

- Extracted and cleaned full Wikipedia dump (XML to structured JSONL)
- Built link graph from SQL dump and stored in SQLite
- Mined high-quality training triplets using internal links and FAISS
- Trained a custom transformer encoder (TensorFlow) with triplet loss
- Built an IVF-PQ FAISS index with millions of paragraph embeddings
- Implemented semantic search and reranking using cosine similarity
- Evaluated with Top-K Accuracy, Precision@K, Recall@K, and MRR

---

## In Progress

- Refine custom transformer encoder for accuracy
- Improve reranking using article-level metadata
- Build an interactive query interface (notebook + API)
- Add a simple web UI for demo purposes (Streamlit or Gradio)
- Refactor code into modular pipeline with CLI support
- Add Dockerfile and environment management

---

## Example Use Case

**Query**: "Who was involved in World War 2?"  
The system returns articles that are contextually relevant to the query, even if the wording differs from Wikipedia article titles.

---

## Project Structure (Work in Progress)

```
semantic-wiki-search/
├── data/ # Wikipedia dump files and preprocessed JSONL
├── utils/ # Processing, mining, and evaluation scripts
├── models/ # Model architecture and weights
├── embeddings/ # FAISS index and paragraph vectors
├── notebooks/ # Exploratory notebooks and testing
├── main.py # End-to-end orchestration script
└── README.md
```

---

## Technologies Used

- TensorFlow / Keras (custom encoder)
- FAISS (vector search and indexing)
- SentenceTransformers (initial embedding model)
- PySpark (for preprocessing and metadata extraction)
- SQLite (for fast link graph lookup)
- JSONL, NumPy, tqdm, shelve (pipeline components)

---

## Evaluation Metrics

- Top-K Accuracy
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

---

## Future Work

- Refine negative sampling for better contrastive training
- Support multilingual Wikipedia input
- Scale to full-enwiki dataset (10M+ articles)
- Deploy model and index behind an API for public search

---

## Notes

This is an independent project exploring end-to-end semantic search using Wikipedia as a dataset. It is actively in development and intended for demonstration, experimentation, and research purposes.
