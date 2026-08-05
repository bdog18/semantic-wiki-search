import os
import faiss
import gc
import json
import sys
import time
import torch
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add root directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.sqlite_lookup import load_linkgraph_sqlite, get_links_for_title_sqlite, load_faiss_meta_sqlite, get_text_and_meta_from_db

# Global variables to hold resources in each worker
_model = None
_index = None
_faiss_meta_conn = None
_article_titles = None
_link_conn = None

def init_worker_lightweight(faiss_index_path, faiss_meta_db_path, article_titles_path, link_graph_path, model_name):
    """Initialize worker by loading only essential resources"""
    global _model, _index, _faiss_meta_conn, _article_titles, _link_conn
    
    worker_id = os.getpid()
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Worker {worker_id}: Starting initialization...")
    print(f"{'='*60}")
    
    # Set tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load FAISS index from disk
    print(f"Worker {worker_id}: [1/6] Loading FAISS index from disk (1.5GB)...")
    print(f"Worker {worker_id}:       Path: {faiss_index_path}")
    faiss_load_start = time.time()
    cpu_index = faiss.read_index(faiss_index_path)
    faiss_load_time = time.time() - faiss_load_start
    print(f"Worker {worker_id}:       ✓ Loaded in {faiss_load_time:.1f}s")
    
        # Move FAISS index to GPU for faster similarity search
    if torch.cuda.is_available():
        try:
            print(f"Worker {worker_id}: [2/6] Transferring FAISS index to GPU with FP16...")
            gpu_transfer_start = time.time()
            
            res = faiss.StandardGpuResources()
            # Set temporary memory allocation
            res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
            
            # Use cloner options to enable FP16 (half precision)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # ✅ Use FP16 instead of FP32 - saves ~50% VRAM
            co.usePrecomputed = False  # Don't precompute (saves memory)
            
            # Clone to GPU with FP16
            _index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            
            gpu_transfer_time = time.time() - gpu_transfer_start
            print(f"Worker {worker_id}:       ✓ Transferred to GPU with FP16 in {gpu_transfer_time:.1f}s")
            print(f"Worker {worker_id}:       ✓ VRAM saved: ~50% (FP16 vs FP32)")
        except Exception as e:
            print(f"Worker {worker_id}:       ✗ Failed to move FAISS to GPU: {e}")
            print(f"Worker {worker_id}:       → Falling back to CPU")
            _index = cpu_index
    else:
        print(f"Worker {worker_id}: [2/6] CUDA not available, using CPU FAISS")
        _index = cpu_index
    
    # Load FAISS metadata database
    print(f"Worker {worker_id}: [3/6] Loading FAISS metadata SQLite database...")
    meta_db_start = time.time()
    _faiss_meta_conn = load_faiss_meta_sqlite(faiss_meta_db_path)
    meta_db_time = time.time() - meta_db_start
    print(f"Worker {worker_id}:       ✓ Loaded in {meta_db_time:.1f}s")
    
    # Load model on GPU with half precision for faster encoding
    print(f"Worker {worker_id}: [4/6] Loading SentenceTransformer model...")
    model_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = SentenceTransformer(model_name, device=device)
    model_load_time = time.time() - model_start
    print(f"Worker {worker_id}:       ✓ Model loaded on {device} in {model_load_time:.1f}s")
    
    # Enable mixed precision (FP16) for 2x faster encoding on GPU
    if device == 'cuda':
        print(f"Worker {worker_id}: [5/6] Converting model to FP16 (mixed precision)...")
        fp16_start = time.time()
        _model.half()
        fp16_time = time.time() - fp16_start
        print(f"Worker {worker_id}:       ✓ Converted to FP16 in {fp16_time:.1f}s")
    else:
        print(f"Worker {worker_id}: [5/6] Skipping FP16 conversion (CPU mode)")
    
    # Load link graph database
    print(f"Worker {worker_id}: [6/6] Loading link graph and article titles...")
    misc_start = time.time()
    _link_conn = load_linkgraph_sqlite(link_graph_path)

    # Load article titles
    with open(article_titles_path, "r", encoding="utf-8") as f:
        _article_titles = json.load(f)
    misc_time = time.time() - misc_start
    print(f"Worker {worker_id}:       ✓ Loaded in {misc_time:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Worker {worker_id}: ✓✓✓ READY! Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

def process_file_worker(args):
    """File worker for triplet mining"""
    file_path, triplet_output_dir, file_index = args
    global _model, _index, _faiss_meta_conn, _article_titles, _link_conn

    # Prepare output file with unique incremental numbering
    safe_name = f"wiki_{file_index}_triplets.jsonl"
    output_path = os.path.join(triplet_output_dir, safe_name)
    
    triplets_written = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f, \
             open(output_path, "w", encoding="utf-8") as out_f:
            
            batch_anchors = []
            batch_positives = []
            batch_metadata = []
            
            for line in f:
                try:
                    article = json.loads(line)
                    
                    # if no title, skip
                    title = article.get("title")
                    if not title:
                        continue
                    
                    # if content too short, skip
                    raw_text = article.get("content", "")
                    if len(raw_text) < MIN_TEXT_LENGTH:
                        continue
                    
                    # if not enough paragraphs, skip
                    paras = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > MIN_PARAGRAPH_LENGTH]
                    if len(paras) < MIN_PARAGRAPHS_FOR_TRIPLETS:
                        continue
                    
                    # if no links, skip
                    linked_titles = get_links_for_title_sqlite(_link_conn, title)
                    if not linked_titles:
                        continue
                    
                    triplet_count = 0
                    # Generate triplets from paragraphs (max limit per article)
                    for i in range(1, min(len(paras), MAX_TRIPLETS_PER_ARTICLE + 1)):
                        anchor = paras[i]
                        # if anchor too short, skip
                        if len(anchor) <= MIN_PARAGRAPH_LENGTH:
                            continue
                        
                        positive = None
                        # Select next paragraph as positive, or wrap around
                        if i + 1 < len(paras):
                            positive = paras[i + 1]
                        elif len(paras) > 2:
                            positive = paras[0] if i != 0 else paras[2]
                        
                        # if positive exists and long enough, add to batch
                        if positive and len(positive) > MIN_PARAGRAPH_LENGTH:
                            batch_anchors.append(anchor)
                            batch_positives.append(positive)
                            batch_metadata.append((title, linked_titles))
                            triplet_count += 1
                            
                            if triplet_count >= MAX_TRIPLETS_PER_ARTICLE:
                                break
                    
                    # Process batch when it reaches optimal size
                    if len(batch_anchors) >= BATCH_SIZE:
                        written = process_batch(
                            batch_anchors, 
                            batch_positives, 
                            batch_metadata, 
                            out_f, 
                            NEGATIVE_POOL_SIZE
                        )
                        triplets_written += written
                        
                        batch_anchors.clear()
                        batch_positives.clear()
                        batch_metadata.clear()
                        
                        if triplets_written % 1000 == 0:
                            gc.collect()
                            
                except (json.JSONDecodeError, Exception):
                    continue
            
            # Process last batch
            if batch_anchors:
                written = process_batch(
                    batch_anchors, 
                    batch_positives, 
                    batch_metadata, 
                    out_f, 
                    NEGATIVE_POOL_SIZE
                )
                triplets_written += written
        
        _rename_completed_file(file_path)

        return file_path, triplets_written
        
    except Exception as e:
        return file_path, 0

def process_batch(anchors, positives, metadata, out_f, neg_pool_size):
    """Batch processing with vectorized operations"""
    global _model, _index, _faiss_meta_conn, _article_titles, _link_conn
    
    # if no anchors, return 0
    if not anchors:
        return 0
    
    # if resources not initialized, return 0
    if not _model or not _index or not _faiss_meta_conn:
        return 0
    
    triplets_count = 0
    try:
        anchor_embeddings = _model.encode(
            anchors, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False
        )
        
        D, I = _index.search(anchor_embeddings.astype('float32'), neg_pool_size)
        
        # For each anchor-positive pair, find a suitable negative
        for idx, (anchor, positive, (src_title, linked_titles)) in enumerate(zip(anchors, positives, metadata)):
            negative = None

            for j in I[idx]:
                neg_para, neg_title = get_text_and_meta_from_db(_faiss_meta_conn, int(j))
                if neg_para and neg_title:
                    if neg_title != src_title and neg_title not in linked_titles:
                        if (len(neg_para) > MIN_PARAGRAPH_LENGTH and 
                            neg_para != anchor and 
                            neg_para != positive):
                            negative = neg_para
                            break
            
            if negative:
                triplet = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "source": src_title,
                    "url": (_article_titles or {}).get(src_title, "")
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                triplets_count += 1
        
        del anchor_embeddings, D, I
        
    except Exception as e:
        pass
        
    return triplets_count

def mine_triplets_parallel_shared(wikidata_jsonl_dir, faiss_index_path, faiss_meta_db_path, 
                                 article_titles_path, link_graph_path, triplet_output_dir, 
                                 num_processes=None, max_files_per_worker=None):
    """Memory-efficient parallel triplet mining with tqdm progress tracking"""
    # Ensure output directory exists
    os.makedirs(triplet_output_dir, exist_ok=True)
    
    # Clean up empty triplet files (from interrupted runs)
    print("Checking for empty triplet files from previous interrupted runs...")
    empty_count = 0
    for filename in os.listdir(triplet_output_dir):
        if filename.endswith("_triplets.jsonl"):
            filepath = os.path.join(triplet_output_dir, filename)
            if os.path.getsize(filepath) == 0:
                os.remove(filepath)
                empty_count += 1
    if empty_count > 0:
        print(f"  Removed {empty_count} empty triplet file(s)")
    
    # Get list of all files to process (including .completed files)
    all_files = [os.path.join(root, file)
                 for root, _, filenames in os.walk(wikidata_jsonl_dir)
                 for file in filenames if file.endswith(".jsonl") or file.endswith(".jsonl.completed")]
    
    # Sort files for deterministic ordering
    all_files.sort()
    
    # Filter out already completed files (keep only non-completed files)
    files_to_process = []
    completed_files = set()
    
    for filepath in all_files:
        if filepath.endswith(".completed"):
            # Track completed files
            completed_files.add(filepath)
            # Also track the original name without .completed
            original_path = filepath[:-len(".completed")]
            completed_files.add(original_path)
        else:
            # Check if there's a .completed version
            completed_path = filepath + ".completed"
            if completed_path not in completed_files and not os.path.exists(completed_path):
                files_to_process.append(filepath)
    
    # If no files to process, return
    if not files_to_process:
        print("No .jsonl files to process (all files already completed).")
        return 0
    
    print(f"Found {len(all_files)} total files, {len(completed_files)} already completed")
    print(f"Files to process: {len(files_to_process)}")
    
    # Find the highest existing triplet file index to continue numbering
    existing_triplet_files = [f for f in os.listdir(triplet_output_dir) if f.startswith("wiki_") and f.endswith("_triplets.jsonl")]
    max_existing_index = 0
    for filename in existing_triplet_files:
        try:
            # Extract number from format: wiki_123_triplets.jsonl
            parts = filename.split("_")
            if len(parts) >= 2:
                index = int(parts[1])
                if index > max_existing_index:
                    max_existing_index = index
        except (ValueError, IndexError):
            continue
    
    start_index = max_existing_index + 1
    if max_existing_index > 0:
        print(f"Continuing numbering from index {start_index} (found existing files up to wiki_{max_existing_index}_triplets.jsonl)")
    
    files = files_to_process
    
    # Reduce number of processes to manage memory
    if num_processes is None:
        num_processes = min(cpu_count() // 2, 2)
    
    # Increase files per worker to reduce reinitialization
    if max_files_per_worker is None:
        max_files_per_worker = max(200, len(files) // num_processes)
    
    print(f"\nStarting triplet mining:")
    print(f"  Files to process: {len(files):,}")
    print(f"  Workers: {num_processes}")
    print(f"  Max files per worker: {max_files_per_worker:,}")
    print(f"\nInitializing workers (this may take 10-30 seconds)...")
    
    start_time = time.time()
    
    # Create process pool
    with Pool(processes=num_processes, 
              initializer=init_worker_lightweight,
              initargs=(faiss_index_path, faiss_meta_db_path, article_titles_path, link_graph_path, MODEL_NAME),
              maxtasksperchild=max_files_per_worker) as pool:
        
        init_time = time.time() - start_time
        print(f"Workers initialized in {init_time:.1f}s")
        print(f"\nProcessing files...\n")
        
        # Prepare arguments for workers with unique file index for each file
        # Start indexing from start_index to continue from where we left off
        worker_args = [(file_path, triplet_output_dir, start_index + idx) 
                      for idx, file_path in enumerate(files)]
        
        # Process files in parallel with tqdm progress bar
        results = list(tqdm(
            pool.imap_unordered(process_file_worker, worker_args),
            total=len(files),
            desc="Mining triplets",
            unit="file",
            ncols=100
        ))
    
    # Calculate total triplets from results
    total_triplets = sum(num_triplets for _, num_triplets in results)
    files_with_triplets = sum(1 for _, num_triplets in results if num_triplets > 0)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Complete!")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
    print(f"  Files processed: {len(results):,}")
    print(f"  Files with triplets: {files_with_triplets:,}")
    print(f"  Total triplets: {total_triplets:,}")
    print(f"  Average: {total_triplets/max(files_with_triplets, 1):.1f} triplets/file")
    print(f"  Rate: {len(files)/(elapsed/3600):.1f} files/hour")
    print(f"{'='*70}\n")
    
    return total_triplets

def _rename_completed_file(src_path):
    """Rename completed file to indicate processing is done using atomic operation"""
    base_name = os.path.basename(src_path)
    completed_name = base_name + ".completed"
    completed_path = os.path.join(os.path.dirname(src_path), completed_name)
    # Use os.replace for atomic rename (overwrites if exists)
    os.replace(src_path, completed_path)


# CONFIGURATION
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TRIPLETS_PER_ARTICLE = 10
NEGATIVE_POOL_SIZE = 10
BATCH_SIZE = 128  # Increased for better GPU utilization 
MIN_TEXT_LENGTH = 100  # Minimum length of article content to consider
MIN_PARAGRAPHS_FOR_TRIPLETS = 2  # Minimum number of paragraphs required to mine triplets
MIN_PARAGRAPH_LENGTH = 30  # Minimum length of paragraph to be considered

# Preprocessed Wikipedia Data Paths
PROCESSED_JSONL_DIR = r"../data/processed/processed_wikidata_jsonl"
WIKIDATA_JSONL_DIR = r"../data/processed/wikidata_jsonl"
ARTICLE_TITLES_JSON = r"../data/processed/article_titles.json"
WIKI_LINK_GRAPH_DB_PATH = r"../data/processed/wiki_link_graph.db"
TRIPLETS_DIR = r"../data/processed/triplets/parallel_parts"
FAISS_PARA_INDEX_PATH = r"../data/processed/faiss_index/paragraphs.index"
FAISS_PARA_META_DB_PATH = r"../data/processed/faiss_index/paragraphs.index.meta.db"

if __name__ == "__main__":
    total_triplets = mine_triplets_parallel_shared(
        wikidata_jsonl_dir=WIKIDATA_JSONL_DIR,
        faiss_index_path=FAISS_PARA_INDEX_PATH,
        faiss_meta_db_path=FAISS_PARA_META_DB_PATH,
        article_titles_path=ARTICLE_TITLES_JSON,
        link_graph_path=WIKI_LINK_GRAPH_DB_PATH,
        triplet_output_dir=TRIPLETS_DIR,
        num_processes=4,  # Reduced to 2 for GPU FAISS + models to fit in 8GB VRAM
        max_files_per_worker=200  # Increased since fewer processes
    )

    print(f"Triplet mining complete. Generated {total_triplets} triplets.")
