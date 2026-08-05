import faiss
import os

def check_faiss_index_type(index_path):
    """Check the type and properties of a FAISS index"""
    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}")
        return
    
    print(f"Analyzing index: {index_path}")
    print("="*70)
    
    # Load index
    index = faiss.read_index(index_path)
    
    # Basic info
    print(f"\nBasic Information:")
    print(f"  Total vectors: {index.ntotal:,}")
    print(f"  Dimension: {index.d}")
    print(f"  Is trained: {index.is_trained}")
    
    # File size
    size_bytes = os.path.getsize(index_path)
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    print(f"  File size: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    
    if index.ntotal > 0:
        bytes_per_vector = size_bytes / index.ntotal
        print(f"  Bytes per vector: {bytes_per_vector:.2f}")
    
    # Determine index type
    print(f"\nIndex Type Detection:")
    index_class = index.__class__.__name__
    print(f"  Class: {index_class}")
    
    # Check for specific index types
    if isinstance(index, faiss.IndexFlat):
        print(f"  Type: Flat (Uncompressed, Exact Search)")
        print(f"  Metric: L2" if isinstance(index, faiss.IndexFlatL2) else "  Metric: IP")
        print(f"  ✓ Supports GPU: Yes")
        print(f"  ✓ Supports reconstruction: Yes")
        
    elif isinstance(index, faiss.IndexIVFFlat):
        print(f"  Type: IVF-Flat (Inverted File, Uncompressed)")
        print(f"  Number of clusters (nlist): {index.nlist}")
        print(f"  Search clusters (nprobe): {index.nprobe}")
        print(f"  ✓ Supports GPU: Yes")
        print(f"  ✗ Supports reconstruction: No (no direct map)")
        
    elif isinstance(index, faiss.IndexIVFPQ):
        print(f"  Type: IVF-PQ (Inverted File + Product Quantization)")
        print(f"  Number of clusters (nlist): {index.nlist}")
        print(f"  Search clusters (nprobe): {index.nprobe}")
        print(f"  PQ subquantizers (M): {index.pq.M}")
        print(f"  PQ bits per code: {index.pq.nbits}")
        compression_ratio = (index.d * 4) / (index.pq.M * index.pq.nbits / 8)
        print(f"  Compression ratio: ~{compression_ratio:.1f}x")
        print(f"  ✓ Supports GPU: Yes")
        print(f"  ✗ Supports reconstruction: No (PQ compression)")
        
    elif isinstance(index, faiss.IndexPQ):
        print(f"  Type: PQ (Product Quantization Only)")
        print(f"  PQ subquantizers (M): {index.pq.M}")
        print(f"  PQ bits per code: {index.pq.nbits}")
        compression_ratio = (index.d * 4) / (index.pq.M * index.pq.nbits / 8)
        print(f"  Compression ratio: ~{compression_ratio:.1f}x")
        print(f"  ✓ Supports GPU: Yes")
        print(f"  ✗ Supports reconstruction: No (PQ compression)")
        
    else:
        print(f"  Type: {index_class}")
    
    # Memory estimation for GPU
    print(f"\nGPU Memory Estimation:")
    if isinstance(index, faiss.IndexFlat):
        gpu_mem = size_bytes
        print(f"  Approximate VRAM needed: {gpu_mem / (1024**2):.1f} MB ({gpu_mem / (1024**3):.2f} GB)")
        print(f"  Note: Flat indices use full size on GPU")
    elif isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexPQ)):
        gpu_mem = size_bytes * 1.2  # Some overhead
        print(f"  Approximate VRAM needed: {gpu_mem / (1024**2):.1f} MB ({gpu_mem / (1024**3):.2f} GB)")
        print(f"  Note: Compressed indices are GPU-friendly")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Check your index
    INDEX_PATH = "../data/processed/faiss_index/paragraphs.index"
    check_faiss_index_type(INDEX_PATH)