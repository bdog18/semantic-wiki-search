import os

import faiss


def inspect_index(index_path: str) -> None:
    """Print type, size, and GPU-memory-estimate diagnostics for a FAISS index file."""
    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}")
        return

    print(f"Analyzing index: {index_path}")
    print("=" * 70)

    index = faiss.read_index(index_path)

    print("\nBasic Information:")
    print(f"  Total vectors: {index.ntotal:,}")
    print(f"  Dimension: {index.d}")
    print(f"  Is trained: {index.is_trained}")

    size_bytes = os.path.getsize(index_path)
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    print(f"  File size: {size_mb:.1f} MB ({size_gb:.2f} GB)")

    if index.ntotal > 0:
        print(f"  Bytes per vector: {size_bytes / index.ntotal:.2f}")

    print("\nIndex Type Detection:")
    print(f"  Class: {index.__class__.__name__}")

    if isinstance(index, faiss.IndexFlat):
        print("  Type: Flat (Uncompressed, Exact Search)")
        print("  Metric: L2" if isinstance(index, faiss.IndexFlatL2) else "  Metric: IP")
        print("  Supports GPU: Yes")
        print("  Supports reconstruction: Yes")

    elif isinstance(index, faiss.IndexIVFFlat):
        print("  Type: IVF-Flat (Inverted File, Uncompressed)")
        print(f"  Number of clusters (nlist): {index.nlist}")
        print(f"  Search clusters (nprobe): {index.nprobe}")
        print("  Supports GPU: Yes")
        print("  Supports reconstruction: No (no direct map)")

    elif isinstance(index, faiss.IndexIVFPQ):
        print("  Type: IVF-PQ (Inverted File + Product Quantization)")
        print(f"  Number of clusters (nlist): {index.nlist}")
        print(f"  Search clusters (nprobe): {index.nprobe}")
        print(f"  PQ subquantizers (M): {index.pq.M}")
        print(f"  PQ bits per code: {index.pq.nbits}")
        compression_ratio = (index.d * 4) / (index.pq.M * index.pq.nbits / 8)
        print(f"  Compression ratio: ~{compression_ratio:.1f}x")
        print("  Supports GPU: Yes")
        print("  Supports reconstruction: No (PQ compression)")

    elif isinstance(index, faiss.IndexPQ):
        print("  Type: PQ (Product Quantization Only)")
        print(f"  PQ subquantizers (M): {index.pq.M}")
        print(f"  PQ bits per code: {index.pq.nbits}")
        compression_ratio = (index.d * 4) / (index.pq.M * index.pq.nbits / 8)
        print(f"  Compression ratio: ~{compression_ratio:.1f}x")
        print("  Supports GPU: Yes")
        print("  Supports reconstruction: No (PQ compression)")

    else:
        print(f"  Type: {index.__class__.__name__}")

    print("\nGPU Memory Estimation:")
    if isinstance(index, faiss.IndexFlat):
        gpu_mem = size_bytes
        print(f"  Approximate VRAM needed: {gpu_mem / (1024**2):.1f} MB ({gpu_mem / (1024**3):.2f} GB)")
        print("  Note: Flat indices use full size on GPU")
    elif isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexPQ)):
        gpu_mem = size_bytes * 1.2
        print(f"  Approximate VRAM needed: {gpu_mem / (1024**2):.1f} MB ({gpu_mem / (1024**3):.2f} GB)")
        print("  Note: Compressed indices are GPU-friendly")

    print(f"\n{'=' * 70}")
