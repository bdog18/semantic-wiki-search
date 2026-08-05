import sqlite3
import json
import os
from tqdm import tqdm

def build_faiss_meta_sqlite(all_texts, text_to_meta, db_path):
    """
    Create a SQLite database from FAISS metadata instead of using massive JSON file.
    This significantly reduces memory usage and file size.
    
    Args:
        all_texts: List of all paragraph texts from FAISS index
        text_to_meta: List mapping index position to article title
        db_path: Path to output SQLite database file
    """
    print(f"Creating FAISS metadata database at {db_path}...")
    print(f"Total texts to store: {len(all_texts):,}")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with indexed columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faiss_meta (
            idx INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            article_title TEXT NOT NULL
        )
    ''')
    
    # Create index on article_title for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_article ON faiss_meta(article_title)')
    
    # Batch insert for efficiency
    batch_size = 10000
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    
    print("Inserting data in batches...")
    for batch_num in tqdm(range(0, len(all_texts), batch_size), total=total_batches, desc="Building FAISS meta DB"):
        batch_end = min(batch_num + batch_size, len(all_texts))
        batch = [(j, all_texts[j], text_to_meta[j]) 
                 for j in range(batch_num, batch_end)]
        cursor.executemany('INSERT INTO faiss_meta VALUES (?, ?, ?)', batch)
        conn.commit()
    
    # Verify and show stats
    cursor.execute('SELECT COUNT(*) FROM faiss_meta')
    count = cursor.fetchone()[0]
    print(f"✅ Database created successfully with {count:,} entries")
    
    # Show file size comparison
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"   Database size: {db_size_mb:.2f} MB")
    
    conn.close()
    return db_path

def load_faiss_meta_sqlite(db_path):
    """Load and return connection to FAISS metadata database"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"FAISS metadata database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_text_and_meta_from_db(conn, idx):
    """
    Retrieve text and metadata by FAISS index position.
    
    Args:
        conn: SQLite connection
        idx: FAISS index position
        
    Returns:
        tuple: (text, article_title) or (None, None) if not found
    """
    cursor = conn.cursor()
    cursor.execute('SELECT text, article_title FROM faiss_meta WHERE idx = ?', (int(idx),))
    result = cursor.fetchone()
    if result:
        return result['text'], result['article_title']
    return None, None

def get_count_from_faiss_meta_db(conn):
    """Get total count of entries in FAISS metadata database"""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM faiss_meta')
    return cursor.fetchone()[0]

def build_linkgraph_sqlite(jsonl_dir, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS links (from_title TEXT PRIMARY KEY, linked_titles TEXT)")

    for root, _, files in os.walk(jsonl_dir):
        for file in tqdm(files, desc="Building SQLite"):
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            cur.execute(
                                "INSERT INTO links (from_title, linked_titles) VALUES (?, ?)",
                                (entry["from_title"], json.dumps(entry["linked_titles"]))
                            )
                        except Exception:
                            continue

    conn.commit()
    conn.close()

def load_linkgraph_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_links_for_title_sqlite(conn, title):
    cur = conn.cursor()
    cur.execute("SELECT linked_titles FROM links WHERE from_title = ?", (title,))
    row = cur.fetchone()
    if row:
        cleaned = row["linked_titles"].replace("\\'", "'").replace("\\\'", "'")
        return set(json.loads(cleaned))
    return set()

def convert_faiss_meta_json_to_sqlite(json_path, db_path):
    """
    Convert existing paragraphs.index.meta.json to SQLite database.
    This is a one-time conversion utility.
    
    Args:
        json_path: Path to existing .meta.json file
        db_path: Path where SQLite database should be created
    """
    print(f"Converting {json_path} to SQLite database...")
    print("⚠️  This may take a while for large files...")
    
    # Load the JSON file
    print("Loading JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    all_texts = meta['all_texts']
    text_to_meta = meta['text_to_meta']
    
    print(f"Loaded {len(all_texts):,} texts")
    
    # Build the SQLite database
    build_faiss_meta_sqlite(all_texts, text_to_meta, db_path)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Original JSON: {json_path}")
    print(f"   New SQLite DB: {db_path}")
    print(f"\n💡 You can now delete the JSON file to save space.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SQLite utilities for semantic wiki search')
    parser.add_argument('--convert-faiss-meta', action='store_true', 
                       help='Convert FAISS meta JSON to SQLite')
    parser.add_argument('--json-path', type=str, 
                       help='Path to FAISS meta JSON file')
    parser.add_argument('--db-path', type=str, 
                       help='Path for output SQLite database')
    parser.add_argument('--build-linkgraph', action='store_true',
                       help='Build link graph database')
    
    args = parser.parse_args()
    
    if args.convert_faiss_meta:
        if not args.json_path or not args.db_path:
            print("Error: --json-path and --db-path required for conversion")
        else:
            convert_faiss_meta_json_to_sqlite(args.json_path, args.db_path)
    elif args.build_linkgraph:
        WIKI_LINK_GRAPH_JSONL_PATH = r"../data/processed/wiki_link_graph_jsonl"
        WIKI_LINK_GRAPH_DB_PATH = r"../data/processed/wiki_link_graph.db"
        build_linkgraph_sqlite(WIKI_LINK_GRAPH_JSONL_PATH, WIKI_LINK_GRAPH_DB_PATH)
        conn = load_linkgraph_sqlite(WIKI_LINK_GRAPH_DB_PATH)
        row = get_links_for_title_sqlite(conn, "Heroic couplet")
        print(row)
    else:
        print("No action specified. Use --help for options.")