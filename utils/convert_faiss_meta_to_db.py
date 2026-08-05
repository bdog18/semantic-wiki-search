#!/usr/bin/env python3
"""
One-time conversion script to convert paragraphs.index.meta.json to SQLite database.
This will significantly reduce memory usage and improve loading times.

Usage:
    python convert_faiss_meta_to_db.py
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sqlite_lookup import convert_faiss_meta_json_to_sqlite

# Configuration
FAISS_META_JSON_PATH = "../data/processed/faiss_index/paragraphs.index.meta.json"
FAISS_META_DB_PATH = "../data/processed/faiss_index/paragraphs.index.meta.db"

def main():
    print("=" * 70)
    print("FAISS Metadata JSON to SQLite Conversion")
    print("=" * 70)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.abspath(os.path.join(script_dir, FAISS_META_JSON_PATH))
    db_path = os.path.abspath(os.path.join(script_dir, FAISS_META_DB_PATH))
    
    print(f"\nInput JSON:  {json_path}")
    print(f"Output DB:   {db_path}\n")
    
    # Check if JSON exists
    if not os.path.exists(json_path):
        print(f"❌ ERROR: JSON file not found at {json_path}")
        print(f"\nPlease update FAISS_META_JSON_PATH in this script to point to your .meta.json file")
        return 1
    
    # Get JSON file size
    json_size_gb = os.path.getsize(json_path) / (1024**3)
    print(f"Original JSON size: {json_size_gb:.2f} GB")
    
    # Confirm conversion
    if os.path.exists(db_path):
        response = input(f"\n⚠️  Database already exists at {db_path}\nOverwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Conversion cancelled.")
            return 0
    
    print("\n🚀 Starting conversion...")
    print("⏳ This may take 10-30 minutes depending on file size...\n")
    
    # Perform conversion
    try:
        convert_faiss_meta_json_to_sqlite(json_path, db_path)
        
        # Show size comparison
        if os.path.exists(db_path):
            db_size_gb = os.path.getsize(db_path) / (1024**3)
            savings_pct = ((json_size_gb - db_size_gb) / json_size_gb) * 100
            
            print(f"\n{'='*70}")
            print("✅ CONVERSION COMPLETE!")
            print(f"{'='*70}")
            print(f"Original JSON: {json_size_gb:.2f} GB")
            print(f"New SQLite DB: {db_size_gb:.2f} GB")
            print(f"Space saved:   {json_size_gb - db_size_gb:.2f} GB ({savings_pct:.1f}%)")
            print(f"\n💡 Next steps:")
            print(f"   1. Update your code to use: {db_path}")
            print(f"   2. Test that triplet mining works with the new database")
            print(f"   3. Delete the old JSON file to free up space:")
            print(f"      rm {json_path}")
            print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
