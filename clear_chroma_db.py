#!/usr/bin/env python3
"""
Script to delete all data from ChromaDB
This will:
1. Delete all collections (agents and tools)
2. Optionally delete the entire database directory
"""

import logging
import os
import shutil
from chroma_inserter import create_chroma_inserter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_chroma_collections():
    """Delete all collections from ChromaDB"""
    try:
        chroma = create_chroma_inserter()
        
        # List all collections
        collections = chroma.client.list_collections()
        logger.info(f"Found {len(collections)} collection(s) in ChromaDB")
        
        # Delete each collection
        for collection in collections:
            collection_name = collection.name
            logger.info(f"Deleting collection: {collection_name}")
            chroma.client.delete_collection(name=collection_name)
            logger.info(f"✓ Deleted collection: {collection_name}")
        
        logger.info("✅ All collections deleted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting collections: {e}")
        import traceback
        traceback.print_exc()
        return False

def clear_chroma_directory():
    """Delete the entire ChromaDB directory"""
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    abs_path = os.path.abspath(persist_dir)
    
    if os.path.exists(abs_path):
        logger.info(f"Deleting ChromaDB directory: {abs_path}")
        try:
            shutil.rmtree(abs_path)
            logger.info(f"✅ Deleted ChromaDB directory: {abs_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting directory: {e}")
            return False
    else:
        logger.info(f"ChromaDB directory does not exist: {abs_path}")
        return True

def main():
    """Main function"""
    print("=" * 80)
    print("🗑️  ChromaDB Data Cleanup Script")
    print("=" * 80)
    print("\nThis will delete:")
    print("  1. All collections (agents, tools)")
    print("  2. Optionally: the entire database directory")
    print()
    
    response = input("Are you sure you want to delete all ChromaDB data? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    print("\nStep 1: Deleting collections...")
    if clear_chroma_collections():
        print("✅ Collections deleted")
    else:
        print("❌ Failed to delete collections")
        return
    
    print("\nStep 2: Delete database directory?")
    response2 = input("Delete the entire ./chroma_db directory? (yes/no): ").strip().lower()
    
    if response2 in ['yes', 'y']:
        if clear_chroma_directory():
            print("✅ Database directory deleted")
        else:
            print("❌ Failed to delete directory")
    else:
        print("⏭ Skipping directory deletion")
    
    print("\n" + "=" * 80)
    print("✅ Cleanup completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python3 insert_tools_script.py")
    print("  2. This will recreate agents and tools in ChromaDB")

if __name__ == "__main__":
    main()

