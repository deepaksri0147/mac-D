#!/usr/bin/env python3
"""
Quick script to check if agents are in ChromaDB
"""

import logging
import json
from chroma_inserter import create_chroma_inserter

logging.basicConfig(level=logging.INFO)

chroma = create_chroma_inserter()

# Get all agents from ChromaDB
try:
    # Get collection count
    count = chroma.agent_col.count()
    print(f"Total agents in ChromaDB: {count}")
    
    if count > 0:
        # Get all agents
        all_agents = chroma.agent_col.get()
        print(f"\nAgent IDs: {all_agents.get('ids', [])}")
        print(f"\nAgent Metadata:")
        for idx, agent_id in enumerate(all_agents.get('ids', [])):
            metadata = all_agents.get('metadatas', [])[idx] if idx < len(all_agents.get('metadatas', [])) else {}
            # Desanitize metadata (parse JSON strings back to lists/dicts)
            desanitized = chroma._desanitize_metadata(metadata)
            print(f"\n  {agent_id}:")
            print(f"    Name: {desanitized.get('name', 'N/A')}")
            keywords = desanitized.get('keywords', 'N/A')
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except:
                    pass
            print(f"    Keywords: {keywords}")
            example_prompts = desanitized.get('example_prompts', 'N/A')
            if isinstance(example_prompts, str):
                try:
                    example_prompts = json.loads(example_prompts)
                except:
                    pass
            print(f"    Example Prompts: {example_prompts}")
    else:
        print("\n⚠️  No agents found in ChromaDB!")
        print("   Run: python3 insert_tools_script.py")
        
except Exception as e:
    print(f"Error checking ChromaDB: {e}")
    import traceback
    traceback.print_exc()

