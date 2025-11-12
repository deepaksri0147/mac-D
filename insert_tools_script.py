#!/usr/bin/env python3
"""
Insert API Tools into Neo4j with Embeddings
Supports batch insertion and individual tool addition
"""

import logging
import sys
from typing import List, Dict, Any
from neo4j_tool_structure import ALL_TOOLS_ENHANCED
from neo4j_enhanced_manager import create_neo4j_manager
from embedding_service_ollama import create_ollama_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def insert_tool(tool: Dict[str, Any]) -> bool:
    """
    Insert a single tool into Neo4j
    
    Args:
        tool: Tool dictionary with all fields
        
    Returns:
        True if successful
    """
    try:
        # Initialize services
        neo4j_manager = create_neo4j_manager()
        embedding_service = create_ollama_embedding_service()
        
        # Create embedding
        logger.info(f"Creating embedding for: {tool['tool_id']}")
        
        # Combine relevant fields for embedding
        embedding_text = f"""
        {tool['name']}
        {tool['description']}
        {' '.join(tool.get('keywords', []))}
        {' '.join(tool.get('example_prompts', []))}
        """
        
        embedding = embedding_service.create_embedding(embedding_text.strip())
        logger.info(f"✓ Embedding created (dimension: {len(embedding)})")
        
        # Insert into Neo4j
        logger.info(f"Inserting tool: {tool['tool_id']}")
        success = neo4j_manager.insert_tool(tool, embedding)
        
        neo4j_manager.close()
        
        if success:
            logger.info(f"✅ '{tool['tool_id']}' added successfully\n")
            return True
        else:
            logger.error(f"❌ Failed to add '{tool['tool_id']}'\n")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error adding tool: {e}")
        return False


def insert_all_tools_in_order() -> bool:
    """
    Insert all predefined tools in dependency order
    Order: token_generation → create_dataverse → create_mongo_schema
    
    Returns:
        True if all successful
    """
    print("=" * 80)
    print("🔧 Inserting All API Tools into Neo4j")
    print("=" * 80)
    print("\nOrder: Token Generation → Dataverse Creation → Schema Creation")
    print("-" * 80)
    
    results = []
    for i, tool in enumerate(ALL_TOOLS_ENHANCED, 1):
        print(f"\n[{i}/{len(ALL_TOOLS_ENHANCED)}] Processing: {tool['name']}")
        success = insert_tool(tool)
        results.append((tool['name'], success))
        
        if not success:
            print(f"\n❌ Failed to add {tool['name']}. Stopping batch insertion.")
            return False
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ All Tools Inserted Successfully!")
    print("=" * 80)
    print("\n📊 Summary:")
    for tool_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {tool_name}")
    
    print("\n🔗 Dependency Relationships:")
    print("   • token_generation_api (root - no dependencies)")
    print("   • create_dataverse_api → token_generation_api")
    print("   • create_mongo_schema_api → token_generation_api, create_dataverse_api")
    
    print("\n💡 Next Steps:")
    print("   1. Test with: python main_orchestrator.py")
    print("   2. Try prompt: 'generate token'")
    print("   3. Try prompt: 'use requestType as SUBTENANT and password as Gaian1234'")
    
    return True


def add_custom_tool_interactive():
    """Interactive mode to add a custom tool"""
    print("=" * 80)
    print("🔧 Add Custom API Tool to Neo4j")
    print("=" * 80)
    
    tool = {}
    
    # Basic info
    tool["tool_id"] = input("\n1. Enter tool_id (unique identifier): ").strip()
    if not tool["tool_id"]:
        print("❌ tool_id is required")
        return False
    
    tool["name"] = input("2. Enter tool name: ").strip()
    tool["description"] = input("3. Enter description: ").strip()
    
    # Keywords
    keywords_input = input("4. Enter keywords (comma-separated): ").strip()
    tool["keywords"] = [k.strip() for k in keywords_input.split(",") if k.strip()]
    
    # Schema - URL
    url = input("\n5. Enter API URL: ").strip()
    method = input("6. Enter HTTP method (GET/POST/PUT/DELETE) [POST]: ").strip().upper() or "POST"
    
    # Request body
    print("\n7. Enter request body fields (one per line, format: fieldName:defaultValue)")
    print("   Press Enter twice when done:")
    request_body = {}
    field_descriptions = {}
    
    while True:
        field_input = input("   ").strip()
        if not field_input:
            break
        
        if ":" in field_input:
            field_name, default_value = field_input.split(":", 1)
            request_body[field_name.strip()] = default_value.strip()
            
            # Ask for field description
            desc = input(f"     Description for '{field_name.strip()}': ").strip()
            if desc:
                field_descriptions[field_name.strip()] = desc
    
    # Required fields
    required_input = input("\n8. Enter required fields (comma-separated): ").strip()
    required_fields = [f.strip() for f in required_input.split(",") if f.strip()]
    
    # Authentication
    auth_required = input("9. Does this API require authentication? (y/n) [n]: ").strip().lower() == 'y'
    returns_token = input("10. Does this API return a token? (y/n) [n]: ").strip().lower() == 'y'
    
    # Dependencies
    deps_input = input("11. Enter dependency tool_ids (comma-separated) [none]: ").strip()
    dependencies = [d.strip() for d in deps_input.split(",") if d.strip()]
    
    # Build complete tool structure
    tool["schema"] = {
        "url": url,
        "method": method,
        "headers": {"Content-Type": "application/json"},
        "requestBody": request_body,
        "field_descriptions": field_descriptions,
        "required_fields": required_fields,
        "authentication_required": auth_required,
        "returns_token": returns_token,
        "token_field": "token" if returns_token else None
    }
    tool["dependencies"] = dependencies
    tool["example_prompts"] = []
    
    # Confirm and insert
    print("\n" + "=" * 80)
    print("📝 Tool Configuration:")
    import json
    print(json.dumps(tool, indent=2))
    print("=" * 80)
    
    confirm = input("\nInsert this tool? (y/n): ").strip().lower()
    if confirm == 'y':
        return insert_tool(tool)
    else:
        print("❌ Cancelled")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--all":
            # Insert all predefined tools
            success = insert_all_tools_in_order()
            sys.exit(0 if success else 1)
        
        elif command == "--custom":
            # Interactive custom tool addition
            success = add_custom_tool_interactive()
            sys.exit(0 if success else 1)
        
        elif command == "--help":
            print("""
Usage: python insert_tools.py [OPTIONS]

Options:
  --all       Insert all predefined API tools in order
  --custom    Interactive mode to add a custom tool
  --help      Show this help message

Examples:
  python insert_tools.py --all        # Insert all tools
  python insert_tools.py --custom     # Add custom tool interactively
            """)
            sys.exit(0)
        
        else:
            print(f"Unknown option: {command}")
            print("Use --help for usage information")
            sys.exit(1)
    
    else:
        # Default: insert all tools
        success = insert_all_tools_in_order()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
