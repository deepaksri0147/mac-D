#!/usr/bin/env python3
"""
Insert API Tools into Neo4j with Embeddings
Supports batch insertion and individual tool addition
"""

import logging
import sys
from typing import List, Dict, Any
from neo4j_tool_structure import ALL_TOOLS_ENHANCED
from agent_structure import ALL_AGENTS
from neo4j_enhanced_manager import create_neo4j_manager
from discovery_agent_manager import create_discovery_agent_manager
from embedding_service_ollama import create_ollama_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def insert_agent(agent: Dict[str, Any]) -> bool:
    """
    Insert a single agent into discovery agent DB
    
    Args:
        agent: Agent dictionary with all fields
        
    Returns:
        True if successful
    """
    try:
        # Initialize services
        discovery_manager = create_discovery_agent_manager()
        embedding_service = create_ollama_embedding_service()
        
        # Create embedding
        logger.info(f"Creating embedding for agent: {agent['agent_id']}")
        
        # Combine relevant fields for embedding
        embedding_text = f"""
        {agent['name']}
        {agent['description']}
        {' '.join(agent.get('keywords', []))}
        {' '.join(agent.get('example_prompts', []))}
        """
        
        embedding = embedding_service.create_embedding(embedding_text.strip())
        logger.info(f"✓ Embedding created (dimension: {len(embedding)})")
        
        # Insert into discovery agent DB
        logger.info(f"Inserting agent: {agent['agent_id']}")
        success = discovery_manager.insert_agent(agent, embedding)
        
        discovery_manager.close()
        
        if success:
            logger.info(f"✅ Agent '{agent['agent_id']}' added successfully\n")
            return True
        else:
            logger.error(f"❌ Failed to add agent '{agent['agent_id']}'\n")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error adding agent: {e}")
        return False


def insert_tool(tool: Dict[str, Any], agent_database: str = None) -> bool:
    """
    Insert a single tool into agent-specific Neo4j database
    
    Args:
        tool: Tool dictionary with all fields
        agent_database: Name of the agent's database (e.g., "piagent", "runrun")
                       If None, will try to get from tool's agent_id
        
    Returns:
        True if successful
    """
    try:
        # Get agent database from tool if not provided
        if not agent_database:
            agent_id = tool.get("agent_id")
            if agent_id:
                # Map agent_id to database name
                agent_db_map = {
                    "pi_agent": "neo4j",
                    "runrun_agent": "neo4j"
                }
                agent_database = agent_db_map.get(agent_id)
                logger.info(f"Using agent_database '{agent_database}' for tool '{tool.get('tool_id')}'")
                if not agent_database:
                    logger.error(f"❌ Unknown agent_id: {agent_id}")
                    return False
            else:
                logger.error(f"❌ Tool {tool.get('tool_id')} has no agent_id")
                return False
        
        # Initialize services
        neo4j_manager = create_neo4j_manager(database=agent_database)
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
        
        # Insert into agent's Neo4j database
        logger.info(f"Inserting tool: {tool['tool_id']} into database: {agent_database}")
        success = neo4j_manager.insert_tool(tool, embedding)
        
        neo4j_manager.close()
        
        if success:
            logger.info(f"✅ '{tool['tool_id']}' added successfully to {agent_database}\n")
            return True
        else:
            logger.error(f"❌ Failed to add '{tool['tool_id']}'\n")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error adding tool: {e}")
        return False


def insert_all_agents() -> bool:
    """
    Insert all agents into discovery agent DB
    
    Returns:
        True if all successful
    """
    print("=" * 80)
    print("🤖 Inserting All Agents into Discovery Agent DB")
    print("=" * 80)
    print("-" * 80)
    
    results = []
    for i, agent in enumerate(ALL_AGENTS, 1):
        print(f"\n[{i}/{len(ALL_AGENTS)}] Processing: {agent['name']}")
        success = insert_agent(agent)
        results.append((agent['name'], success))
        
        if not success:
            print(f"\n❌ Failed to add {agent['name']}. Stopping batch insertion.")
            return False
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ All Agents Inserted Successfully!")
    print("=" * 80)
    print("\n📊 Summary:")
    for agent_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {agent_name}")
    
    return True


def insert_all_tools_in_order() -> bool:
    """
    Insert all predefined tools into their respective agent databases
    Order: token_generation → create_dataverse → create_mongo_schema
    
    Returns:
        True if all successful
    """
    print("=" * 80)
    print("🔧 Inserting All API Tools into Agent-Specific Databases")
    print("=" * 80)
    print("\nOrder: Token Generation → Dataverse Creation → Schema Creation → Data Ingestion → Vulnerability Check")
    print("-" * 80)
    
    # Group tools by agent
    tools_by_agent = {}
    for tool in ALL_TOOLS_ENHANCED:
        agent_id = tool.get("agent_id")
        if not agent_id:
            logger.warning(f"⚠ Tool {tool.get('tool_id')} has no agent_id, skipping")
            continue
        
        if agent_id not in tools_by_agent:
            tools_by_agent[agent_id] = []
        tools_by_agent[agent_id].append(tool)
    
    results = []
    tool_count = 0
    for agent_id, tools in tools_by_agent.items():
        # Map agent_id to database name
        agent_db_map = {
            "pi_agent": "neo4j",
            "runrun_agent": "neo4j"
        }
        agent_database = agent_db_map.get(agent_id)
        if not agent_database:
            logger.error(f"❌ Unknown agent_id: {agent_id}")
            continue
        
        print(f"\n📦 Processing {len(tools)} tools for {agent_id} (database: {agent_database})")
        
        for tool in tools:
            tool_count += 1
            print(f"\n[{tool_count}/{len(ALL_TOOLS_ENHANCED)}] Processing: {tool['name']}")
            success = insert_tool(tool, agent_database)
            results.append((tool['name'], agent_database, success))
            
            if not success:
                print(f"\n❌ Failed to add {tool['name']}. Stopping batch insertion.")
                return False
    
    # Rebuild dependency relationships for each agent database
    print("\n" + "=" * 80)
    print("🔗 Rebuilding Dependency Relationships")
    print("=" * 80)
    
    agent_db_map = {
        "pi_agent": "piagent",
        "runrun_agent": "runrun"
    }
    
    for agent_id, agent_database in agent_db_map.items():
        print(f"\n📦 Rebuilding relationships for {agent_id} (database: {agent_database})")
        neo4j_manager = create_neo4j_manager(database=agent_database)
        neo4j_manager.rebuild_dependency_relationships()
        neo4j_manager.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ All Tools Inserted Successfully!")
    print("=" * 80)
    print("\n📊 Summary:")
    for tool_name, agent_db, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {tool_name} → {agent_db}")
    
    print("\n🔗 Dependency Relationships:")
    print("   PI Agent:")
    print("   • token_generation_api (root - no dependencies)")
    print("   • create_dataverse_api → token_generation_api")
    print("   • create_mongo_schema_api → token_generation_api, create_dataverse_api")
    print("   • ingest_data_api → token_generation_api, create_mongo_schema_api")
    print("   RunRun Agent:")
    print("   • runrun_token_api (root - no dependencies)")
    print("   • vulnerability_check_api → runrun_token_api")
    
    print("\n💡 Next Steps:")
    print("   1. Test with: python main_orchestrator.py")
    print("   2. Try prompt: 'generate token'")
    print("   3. Try prompt: 'use requestType as SUBTENANT and password as Gaian1234'")
    print("   4. Try prompt: 'can u run the vulnerability check'")
    
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
        
        if command == "--agents":
            # Insert all agents into discovery agent DB
            success = insert_all_agents()
            sys.exit(0 if success else 1)
        
        elif command == "--tools":
            # Insert all tools into agent-specific databases
            success = insert_all_tools_in_order()
            sys.exit(0 if success else 1)
        
        elif command == "--all":
            # Insert agents first, then tools
            print("Step 1: Inserting agents...")
            agents_success = insert_all_agents()
            if not agents_success:
                print("❌ Failed to insert agents. Aborting.")
                sys.exit(1)
            
            print("\n" + "=" * 80)
            print("Step 2: Inserting tools...")
            tools_success = insert_all_tools_in_order()
            
            sys.exit(0 if (agents_success and tools_success) else 1)
        
        elif command == "--custom":
            # Interactive custom tool addition
            success = add_custom_tool_interactive()
            sys.exit(0 if success else 1)
        
        elif command == "--help":
            print("""
Usage: python insert_tools.py [OPTIONS]

Options:
  --agents    Insert all agents into discovery agent DB
  --tools     Insert all tools into agent-specific databases
  --all       Insert agents first, then all tools (recommended)
  --custom    Interactive mode to add a custom tool
  --help      Show this help message

Examples:
  python insert_tools.py --all        # Insert agents and tools (recommended)
  python insert_tools.py --agents      # Insert only agents
  python insert_tools.py --tools       # Insert only tools
  python insert_tools.py --custom      # Add custom tool interactively
            """)
            sys.exit(0)
        
        else:
            print(f"Unknown option: {command}")
            print("Use --help for usage information")
            sys.exit(1)
    
    else:
        # Default: insert agents and tools
        print("Step 1: Inserting agents...")
        agents_success = insert_all_agents()
        if not agents_success:
            print("❌ Failed to insert agents. Aborting.")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("Step 2: Inserting tools...")
        tools_success = insert_all_tools_in_order()
        
        sys.exit(0 if (agents_success and tools_success) else 1)


if __name__ == "__main__":
    main()
