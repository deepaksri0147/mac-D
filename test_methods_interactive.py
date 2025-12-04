#!/usr/bin/env python3
"""
Interactive test script for ChromaDB and MongoDB methods
Simple command-line interface to test each method
"""

import json
from insert_tools_chroma_mongo import (
    chroma_find_agent,
    chroma_find_tools,
    mongo_get_tool,
    mongo_get_agent
)


def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2, default=str))


def test_find_agent():
    """Test chroma_find_agent"""
    print("\n" + "=" * 80)
    print("🔍 Test: chroma_find_agent(query)")
    print("=" * 80)
    query = input("\nEnter search query: ").strip()
    if not query:
        query = "platform intelligence"
        print(f"Using default query: '{query}'")
    
    print(f"\nSearching for agents with query: '{query}'...")
    agent_ids = chroma_find_agent(query)
    
    print(f"\n✅ Found {len(agent_ids)} agent(s):")
    for agent_id in agent_ids:
        print(f"   - {agent_id}")
        agent = mongo_get_agent(agent_id)
        if agent:
            print(f"     Name: {agent.get('name', 'N/A')}")
            print(f"     Description: {agent.get('description', 'N/A')[:80]}...")


def test_find_tools():
    """Test chroma_find_tools"""
    print("\n" + "=" * 80)
    print("🔍 Test: chroma_find_tools(query, agent_id_list)")
    print("=" * 80)
    
    query = input("\nEnter search query: ").strip()
    if not query:
        query = "token generation"
        print(f"Using default query: '{query}'")
    
    print("\nEnter agent IDs (comma-separated, or press Enter to find agents first):")
    agent_input = input().strip()
    
    if not agent_input:
        # Auto-find agents
        print("\nFinding agents first...")
        agent_ids = chroma_find_agent("platform intelligence")
        if not agent_ids:
            agent_ids = chroma_find_agent("security")
        if not agent_ids:
            print("❌ No agents found. Please insert agents first.")
            return
        print(f"Using found agents: {agent_ids}")
    else:
        agent_ids = [aid.strip() for aid in agent_input.split(",")]
    
    print(f"\nSearching for tools with query: '{query}' and agents: {agent_ids}...")
    tool_ids = chroma_find_tools(query, agent_ids)
    
    print(f"\n✅ Found {len(tool_ids)} tool(s):")
    for tool_id in tool_ids:
        print(f"   - {tool_id}")
        tool = mongo_get_tool(tool_id)
        if tool:
            print(f"     Name: {tool.get('name', 'N/A')}")
            print(f"     Description: {tool.get('description', 'N/A')[:80]}...")


def test_get_tool():
    """Test mongo_get_tool"""
    print("\n" + "=" * 80)
    print("🔍 Test: mongo_get_tool(tool_id)")
    print("=" * 80)
    tool_id = input("\nEnter tool_id: ").strip()
    if not tool_id:
        tool_id = "token_generation_api"
        print(f"Using default tool_id: '{tool_id}'")
    
    print(f"\nFetching tool: '{tool_id}'...")
    tool = mongo_get_tool(tool_id)
    
    if tool:
        print("\n✅ Tool found:")
        print_json(tool)
    else:
        print(f"\n❌ Tool '{tool_id}' not found")


def test_get_agent():
    """Test mongo_get_agent"""
    print("\n" + "=" * 80)
    print("🔍 Test: mongo_get_agent(agent_id)")
    print("=" * 80)
    agent_id = input("\nEnter agent_id: ").strip()
    if not agent_id:
        agent_id = "pi_agent"
        print(f"Using default agent_id: '{agent_id}'")
    
    print(f"\nFetching agent: '{agent_id}'...")
    agent = mongo_get_agent(agent_id)
    
    if agent:
        print("\n✅ Agent found:")
        print_json(agent)
    else:
        print(f"\n❌ Agent '{agent_id}' not found")


def main():
    """Interactive menu"""
    print("=" * 80)
    print("🧪 Interactive Test Menu - ChromaDB & MongoDB Methods")
    print("=" * 80)
    
    menu = {
        "1": ("chroma_find_agent(query)", test_find_agent),
        "2": ("chroma_find_tools(query, agent_id_list)", test_find_tools),
        "3": ("mongo_get_tool(tool_id)", test_get_tool),
        "4": ("mongo_get_agent(agent_id)", test_get_agent),
        "5": ("Run all tests", lambda: [test_find_agent(), test_find_tools(), test_get_tool(), test_get_agent()]),
        "0": ("Exit", None)
    }
    
    while True:
        print("\n" + "-" * 80)
        print("Select a test to run:")
        for key, (desc, _) in menu.items():
            print(f"  {key}. {desc}")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice in menu:
            try:
                menu[choice][1]()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

