#!/usr/bin/env python3
"""
Test script for ChromaDB and MongoDB methods
Tests: chroma_find_agent, chroma_find_tools, mongo_get_tool, mongo_get_agent
"""

import json
from insert_tools_chroma_mongo import (
    chroma_find_agent,
    chroma_find_tools,
    mongo_get_tool,
    mongo_get_agent,
    insert_agent,
    insert_tool
)
from agent_structure import PI_AGENT, RUNRUN_AGENT
from neo4j_tool_structure import ALL_TOOLS_ENHANCED


def test_chroma_find_agent():
    """Test chroma_find_agent method"""
    print("\n" + "=" * 80)
    print("🔍 Testing chroma_find_agent()")
    print("=" * 80)
    
    test_queries = [
        "platform intelligence",
        "security vulnerability",
        "dataverse",
        "token generation"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        agent_ids = chroma_find_agent(query)
        print(f"   Found {len(agent_ids)} agent(s): {agent_ids}")
        if agent_ids:
            for agent_id in agent_ids:
                agent_info = mongo_get_agent(agent_id)
                if agent_info:
                    print(f"      - {agent_info.get('name', agent_id)}")


def test_chroma_find_tools():
    """Test chroma_find_tools method"""
    print("\n" + "=" * 80)
    print("🔍 Testing chroma_find_tools()")
    print("=" * 80)
    
    # First, find some agents
    print("\n📝 Step 1: Finding agents...")
    agent_ids = chroma_find_agent("platform intelligence")
    if not agent_ids:
        print("   ⚠ No agents found. Trying with 'security'...")
        agent_ids = chroma_find_agent("security")
    
    if not agent_ids:
        print("   ❌ No agents found. Please insert agents first.")
        return
    
    print(f"   Found agents: {agent_ids}")
    
    # Now search for tools with these agent IDs
    test_queries = [
        "token generation",
        "create schema",
        "vulnerability check",
        "data ingestion"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}' (filtered by agents: {agent_ids})")
        tool_ids = chroma_find_tools(query, agent_ids)
        print(f"   Found {len(tool_ids)} tool(s): {tool_ids}")
        if tool_ids:
            for tool_id in tool_ids:
                tool_info = mongo_get_tool(tool_id)
                if tool_info:
                    print(f"      - {tool_info.get('name', tool_id)}")


def test_mongo_get_tool():
    """Test mongo_get_tool method"""
    print("\n" + "=" * 80)
    print("🔍 Testing mongo_get_tool()")
    print("=" * 80)
    
    # Try to get some known tool IDs
    test_tool_ids = [
        "token_generation_api",
        "create_dataverse_api",
        "vulnerability_check_api",
        "ingest_data_api"
    ]
    
    for tool_id in test_tool_ids:
        print(f"\n📝 Getting tool: '{tool_id}'")
        tool = mongo_get_tool(tool_id)
        if tool:
            print(f"   ✅ Found tool: {tool.get('name', tool_id)}")
            print(f"   Description: {tool.get('description', 'N/A')[:100]}...")
            print(f"   Keywords: {tool.get('keywords', [])[:3]}")
        else:
            print(f"   ❌ Tool not found")


def test_mongo_get_agent():
    """Test mongo_get_agent method"""
    print("\n" + "=" * 80)
    print("🔍 Testing mongo_get_agent()")
    print("=" * 80)
    
    # Try to get known agent IDs
    test_agent_ids = [
        "pi_agent",
        "runrun_agent"
    ]
    
    for agent_id in test_agent_ids:
        print(f"\n📝 Getting agent: '{agent_id}'")
        agent = mongo_get_agent(agent_id)
        if agent:
            print(f"   ✅ Found agent: {agent.get('name', agent_id)}")
            print(f"   Description: {agent.get('description', 'N/A')[:100]}...")
            print(f"   Tools: {agent.get('tools', [])}")
        else:
            print(f"   ❌ Agent not found")


def setup_test_data():
    """Insert test data if not already present"""
    print("\n" + "=" * 80)
    print("📦 Setting up test data...")
    print("=" * 80)
    
    # Insert agents
    print("\n🤖 Inserting agents...")
    for agent in [PI_AGENT, RUNRUN_AGENT]:
        print(f"   Inserting: {agent['name']}")
        success = insert_agent(agent)
        if success:
            print(f"      ✅ Success")
        else:
            print(f"      ⚠ Failed or already exists")
    
    # Insert a few tools
    print("\n🔧 Inserting tools...")
    tools_to_insert = ALL_TOOLS_ENHANCED[:3]  # Insert first 3 tools
    for tool in tools_to_insert:
        print(f"   Inserting: {tool['name']}")
        success = insert_tool(tool)
        if success:
            print(f"      ✅ Success")
        else:
            print(f"      ⚠ Failed or already exists")


def main():
    """Run all tests"""
    print("=" * 80)
    print("🧪 Testing ChromaDB and MongoDB Methods")
    print("=" * 80)
    
    # Ask user if they want to setup test data
    print("\n⚠️  Note: This script assumes agents and tools are already inserted.")
    print("   If not, you can run: python3 insert_tools_chroma_mongo.py --all")
    
    response = input("\nDo you want to insert test data first? (y/n): ").strip().lower()
    if response == 'y':
        setup_test_data()
    
    # Run all tests
    try:
        test_chroma_find_agent()
        test_chroma_find_tools()
        test_mongo_get_tool()
        test_mongo_get_agent()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

