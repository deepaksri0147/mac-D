#!/usr/bin/env python3
"""
Insert API Tools and Agents into ChromaDB and MongoDB
- Agents: ChromaDB only
- Tools: MongoDB (full document) + ChromaDB (embedding + metadata)
"""

import logging
import sys
import json
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from pymongo import MongoClient
from pymongo.collection import ReturnDocument

from neo4j_tool_structure import ALL_TOOLS_ENHANCED
from agent_structure import ALL_AGENTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBInserter:
    """Handles insertion of agents and tools into ChromaDB"""
    
    def __init__(self, chroma_host: Optional[str] = None):
        """
        Initialize ChromaDB client
        
        Args:
            chroma_host: ChromaDB server URL (e.g., 'http://localhost:8000')
                        If None, checks environment variable CHROMA_HOST or defaults to 'http://localhost:8000'
        """
        import os
        # Get host from parameter, environment variable, or default
        # Since ChromaDB is in HTTP-only mode, we must use HttpClient
        host = chroma_host or os.getenv("CHROMA_HOST", "http://localhost:8000")
        
        # Extract hostname and port from URL if full URL provided
        if host.startswith("http://") or host.startswith("https://"):
            # Full URL provided
            pass
        else:
            # Just hostname provided, add http://
            host = f"http://{host}"
        
        # Always use HttpClient (required in HTTP-only mode)
        try:
            self.client = chromadb.HttpClient(host=host)
            logger.info(f"Connected to ChromaDB at {host}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB at {host}: {e}")
            raise RuntimeError(
                f"ChromaDB connection failed. Please ensure ChromaDB server is running at {host}.\n"
                "To start ChromaDB server, run: chroma run --host localhost --port 8000\n"
                "Or set CHROMA_HOST environment variable to your ChromaDB server URL."
            ) from e
        
        # Get or create a SINGLE collection for both agents and tools
        # We'll distinguish them via a metadata field: item_type = "agent" | "tool"
        self.collection = self.client.get_or_create_collection(
            name="agents_tools",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaDB client initialized")
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using ChromaDB's default embedding function
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # ChromaDB will generate embeddings automatically when we provide documents
        # For manual embedding, we need to access the embedding function
        # Try to get the embedding function from the collection
        try:
            # In newer ChromaDB versions, we can use the collection's embedding function
            if hasattr(self.collection, '_embedding_function'):
                return self.collection._embedding_function([text])[0]
            elif hasattr(self.collection, 'embedding_function'):
                # Some versions expose it differently
                return self.collection.embedding_function([text])[0]
            else:
                # Fallback: Let ChromaDB handle it by providing document instead of embedding
                # This will be handled in the insert methods
                raise AttributeError("Embedding function not accessible")
        except (AttributeError, TypeError):
            # If embedding function is not accessible, we'll use ChromaDB's automatic embedding
            # by providing documents instead of embeddings in the upsert call
            raise ValueError("Cannot access embedding function. Use document-based insertion instead.")
    
    def _sanitize_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata by converting complex types to JSON strings
        
        Args:
            doc: Document dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in doc.items():
            # Skip embedding fields
            if key == "embedding":
                continue
            # Convert complex types to JSON strings
            if isinstance(value, (dict, list)):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def insert_agent(self, agent: Dict[str, Any], embed_text: Optional[str] = None) -> None:
        """
        Insert/update an agent into the agents collection.
        
        Args:
            agent: must contain agent_id
            embed_text: optional text to embed; if None, will be constructed from agent fields
        """
        if "agent_id" not in agent:
            raise ValueError("agent must contain 'agent_id'")

        agent_id = agent["agent_id"]
        # Build comprehensive embedding text from all relevant fields
        if embed_text:
            text = embed_text
        else:
            parts = []
            parts.append(agent.get("name", ""))
            parts.append(agent.get("description", ""))
            # Add keywords
            keywords = agent.get("keywords", [])
            if keywords:
                parts.extend(keywords)
            # Add example prompts
            example_prompts = agent.get("example_prompts", [])
            if example_prompts:
                parts.extend(example_prompts)
            # Add tools list if available
            tools = agent.get("tools", [])
            if tools:
                parts.append(" ".join(tools))
            text = " ".join(filter(None, parts))  # Filter out empty strings

        # Sanitize metadata (convert lists/dicts to JSON strings)
        sanitized_metadata = self._sanitize_metadata(agent)
        # Mark this record as an agent
        sanitized_metadata["item_type"] = "agent"
        # Upsert using ids, documents (for automatic embedding), and metadata
        # ChromaDB will automatically generate embeddings from the document text
        self.collection.upsert(ids=[agent_id], documents=[text], metadatas=[sanitized_metadata])
        logger.info(f"Upserted agent to Chroma: {agent_id}")
    
    def insert_tool(self, tool: Dict[str, Any], embed_text: Optional[str] = None) -> None:
        """
        Insert/update a tool into the tools collection.
        
        Args:
            tool: must contain tool_id
            embed_text: optional text to embed; if None, will be constructed from tool fields
        """
        if "tool_id" not in tool:
            raise ValueError("tool must contain 'tool_id'")

        tool_id = tool["tool_id"]
        text = embed_text or " ".join([
            tool.get("name", ""),
            tool.get("description", ""),
            " ".join(tool.get("keywords", [])),
            " ".join(tool.get("example_prompts", []))
        ])

        # Sanitize metadata (convert lists/dicts to JSON strings)
        sanitized_metadata = self._sanitize_metadata(tool)
        # Mark this record as a tool
        sanitized_metadata["item_type"] = "tool"
        # Ensure agent_id is in metadata for filtering (if tool has agent_id)
        if "agent_id" in tool:
            sanitized_metadata["agent_id"] = tool["agent_id"]
        # Upsert using ids, documents (for automatic embedding), and metadata
        # ChromaDB will automatically generate embeddings from the document text
        self.collection.upsert(ids=[tool_id], documents=[text], metadatas=[sanitized_metadata])
        logger.info(f"Upserted tool to Chroma: {tool_id}")
    
    def chroma_find_agent(self, query: str, limit: int = 5) -> List[str]:
        """
        Search for agents in ChromaDB based on query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of agent IDs (strings)
        """
        try:
            # Query ChromaDB collection with filter for agents only
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"item_type": "agent"}  # Filter to only agents
            )
            
            # Extract agent IDs from results
            agent_ids = []
            if results and "ids" in results and len(results["ids"]) > 0:
                agent_ids = results["ids"][0]  # First query result
            
            logger.info(f"Found {len(agent_ids)} agents for query: '{query}'")
            return agent_ids
        except Exception as e:
            logger.error(f"Failed to search agents in ChromaDB: {e}", exc_info=True)
            return []
    
    def chroma_find_tools(self, query: str, agent_id_list: List[str], limit: int = 10) -> List[str]:
        """
        Search for tools in ChromaDB based on query, filtered by agent IDs.
        
        Args:
            query: Search query string
            agent_id_list: List of agent IDs to filter tools by
            limit: Maximum number of results to return
            
        Returns:
            List of tool IDs (strings)
        """
        try:
            # Base filter for tools
            where_clause = {"item_type": "tool"}
            
            # If agent_id_list is provided, add it to the filter
            if agent_id_list:
                # Use $in operator for a list of agent IDs
                where_clause = {
                    "$and": [
                        {"item_type": "tool"},
                        {"agent_id": {"$in": agent_id_list}}
                    ]
                }
            
            # Query ChromaDB collection
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause
            )
            
            # Extract tool IDs from results
            tool_ids = []
            if results and "ids" in results and len(results["ids"]) > 0:
                tool_ids = results["ids"][0]
            
            logger.info(f"Found {len(tool_ids)} tools for query: '{query}' with agents: {agent_id_list if agent_id_list else 'all'}")
            return tool_ids
        except Exception as e:
            logger.error(f"Failed to search tools in ChromaDB: {e}", exc_info=True)
            return []


class MongoToolInserter:
    """Handles insertion of tools and agents into MongoDB"""
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "agentProd"):
        """
        Initialize MongoDB client
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.tools = self.db.tools
        self.agents = self.db.agents  # Add agents collection
        logger.info(f"MongoDB client initialized: {db_name}")
    
    def insert_tool(self, tool: Dict[str, Any]) -> bool:
        """
        Upsert tool document by tool_id.
        
        Args:
            tool: Tool dictionary with tool_id
            
        Returns:
            True if success, False otherwise
        """
        if "tool_id" not in tool:
            logger.error("Tool must contain 'tool_id'")
            return False

        try:
            # Prepare document (do not store Mongo _id passed by user)
            tool_doc = dict(tool)
            tool_doc.pop("_id", None)

            res = self.tools.find_one_and_update(
                {"tool_id": tool_doc["tool_id"]},
                {"$set": tool_doc},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            logger.info(f"Upserted tool in Mongo: {tool_doc['tool_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert tool in Mongo: {e}", exc_info=True)
            return False
    
    def insert_agent(self, agent: Dict[str, Any]) -> bool:
        """
        Upsert agent document by agent_id.
        
        Args:
            agent: Agent dictionary with agent_id
            
        Returns:
            True if success, False otherwise
        """
        if "agent_id" not in agent:
            logger.error("Agent must contain 'agent_id'")
            return False

        try:
            # Prepare document (do not store Mongo _id passed by user)
            agent_doc = dict(agent)
            agent_doc.pop("_id", None)

            res = self.agents.find_one_and_update(
                {"agent_id": agent_doc["agent_id"]},
                {"$set": agent_doc},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            logger.info(f"Upserted agent in Mongo: {agent_doc['agent_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert agent in Mongo: {e}", exc_info=True)
            return False
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tool document by tool_id from MongoDB.
        
        Args:
            tool_id: Tool ID to retrieve
            
        Returns:
            Full tool JSON object or None if not found
        """
        try:
            tool = self.tools.find_one({"tool_id": tool_id})
            if tool:
                # Remove MongoDB's _id field for cleaner output
                tool.pop("_id", None)
                return tool
            return None
        except Exception as e:
            logger.error(f"Failed to get tool from Mongo: {e}", exc_info=True)
            return None
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent document by agent_id from MongoDB.
        
        Args:
            agent_id: Agent ID to retrieve
            
        Returns:
            Full agent JSON object or None if not found
        """
        try:
            agent = self.agents.find_one({"agent_id": agent_id})
            if agent:
                # Remove MongoDB's _id field for cleaner output
                agent.pop("_id", None)
                return agent
            return None
        except Exception as e:
            logger.error(f"Failed to get agent from Mongo: {e}", exc_info=True)
            return None


def create_chroma_inserter(chroma_host: Optional[str] = None) -> ChromaDBInserter:
    """
    Factory to create ChromaDBInserter.
    
    Args:
        chroma_host: ChromaDB server URL (e.g., 'http://localhost:8000')
                    If None, uses persistent client
    """
    return ChromaDBInserter(chroma_host=chroma_host)


def create_mongo_tool_inserter(uri: str = "mongodb://localhost:27017", db_name: str = "agentProd") -> MongoToolInserter:
    """
    Factory to create MongoToolInserter.
    
    Args:
        uri: MongoDB connection URI
        db_name: Database name
    """
    return MongoToolInserter(uri=uri, db_name=db_name)


def insert_agent(agent: Dict[str, Any]) -> bool:
    """
    Insert a single agent:
      1) Store canonical agent document in MongoDB agents collection (upsert)
      2) Store embedding + light metadata in Chroma agents_tools collection
    agent: must contain at least 'agent_id', 'name', 'description' (recommended: keywords, example_prompts)
    """
    mongo = create_mongo_tool_inserter()
    chroma = create_chroma_inserter()
    
    try:
        # 1) Insert/update in Mongo
        ok_mongo = mongo.insert_agent(agent)
        if not ok_mongo:
            logger.error(f"Mongo insertion failed for agent: {agent.get('agent_id')}")
            return False
        
        # 2) Insert embedding + metadata in Chroma agents_tools collection
        chroma.insert_agent(agent)
        logger.info(f"Inserted/updated agent in Chroma: {agent.get('agent_id')}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert agent: {e}", exc_info=True)
        return False


def insert_tool(tool: Dict[str, Any]) -> bool:
    """
    Insert a single tool:
      1) Store canonical tool document in MongoDB tools collection (upsert)
      2) Store embedding + light metadata in Chroma tools collection
    """
    mongo = create_mongo_tool_inserter()
    chroma = create_chroma_inserter()

    try:
        # 1) Insert/update in Mongo
        ok_mongo = mongo.insert_tool(tool)
        if not ok_mongo:
            logger.error(f"Mongo insertion failed for tool: {tool.get('tool_id')}")
            return False

        # 2) Insert embedding + metadata in Chroma tools collection
        chroma.insert_tool(tool)
        logger.info(f"Inserted/updated tool in Chroma: {tool.get('tool_id')}")
        return True

    except Exception as e:
        logger.error(f"Failed to insert tool: {e}", exc_info=True)
        return False


def chroma_find_agent(query: str) -> List[str]:
    """
    Search for agents in ChromaDB based on query.
    
    Args:
        query: Search query string
        
    Returns:
        List of agent IDs (strings)
    """
    chroma = create_chroma_inserter()
    try:
        return chroma.chroma_find_agent(query)
    except Exception as e:
        logger.error(f"Failed to find agents: {e}", exc_info=True)
        return []


def chroma_find_tools(query: str, agent_id_list: List[str]) -> List[str]:
    """
    Search for tools in ChromaDB based on query, filtered by agent IDs.
    
    Args:
        query: Search query string
        agent_id_list: List of agent IDs to filter tools by
        
    Returns:
        List of tool IDs (strings)
    """
    chroma = create_chroma_inserter()
    try:
        return chroma.chroma_find_tools(query, agent_id_list)
    except Exception as e:
        logger.error(f"Failed to find tools: {e}", exc_info=True)
        return []


def mongo_get_tool(tool_id: str) -> Optional[Dict[str, Any]]:
    """
    Get tool document by tool_id from MongoDB.
    
    Args:
        tool_id: Tool ID to retrieve
        
    Returns:
        Full tool JSON object or None if not found
    """
    mongo = create_mongo_tool_inserter()
    try:
        return mongo.get_tool(tool_id)
    except Exception as e:
        logger.error(f"Failed to get tool: {e}", exc_info=True)
        return None


def mongo_get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Get agent document by agent_id from MongoDB.
    
    Args:
        agent_id: Agent ID to retrieve
        
    Returns:
        Full agent JSON object or None if not found
    """
    mongo = create_mongo_tool_inserter()
    try:
        return mongo.get_agent(agent_id)
    except Exception as e:
        logger.error(f"Failed to get agent: {e}", exc_info=True)
        return None


def insert_all_agents() -> bool:
    """
    Insert all agents into ChromaDB
    
    Returns:
        True if all successful
    """
    print("=" * 80)
    print("🤖 Inserting All Agents into ChromaDB")
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
    Insert all predefined tools into MongoDB and ChromaDB
    
    Returns:
        True if all successful
    """
    print("=" * 80)
    print("🔧 Inserting All API Tools into MongoDB and ChromaDB")
    print("=" * 80)
    print("\nOrder: Token Generation → Dataverse Creation → Schema Creation → Data Ingestion → Vulnerability Check")
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
    
    print("\n💡 Next Steps:")
    print("   1. Tools are stored in MongoDB (full documents)")
    print("   2. Tools are stored in ChromaDB (embeddings + metadata)")
    print("   3. Agents are stored in ChromaDB only")
    
    return True


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--agents":
            # Insert all agents into ChromaDB
            success = insert_all_agents()
            sys.exit(0 if success else 1)
        
        elif command == "--tools":
            # Insert all tools into MongoDB and ChromaDB
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
        
        elif command == "--help":
            print("""
Usage: python insert_tools_chroma_mongo.py [OPTIONS]

Options:
  --agents    Insert all agents into ChromaDB
  --tools     Insert all tools into MongoDB and ChromaDB
  --all       Insert agents first, then all tools (recommended)
  --help      Show this help message

Examples:
  python insert_tools_chroma_mongo.py --all        # Insert agents and tools (recommended)
  python insert_tools_chroma_mongo.py --agents      # Insert only agents
  python insert_tools_chroma_mongo.py --tools       # Insert only tools

Storage:
  - Agents: ChromaDB only
  - Tools: MongoDB (full document) + ChromaDB (embedding + metadata)
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

