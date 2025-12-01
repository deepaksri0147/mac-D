#!/usr/bin/env python3
"""
chroma_inserter.py

Chroma helper for:
  - agents collection (agent embeddings + metadata)
  - tools collection  (tool embeddings + metadata)

It uses the embedding service factory `create_ollama_embedding_service()` (from your
existing embedding_service_ollama.py) to create embeddings with the model you configured.
Chroma client is created via chromadb.Client().

Functions / classes:
  - ChromaDBInserter: main helper with methods:
      - insert_agent(agent: dict)
      - insert_tool(tool: dict)
      - query_agents(query_embedding, n_results=5) -> chroma style dict
      - query_tools(query_embedding, n_results=5) -> chroma style dict
  - create_chroma_inserter() -> factory
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional

import chromadb
from chromadb import PersistentClient, HttpClient

from embedding_service_ollama import create_ollama_embedding_service

logger = logging.getLogger(__name__)


class ChromaDBInserter:
    def __init__(
        self,
        chroma_host: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        chroma_host: if you're using chroma server/http-api, you can pass host like "http://localhost:8000".
                    The chromadb Python client will auto-detect. If you have custom config, change Settings below.
        persist_directory: if using local persistence, set directory path (optional).
        """
        # Use new ChromaDB client API (no deprecated Settings)
        import os
        try:
            # Check for ChromaDB server URL in environment variable
            env_chroma_host = os.getenv("CHROMA_HOST")
            if chroma_host or env_chroma_host:
                # Use HTTP client for ChromaDB server
                host_to_use = chroma_host or env_chroma_host
                # Parse host:port from URL
                host = host_to_use.replace("http://", "").replace("https://", "")
                parts = host.split(":")
                host_only = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 8000
                self.client = HttpClient(host=host_only, port=port)
                logger.info(f"Using ChromaDB server at: {host_only}:{port}")
            else:
                # Use persistent storage by default
                # Store data in ./chroma_db directory (relative to current working directory)
                persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
                os.makedirs(persist_dir, exist_ok=True)
                self.client = PersistentClient(path=persist_dir)
                logger.info(f"Using persistent ChromaDB storage at: {os.path.abspath(persist_dir)}")

        except Exception as e:
            logger.warning(f"Chroma client init failed ({e}), falling back to persistent storage.")
            # Fallback to persistent storage
            persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            os.makedirs(persist_dir, exist_ok=True)
            self.client = PersistentClient(path=persist_dir)
            logger.info(f"Using persistent ChromaDB storage (fallback) at: {os.path.abspath(persist_dir)}")

        # Collections for agents and tools
        self.agent_col = self._get_or_create_collection("agents")
        self.tool_col = self._get_or_create_collection("tools")

        # Embedding service (Ollama)
        self.embedder = create_ollama_embedding_service()

    def _get_or_create_collection(self, name: str):
        try:
            col = self.client.get_collection(name=name)
            logger.info(f"Found existing Chroma collection: {name}")
            return col
        except Exception:
            logger.info(f"Creating Chroma collection: {name}")
            return self.client.create_collection(name=name)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata to ChromaDB-compatible format.
        ChromaDB only accepts str, int, float, bool, None in metadata.
        Lists and dicts are converted to JSON strings.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                # Convert lists and dicts to JSON strings
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def _desanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ChromaDB metadata back to original format.
        Parses JSON strings back to lists/dicts where applicable.
        """
        desanitized = {}
        for key, value in metadata.items():
            if value is None:
                desanitized[key] = None
            elif isinstance(value, str):
                # Try to parse as JSON (might be a JSON string from sanitization)
                try:
                    parsed = json.loads(value)
                    # Only use parsed value if it's a list or dict (was originally one)
                    if isinstance(parsed, (list, dict)):
                        desanitized[key] = parsed
                    else:
                        desanitized[key] = value
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, keep as string
                    desanitized[key] = value
            else:
                # Keep other types as-is
                desanitized[key] = value
        return desanitized

    # -------------------
    # Embedding helper
    # -------------------
    def _embed_text(self, text: str) -> List[float]:
        """
        Create an embedding using the Ollama embedding service.
        This uses create_ollama_embedding_service().create_embedding(text).
        """
        # Retry a couple times for robustness
        last_exc = None
        for i in range(3):
            try:
                emb = self.embedder.create_embedding(text)
                return emb
            except Exception as e:
                last_exc = e
                logger.warning(f"Embedding attempt {i+1} failed: {e}")
                time.sleep(0.5)
        logger.error(f"All embedding attempts failed: {last_exc}")
        raise last_exc

    # -------------------
    # Insert / Upsert
    # -------------------
    def insert_agent(self, agent: Dict[str, Any], embed_text: Optional[str] = None) -> None:
        """
        Insert/update an agent into the agents collection.
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

        emb = self._embed_text(text)
        # Sanitize metadata (convert lists/dicts to JSON strings)
        sanitized_metadata = self._sanitize_metadata(agent)
        # Upsert using ids
        self.agent_col.upsert(ids=[agent_id], embeddings=[emb], metadatas=[sanitized_metadata])
        logger.info(f"Upserted agent to Chroma: {agent_id}")

    def insert_tool(self, tool: Dict[str, Any], embed_text: Optional[str] = None) -> None:
        """
        Insert/update a tool into the tools collection.
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

        emb = self._embed_text(text)
        # Sanitize metadata (convert lists/dicts to JSON strings)
        sanitized_metadata = self._sanitize_metadata(tool)
        self.tool_col.upsert(ids=[tool_id], embeddings=[emb], metadatas=[sanitized_metadata])
        logger.info(f"Upserted tool to Chroma: {tool_id}")

    # -------------------
    # Query helpers
    # -------------------
    def query_agents(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Query agents collection with provided embedding.
        Returns Chroma-style dict with keys: ids, metadatas, distances
        Metadata is automatically desanitized (JSON strings parsed back to lists/dicts)
        """
        results = self.agent_col.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "distances"])
        # Desanitize metadata (parse JSON strings back to lists/dicts)
        if "metadatas" in results and results["metadatas"]:
            desanitized_metadatas = []
            for metadata_list in results["metadatas"]:
                desanitized_list = [self._desanitize_metadata(meta) if meta else meta for meta in metadata_list]
                desanitized_metadatas.append(desanitized_list)
            results["metadatas"] = desanitized_metadatas
        return results

    def query_tools(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Query tools collection with provided embedding.
        Returns Chroma-style dict with keys: ids, metadatas, distances
        Metadata is automatically desanitized (JSON strings parsed back to lists/dicts)
        """
        results = self.tool_col.query(query_embeddings=[query_embedding], n_results=n_results, include=["metadatas", "distances"])
        # Desanitize metadata (parse JSON strings back to lists/dicts)
        if "metadatas" in results and results["metadatas"]:
            desanitized_metadatas = []
            for metadata_list in results["metadatas"]:
                desanitized_list = [self._desanitize_metadata(meta) if meta else meta for meta in metadata_list]
                desanitized_metadatas.append(desanitized_list)
            results["metadatas"] = desanitized_metadatas
        return results


def create_chroma_inserter(chroma_host: Optional[str] = None) -> ChromaDBInserter:
    """
    Factory to create ChromaDBInserter.
    chroma_host example: 'http://localhost:8000'
    """
    return ChromaDBInserter(chroma_host=chroma_host)


# -------------------
# Quick test / demo
# -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inserter = create_chroma_inserter()
    sample_agent = {
        "agent_id": "pi_agent",
        "name": "PI Agent",
        "description": "Platform Intelligence Agent that handles dataverse and schemas.",
        "keywords": ["pi", "dataverse", "schema"],
        "example_prompts": ["create dataverse", "create schema"]
    }
    sample_tool = {
        "tool_id": "token_generation_api",
        "name": "Token Generation API",
        "description": "Generates authentication token",
        "keywords": ["token", "login"],
        "example_prompts": ["generate token"]
    }

    inserter.insert_agent(sample_agent)
    inserter.insert_tool(sample_tool)
    print("Inserted sample agent & tool into Chroma (agents/tools)")




# #!/usr/bin/env python3
# """
# chroma_inserter.py
# ------------------
# Stores embeddings for:
#   - Agents (in "agents" collection)
#   - Tools  (in "tools" collection)

# Embedding model: qwen3:30b via Ollama
# """

# import chromadb
# import requests
# import logging
# from typing import Dict, Any, List

# logger = logging.getLogger(__name__)

# OLLAMA_EMBED_URL = "http://ollama-keda.mobiusdtaas.ai/api/embeddings"
# EMBED_MODEL = "qwen3:30b"


# class ChromaDBInserter:
#     def __init__(self, chroma_host="http://localhost:8000"):
#         self.chroma = chromadb.HttpClient(host="localhost", port=8000)

#         # separate collections
#         self.agent_col = self._get_or_create("agents")
#         self.tool_col = self._get_or_create("tools")

#     def _get_or_create(self, name: str):
#         try:
#             return self.chroma.get_collection(name=name)
#         except:
#             return self.chroma.create_collection(name=name)

#     def _embed(self, text: str) -> List[float]:
#         try:
#             res = requests.post(
#                 OLLAMA_EMBED_URL,
#                 json={"model": EMBED_MODEL, "prompt": text},
#                 timeout=30
#             )

#             if res.status_code != 200:
#                 raise Exception(f"Ollama failed: {res.text}")

#             embedding = res.json().get("embedding", [])
#             if not embedding:
#                 raise Exception("Empty embedding returned")

#             return embedding

#         except Exception as e:
#             logger.error(f"Embedding error: {e}")
#             raise

#     # -------------------------
#     # Agent insertion
#     # -------------------------
#     def insert_agent(self, agent: Dict[str, Any]):
#         text = " ".join([
#             agent.get("name", ""),
#             agent.get("description", ""),
#             " ".join(agent.get("keywords", [])),
#             " ".join(agent.get("example_prompts", [])),
#         ])

#         embedding = self._embed(text)
#         agent_id = agent["agent_id"]

#         self.agent_col.upsert(
#             ids=[agent_id],
#             embeddings=[embedding],
#             metadatas=[agent]
#         )

#         logger.info(f"Inserted agent into Chroma: {agent_id}")

#     # -------------------------
#     # Tool insertion
#     # -------------------------
#     def insert_tool(self, tool: Dict[str, Any]):
#         text = " ".join([
#             tool.get("name", ""),
#             tool.get("description", ""),
#             " ".join(tool.get("keywords", [])),
#             " ".join(tool.get("example_prompts", [])),
#         ])

#         embedding = self._embed(text)
#         tool_id = tool["tool_id"]

#         self.tool_col.upsert(
#             ids=[tool_id],
#             embeddings=[embedding],
#             metadatas=[tool]
#         )

#         logger.info(f"Inserted tool into Chroma: {tool_id}")


# def create_chroma_inserter():
#     return ChromaDBInserter()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     sample_agent = {
#         "agent_id": "pi_agent",
#         "name": "PI Agent",
#         "description": "Platform Intelligence Agent",
#         "keywords": ["pi", "data", "schema"],
#         "example_prompts": ["create dataverse"]
#     }

#     inserter = create_chroma_inserter()
#     inserter.insert_agent(sample_agent)
