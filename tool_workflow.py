#!/usr/bin/env python3
"""
tool_workflow.py

- Finds relevant tools using embeddings in Chroma (tools collection only).
- Uses Ollama (via langchain_ollama.ChatOllama) to filter/select best tool_ids.
- Loads full tool metadata from MongoDB for selected tools.
- Provides a simple agent invocation demo (demo_tool) to simulate execution.

Expectations / dependencies:
- create_chroma_inserter() -> returns an object with `.tool_col` that supports `.query(...)` and `.upsert(...)`.
  - query(...) should accept: query_embeddings=[...], n_results=int, include=["metadatas","distances","ids"]
  - The returned structure is expected to be similar to chromadb-python: dict with keys 'ids','metadatas','distances'
- create_mongo_tool_inserter() -> returns an object with `.tools` being a pymongo.Collection (so .find_one is available)
- create_ollama_embedding_service() -> returns an object with .create_embedding(text) -> List[float]
- Langchain Ollama packages: langchain_ollama and langchain_core.prompts present for LLM filtering.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from embedding_service_ollama import create_ollama_embedding_service
from chroma_inserter import create_chroma_inserter
from mongo_inserter import create_mongo_tool_inserter

logger = logging.getLogger(__name__)

# Ollama configuration (used for LLM filtering & optionally embeddings)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")


# ---------------------------
# Demo tool (simulated execution)
# ---------------------------
@tool
def demo_tool(tool_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Demo tool to simulate calling an API. Replace with real executor (APIExecutor) if needed.
    """
    logger.info(f"[demo_tool] Simulated call -> tool_id: {tool_id}, request_body: {request_body}")
    # Simulate result
    return {
        "status": "success",
        "tool_id": tool_id,
        "request_body": request_body,
        "message": f"Simulated execution of {tool_id}"
    }


# ---------------------------
# Dynamic prompt middleware
# ---------------------------
class ContextDict(ModelRequest):
    """Typed context for the dynamic prompt middleware (tools list)"""
    pass


@dynamic_prompt
def tool_prompt(request: ModelRequest) -> str:
    """
    Build a dynamic system prompt summarizing available tools to the LLM agent.
    Expects runtime.context["tools"] to be a list of tool dicts with 'name' and 'description'.
    """
    tools = request.runtime.context.get("tools", [])
    try:
        with open("prompts/dynamic_agent_prompt.md", "r") as f:
            base_prompt = f.read()
    except FileNotFoundError:
        base_prompt = (
            "You are a helpful assistant that can call tools. Available tools:\n\n{tool_info}\n\n"
            "When you want to call a tool, return a tool call in the correct format."
        )
    if tools:
        tool_info = "\n".join([f"- {t.get('name','<no-name>')}: {t.get('description','')}" for t in tools])
        return base_prompt.format(tool_info=tool_info)
    return "You are a helpful assistant."


# ---------------------------
# ToolWorkflow
# ---------------------------
class ToolWorkflow:
    """
    Tool discovery and selection workflow using:
      - Chroma (vector search on 'tools' collection)
      - MongoDB (for full tool metadata)
      - LLM (Ollama) to filter/select best tools
    """

    def __init__(self, max_retries: int = 3):
        self.embedding_service = create_ollama_embedding_service()
        self.chroma = create_chroma_inserter()
        self.mongo = create_mongo_tool_inserter()

        # LLM for selecting/filtering tools
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.max_retries = max_retries

        # Agent capable of executing tools (demo). Replace or extend with real executor if needed.
        self.agent = create_agent(self.llm, [demo_tool], middleware=[tool_prompt], context_schema=ContextDict)

    # ----------------------
    # Embedding + Search
    # ----------------------
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for query text using the embedding service."""
        logger.info("Creating query embedding for tool search")
        return self.embedding_service.create_embedding(query)

    def find_relevant_tools(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query Chroma 'tools' collection with the given embedding and return a list of
        candidate dicts: {tool_id, similarity_score, metadata}
        """
        logger.info("Querying Chroma tools collection for relevant tools")
        try:
            results = self.chroma.tool_col.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        # Normalise results shape: chroma returns nested lists per query
        ids_list = results.get("ids", [[]])
        metas_list = results.get("metadatas", [[]])
        dist_list = results.get("distances", [[]])

        candidates: List[Dict[str, Any]] = []
        if not ids_list or not metas_list:
            return []

        ids = ids_list[0]
        metas = metas_list[0]
        dists = dist_list[0] if dist_list and isinstance(dist_list, list) and len(dist_list) > 0 else [None] * len(ids)

        for idx, tid in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            dist = dists[idx] if idx < len(dists) else None
            # Some chroma backends return 'distance' where lower is better; convert to similarity if possible.
            similarity = None
            try:
                if dist is None:
                    similarity = None
                else:
                    # treat as distance in [0..1] -> similarity = 1 - distance for interpretability
                    similarity = 1.0 - float(dist)
            except Exception:
                similarity = None

            candidates.append({
                "tool_id": tid,
                "similarity_score": similarity,
                "metadata": meta
            })

        logger.info(f"Found {len(candidates)} tool candidates from Chroma")
        return candidates

    # ----------------------
    # LLM Filtering
    # ----------------------
    def filter_tools_with_llm(self, query: str, potential_tools: List[Dict[str, Any]]) -> List[str]:
        """
        Use LLM to rank/filter the potential tools. The LLM must return JSON like:
            {"tool_ids": ["tool-id-1", "tool-id-2", ...]}
        We'll pass only the metadata to the LLM to reduce token usage.
        """
        logger.info("Filtering candidate tools with LLM")

        # Prepare prompt: concise instruction + examples
        prompt_text = (
            "You are an expert at selecting the most relevant tools to satisfy a user's request.\n\n"
            "User Query: \"{query}\"\n\n"
            "Available Tools (JSON array of metadata objects):\n{potential_tools}\n\n"
            "Task: Return ONLY a JSON object with a single field 'tool_ids' that is a list of tool_id strings\n"
            "in order of relevance (most relevant first). Do NOT include any other text.\n\n"
            "Example response:\n{{\"tool_ids\": [\"token_generation_api\", \"create_dataverse_api\"]}}"
        )
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm

        # Send only metadata to the LLM to reduce token usage
        potential_tools_json = json.dumps([pt.get("metadata", {}) for pt in potential_tools], indent=2)

        for attempt in range(self.max_retries):
            try:
                response = chain.invoke({"query": query, "potential_tools": potential_tools_json})
                content = response.content.strip()
                parsed = json.loads(content)
                if isinstance(parsed, dict) and isinstance(parsed.get("tool_ids"), list):
                    tool_ids = [str(tid) for tid in parsed["tool_ids"]]
                    logger.info(f"LLM selected tool_ids: {tool_ids}")
                    return tool_ids
                else:
                    logger.warning(f"LLM returned unexpected structure: {content}")
            except json.JSONDecodeError:
                logger.warning(f"LLM returned non-JSON content on attempt {attempt+1}. Content: {response.content.strip() if 'response' in locals() else 'NO_RESPONSE'}")
            except Exception as e:
                logger.warning(f"Error during LLM tool filtering attempt {attempt+1}: {e}")

        logger.error("LLM tool filtering failed after retries; falling back to top chroma candidate(s)")
        return []

    # ----------------------
    # Load tool metadata from Mongo
    # ----------------------
    def load_tool_from_mongo(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full tool document from MongoDB tools collection.
        The returned document should be the canonical tool definition.
        """
        try:
            coll = self.mongo.tools
            doc = coll.find_one({"tool_id": tool_id})
            if doc:
                # Remove MongoDB _id for safety if present
                doc.pop("_id", None)
            return doc
        except Exception as e:
            logger.error(f"Failed to load tool {tool_id} from Mongo: {e}")
            return None

    # ----------------------
    # Public run method
    # ----------------------
    def run(self, user_query: str, top_k: int = 8, rerank_with_llm: bool = True) -> Dict[str, Any]:
        """
        Full tool discovery and selection flow:
          1. Embed user query
          2. Query Chroma tools collection to get candidates
          3. Optionally rerank/filter with LLM
          4. Load selected tool metadata from Mongo
        Returns { "selected_tools": [ ... ], "fallback_used": bool }
        """
        logger.info(f"ToolWorkflow.run: {user_query}")

        # 1. create embedding
        query_emb = self.create_query_embedding(user_query)

        # 2. find candidates in Chroma
        candidates = self.find_relevant_tools(query_emb, limit=top_k)
        if not candidates:
            logger.warning("No tool candidates found in Chroma")
            return {"selected_tools": [], "fallback_used": False}

        # 3. use LLM to filter/rank tool ids
        selected_ids: List[str] = []
        if rerank_with_llm:
            selected_ids = self.filter_tools_with_llm(user_query, candidates)

        # If LLM didn't return anything, fallback to top candidate(s)
        fallback_used = False
        if not selected_ids:
            fallback_used = True
            # pick top 1 by similarity (or top N)
            selected_ids = [c["tool_id"] for c in sorted(candidates, key=lambda x: (x.get("similarity_score") is not None, x.get("similarity_score")), reverse=True)[:1]]

        # 4. load full metadata from mongo (or fall back to chroma metadata if missing)
        selected_tools: List[Dict[str, Any]] = []
        for tid in selected_ids:
            tool_doc = self.load_tool_from_mongo(tid)
            if tool_doc:
                selected_tools.append(tool_doc)
            else:
                # fallback: find in candidate metadata
                chroma_meta = next((c["metadata"] for c in candidates if c["tool_id"] == tid), None)
                if chroma_meta:
                    selected_tools.append(chroma_meta)

        logger.info(f"ToolWorkflow: returning {len(selected_tools)} selected tools (fallback_used={fallback_used})")
        return {"selected_tools": selected_tools, "fallback_used": fallback_used}

    # ----------------------
    # Optional: execute selected tool(s) using the langchain agent (demo)
    # ----------------------
    def execute_with_agent(self, user_query: str, tools: List[Dict[str, Any]]) -> Any:
        """
        Invoke the langchain agent (which has demo_tool) with the tools available in context.
        This will let the LLM choose to call demo_tool(...) and the langchain runtime will run it.
        """
        try:
            # The agent expects a context with the tools list
            result = self.agent.invoke({"messages": [{"role": "user", "content": user_query}]}, context={"tools": tools})
            return result
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {"error": str(e)}


# ----------------------
# CLI / quick test
# ----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tw = ToolWorkflow()

    test_query = "ingest data into schema 67dcf66d5ccb2c54260fb156 with data: [{\"id\":\"1\",\"name\":\"Test\"}]"
    out = tw.run(test_query)
    print(json.dumps(out, indent=2))




































# import os
# import logging
# import json
# from typing import Dict, Any, List, TypedDict, Optional

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langchain.agents.middleware import dynamic_prompt, ModelRequest

# from embedding_service_ollama import OllamaEmbeddingService, create_ollama_embedding_service
# from neo4j_enhanced_manager import EnhancedNeo4jToolManager

# logger = logging.getLogger(__name__)

# # Ollama configuration
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
# MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

# @tool
# def demo_tool(tool_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     A demo tool that simulates calling an API with a tool ID and request body.
#     In a real scenario, this would make an HTTP request.
#     """
#     logger.info(f"Executing demo_tool with tool_id: {tool_id} and request_body: {request_body}")
#     # Simulate an API call
#     return {"status": "success", "tool_id": tool_id, "response": f"API call for {tool_id} was successful"}

# class Context(TypedDict):
#     tools: List[Dict[str, Any]]

# @dynamic_prompt
# def tool_prompt(request: ModelRequest) -> str:
#     """Generate system prompt based on available tools."""
#     tools = request.runtime.context.get("tools", [])
#     with open("prompts/dynamic_agent_prompt.md", "r") as f:
#         base_prompt = f.read()
#     if tools:
#         tool_info = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
#         return base_prompt.format(tool_info=tool_info)
#     return "You are a helpful assistant."

# class ToolWorkflow:
#     """
#     Manages the workflow for finding relevant tools based on agent search results
#     and executing a tool using an LLM agent.
#     """
#     def __init__(self, max_retries: int = 3):
#         self.embedding_service: OllamaEmbeddingService = create_ollama_embedding_service()
#         self.tool_manager: EnhancedNeo4jToolManager = EnhancedNeo4jToolManager()
#         self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
#         self.tools = [demo_tool]
#         self.max_retries = max_retries

#         self.tool_manager.create_indexes()

#         self.agent = create_agent(
#             self.llm,
#             self.tools,
#             middleware=[tool_prompt],
#             context_schema=Context
#         )


#     def find_relevant_tools(self, query: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
#         """
#         Finds relevant tools using vector search based on the user query.
#         """
#         logger.info(f"Creating embedding for query: '{query}'")
#         query_embedding = self.embedding_service.create_embedding(query)

#         logger.info(f"Searching for relevant tools with limit={limit}, threshold={similarity_threshold}")
#         return self.tool_manager.search_tools_by_embedding(
#             query_embedding=query_embedding,
#             limit=limit,
#             similarity_threshold=similarity_threshold
#         )
    
#     def filter_tools_with_llm(self, query: str, potential_tools: List[Dict[str, Any]]) -> List[str]:
#         """
#         Uses the LLM to filter and select the best tools from a list of potential tools.
#         Returns a list of tool_ids for the best tools.
#         """
#         logger.info(f"Filtering {len(potential_tools)} tools with LLM for query: '{query}'")
#         if not potential_tools:
#             return None

#         prompt_text = """You are an expert at selecting the most relevant tools to answer a user's query.
# Based on the user's query and the list of available tools, your task is to identify the best tools to use.

# User Query: "{query}"

# Available Tools:
# {potential_tools}

# Carefully review the user's query and the description and parameters of each tool.
# Your response MUST be a JSON object containing a list of `tool_ids` for the best tools to use, in order of relevance.
# Do not provide any explanation or extra text.

# Example response:
# {{
#   "tool_ids": ["tool-id-1", "tool-id-2"]
# }}"""
#         prompt = ChatPromptTemplate.from_template(prompt_text)
#         chain = prompt | self.llm
        
#         # Filter tool properties to reduce token count
#         filtered_tools = [
#             {
#                 "tool_id": tool.get("tool_id"),
#                 "name": tool.get("name"),
#                 "description": tool.get("description"),
#                 "requestBody": tool.get("requestBody"),
#                 "queryParameters": tool.get("queryParameters"),
#                 "field_descriptions": tool.get("field_descriptions"),
#             }
#             for tool in potential_tools
#         ]
#         potential_tools_json = json.dumps(filtered_tools, indent=2)

#         for attempt in range(self.max_retries):
#             logger.info(f"Attempt {attempt + 1}/{self.max_retries} to filter tools with LLM.")
#             try:
#                 response = chain.invoke({"query": query, "potential_tools": potential_tools_json})
                
#                 # The prompt expects a JSON object with a 'tool_ids' list
#                 result_json = json.loads(response.content.strip())
                
#                 if isinstance(result_json, dict) and "tool_ids" in result_json and isinstance(result_json["tool_ids"], list):
#                     best_tool_ids = result_json["tool_ids"]
#                     logger.info(f"LLM selected tools: {best_tool_ids}")
#                     return best_tool_ids
#                 else:
#                     logger.warning(f"LLM returned malformed JSON for tool filtering (attempt {attempt + 1}): {response.content.strip()}")
            
#             except json.JSONDecodeError as e:
#                 logger.warning(f"Failed to parse LLM's tool filtering output as JSON (attempt {attempt + 1}): {e}. Output: {response.content.strip()}")
#             except Exception as e:
#                 logger.error(f"Error during LLM tool filtering (attempt {attempt + 1}): {e}")

#         logger.error(f"Failed to get valid LLM tool filtering output after {self.max_retries} attempts.")
#         return []

#     def execute_tool_with_agent(self, query: str, tools: List[Dict[str, Any]]) -> Any:
#         """
#         Executes a tool using the LLM agent based on the user query and available tools.
#         """
#         if not tools:
#             logger.warning("No tools available for execution.")
#             return {"error": "No tools found to execute."}

#         # The `create_agent` function in the constructor already has the tools.
#         # We invoke the agent with the user's query.
#         logger.info(f"Invoking agent for query: '{query}'")
#         try:
#             result = self.agent.invoke(
#                 {"messages": [{"role": "user", "content": query}]},
#                 context={"tools": tools}
#             )
#             return result
#         except Exception as e:
#             logger.error(f"Error executing tool with agent: {e}")
#             return {"error": str(e)}

#     def run_workflow(self, query: str) -> Any:
#         """
#         Runs the complete workflow:
#         1. Find relevant tools using vector search.
#         2. Filter and select the best tools using an LLM.
#         3. Execute the agent with the selected tools.
#         """
#         logger.info(f"Starting tool workflow for query: '{query}'")

#         # 1. Find relevant tools via vector search
#         potential_tools = self.find_relevant_tools(query)
#         if not potential_tools:
#             logger.warning("No potential tools found from vector search.")
#             return {"error": "No relevant tools found."}

#         # 2. Filter tools with LLM to get the best ones
#         best_tool_ids = self.filter_tools_with_llm(query, potential_tools)
#         if not best_tool_ids:
#             logger.warning("LLM could not select any suitable tools.")
#             return []
        
#         # Find the full details of the selected tools
#         selected_tools = [tool for tool in potential_tools if tool.get("tool_id") in best_tool_ids]

#         # 3. Execute the agent with the selected tools
#         execution_result = self.execute_tool_with_agent(query, selected_tools)

#         logger.info("Tool workflow completed.")
#         return execution_result

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     tool_workflow = ToolWorkflow()

#     user_query = "call the ingest data api"
#     result = tool_workflow.run_workflow(user_query)

#     print("\n--- Tool Execution Result ---")
#     print(result)
#     print("---------------------------")