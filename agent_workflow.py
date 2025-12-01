import os
import logging
import json
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from embedding_service_ollama import create_ollama_embedding_service
from chroma_inserter import create_chroma_inserter

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")


class AgentWorkflow:
    def __init__(self, max_retries: int = 3):
        self.embedding_service = create_ollama_embedding_service()
        self.chroma = create_chroma_inserter()
        # LLM for reranking/filtering
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.prompt_template_path = os.getenv("AGENT_RERANK_PROMPT", "prompts/agent_invoke_prompt.md")
        self.max_retries = max_retries

    def _load_prompt_template(self) -> str:
        try:
            with open(self.prompt_template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Prompt template not found, using fallback template")
            return "You are an assistant that ranks agents. Query: {query}\nCandidates: {potential_agents}"

    def create_query_embedding(self, query: str) -> List[float]:
        logger.info(f"Creating embedding for query: '{query}'")
        return self.embedding_service.create_embedding(query)

    def find_relevant_agents(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Query Chroma's AGENTS collection only and return candidate agent metadatas with scores."""
        logger.info("Searching agents in Chroma...")
        # Use chroma collection query API
        results = self.chroma.agent_col.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "distances"]
        )

        logger.debug(f"Chroma query results: {results}")
        
        agents = []
        # chroma returns lists: results['ids'][0], results['metadatas'][0], results['distances'][0]
        ids_list = results.get("ids", [[]])
        if not ids_list or not ids_list[0]:
            logger.warning("No agent IDs returned from ChromaDB query")
            return agents
        
        ids = ids_list[0]
        metadatas_list = results.get("metadatas", [[]])
        distances_list = results.get("distances", [[]])
        
        metadatas = metadatas_list[0] if metadatas_list and metadatas_list[0] else []
        distances = distances_list[0] if distances_list and distances_list[0] else []
        
        logger.info(f"Found {len(ids)} agent(s) in ChromaDB")
        
        for idx, aid in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            # Desanitize metadata (parse JSON strings back to lists/dicts)
            if metadata:
                metadata = self.chroma._desanitize_metadata(metadata)
            distance = distances[idx] if idx < len(distances) else None
            similarity = 1.0 - distance if distance is not None else None
            logger.info(f"  Agent {idx+1}: {aid} (similarity: {similarity:.3f})")
            agents.append({
                "agent_id": aid,
                "similarity_score": similarity,
                "metadata": metadata
            })
        return agents

    def rerank_agents_with_llm(self, query: str, potential_agents: List[Dict[str, Any]]) -> List[str]:
        """Use the LLM to rerank agent candidates. Returns list of agent_ids."""
        logger.info(f"Reranking {len(potential_agents)} agents with LLM for query: '{query}'")
        raw_prompt_template = self._load_prompt_template()
        prompt = ChatPromptTemplate.from_template(raw_prompt_template)
        chain = prompt | self.llm

        potential_agents_json = json.dumps([p.get("metadata", {}) for p in potential_agents], indent=2)

        for attempt in range(self.max_retries):
            try:
                response = chain.invoke({"query": query, "potential_agents": potential_agents_json})
                # Expect LLM to return JSON array of agent_ids (e.g. ["pi_agent", "runrun_agent"])
                content = response.content.strip()
                ranked = json.loads(content)
                if isinstance(ranked, list):
                    return ranked
                else:
                    logger.warning("LLM returned non-list for agent rerank, retrying")
            except Exception as e:
                logger.warning(f"Rerank attempt failed: {e}")
        logger.error("Failed to get valid rerank output from LLM")
        return []

    def run_workflow(self, query: str, top_k: int = 5, rerank_with_llm: bool = True) -> List[str]:
        logger.info(f"AgentWorkflow: processing query '{query}'")
        q_emb = self.create_query_embedding(query)
        candidates = self.find_relevant_agents(q_emb, limit=top_k)
        if not candidates:
            logger.warning("No agent candidates from Chroma")
            return []

        if rerank_with_llm:
            ranked_ids = self.rerank_agents_with_llm(query, candidates)
            if ranked_ids:
                return ranked_ids
            else:
                # fallback to chroma ordering
                return [c['agent_id'] for c in candidates]
        else:
            return [c['agent_id'] for c in candidates]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    aw = AgentWorkflow()
    print(aw.run_workflow("create dataverse for testing"))


























# import os
# import logging
# from typing import Dict, Any, List, Optional
# import json # Import json for parsing LLM output

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate

# from discovery_agent_manager import DiscoveryAgentManager, create_discovery_agent_manager
# from embedding_service_ollama import OllamaEmbeddingService, create_ollama_embedding_service

# logger = logging.getLogger(__name__)

# # Ollama configuration from tempp.ipynb
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
# MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

# class AgentWorkflow:
#     """
#     Manages the workflow for processing a user query, finding relevant agents,
#     and invoking an LLM with the agent's context.
#     """
#     def __init__(self, max_retries: int = 3):
#         self.embedding_service: OllamaEmbeddingService = create_ollama_embedding_service()
#         self.discovery_agent_manager: DiscoveryAgentManager = create_discovery_agent_manager()
#         self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
#         self.prompt_template_path = "prompts/agent_invoke_prompt.md" # Default path
#         self.max_retries = max_retries

#         # Ensure indexes are created for the agent manager
#         self.discovery_agent_manager.create_indexes()

#     def _load_prompt_template(self) -> str:
#         """Loads the prompt template from a markdown file."""
#         try:
#             with open(self.prompt_template_path, "r") as f:
#                 return f.read()
#         except FileNotFoundError:
#             logger.error(f"Prompt template file not found at {self.prompt_template_path}")
#             return "You are a helpful AI assistant. Answer the following question: {query}\n\nAgent Context: {agent_context}"
#         except Exception as e:
#             logger.error(f"Error loading prompt template: {e}")
#             return "You are a helpful AI assistant. Answer the following question: {query}\n\nAgent Context: {agent_context}"

#     def create_query_embedding(self, query: str) -> List[float]:
#         """Creates an embedding for the given query string."""
#         logger.info(f"Creating embedding for query: '{query}'")
#         return self.embedding_service.create_embedding(query)

#     def find_relevant_agents(self, query_embedding: List[float], limit: int = 1, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
#         """Finds relevant agents based on the query embedding."""
#         logger.info(f"Searching for relevant agents with limit={limit}, threshold={similarity_threshold}")
#         return self.discovery_agent_manager.search_agents_by_embedding(
#             query_embedding=query_embedding,
#             limit=limit,
#             similarity_threshold=similarity_threshold
#         )

#     def invoke_llm_with_agent(self, query: str, agent_context: Dict[str, Any]) -> str:
#         """
#         [DEPRECATED] Invokes the LLM with the user query and the context of a single relevant agent.
#         This method will be refactored or removed in favor of rerank_agents_with_llm.
#         """
#         logger.info(f"[DEPRECATED] Invoking LLM for query: '{query}' with agent: '{agent_context.get('name')}'")
#         raw_prompt_template = self._load_prompt_template() # This will now load the reranking prompt

#         # The prompt template expects 'potential_agents', not 'agent_context' directly
#         # For compatibility with old call, we wrap agent_context into a list for potential_agents
#         potential_agents_list = [agent_context]
        
#         prompt = ChatPromptTemplate.from_template(raw_prompt_template)
#         chain = prompt | self.llm
        
#         # We expect JSON output here, so we'll try to parse it
#         try:
#             response = chain.invoke({"query": query, "potential_agents": potential_agents_list})
#             return response.content.strip()
#         except Exception as e:
#             logger.error(f"Error invoking LLM with agent for reranking (compatibility mode): {e}")
#             return "[]" # Return empty list if LLM invocation fails or parsing fails

#     def rerank_agents_with_llm(self, query: str, potential_agents: List[Dict[str, Any]]) -> List[str]:
#         """
#         Uses the LLM to filter and rerank the given potential agents based on the query.
#         Returns a list of agent_ids in ranked order.
#         """
#         logger.info(f"Reranking {len(potential_agents)} agents with LLM for query: '{query}'")
#         raw_prompt_template = self._load_prompt_template()
        
#         prompt = ChatPromptTemplate.from_template(raw_prompt_template)
#         chain = prompt | self.llm

#         # Convert list of dicts to JSON string for the prompt
#         potential_agents_json = json.dumps(potential_agents, indent=2)

#         for attempt in range(self.max_retries):
#             logger.info(f"Attempt {attempt + 1}/{self.max_retries} to rerank agents with LLM.")
#             try:
#                 response = chain.invoke({"query": query, "potential_agents": potential_agents_json})
#                 # Attempt to parse the LLM's output as a JSON array of agent_ids
#                 ranked_agent_ids = json.loads(response.content.strip())
#                 if isinstance(ranked_agent_ids, list) and all(isinstance(aid, str) for aid in ranked_agent_ids):
#                     logger.info(f"LLM reranked agents: {ranked_agent_ids}")
#                     return ranked_agent_ids
#                 else:
#                     logger.warning(f"LLM returned malformed JSON for reranking (attempt {attempt + 1}): {response.content.strip()}")
#             except json.JSONDecodeError as e:
#                 logger.warning(f"Failed to parse LLM's reranking output as JSON (attempt {attempt + 1}): {e}. Output: {response.content.strip()}")
#             except Exception as e:
#                 logger.error(f"Error during LLM reranking (attempt {attempt + 1}): {e}")
            
#             if attempt < self.max_retries - 1:
#                 logger.info("Retrying LLM reranking...")
            
#         logger.error(f"Failed to get valid LLM reranking output after {self.max_retries} attempts.")
#         return []

#     def run_workflow(self, query: str) -> List[str]:
#         """
#         Runs the complete workflow: embed query, find agents, rerank with LLM, and return ranked agent IDs.
#         """
#         logger.info(f"Starting workflow for query: '{query}'")
#         query_embedding = self.create_query_embedding(query)
#         relevant_agents = self.find_relevant_agents(query_embedding, limit=5) # Fetch more agents for reranking

#         if not relevant_agents:
#             logger.warning("No initial relevant agents found.")
#             return []

#         # Rerank agents using the LLM
#         ranked_agent_ids = self.rerank_agents_with_llm(query, relevant_agents)

#         if not ranked_agent_ids:
#             logger.warning("LLM reranking returned no agents or failed.")
#             return []
        
#         # If the goal is just to return the ranked agent IDs
#         logger.info(f"Workflow completed successfully. Ranked Agent IDs: {ranked_agent_ids}")
#         return ranked_agent_ids

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     workflow = AgentWorkflow()
    

#     user_query = "Platform Intelligence"
    
    
#     ranked_agent_ids = workflow.run_workflow(user_query)
#     print("\n--- Ranked Agent IDs ---")
#     if ranked_agent_ids:
#         for agent_id in ranked_agent_ids:
#             print(f"- {agent_id}")
#     else:
#         print("No agents were ranked or found.")
#     print("------------------------")