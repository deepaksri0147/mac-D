import os
import logging
from typing import Dict, Any, List, Optional
import json # Import json for parsing LLM output

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from discovery_agent_manager import DiscoveryAgentManager, create_discovery_agent_manager
from embedding_service_ollama import OllamaEmbeddingService, create_ollama_embedding_service

logger = logging.getLogger(__name__)

# Ollama configuration from tempp.ipynb
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

class AgentWorkflow:
    """
    Manages the workflow for processing a user query, finding relevant agents,
    and invoking an LLM with the agent's context.
    """
    def __init__(self, max_retries: int = 3):
        self.embedding_service: OllamaEmbeddingService = create_ollama_embedding_service()
        self.discovery_agent_manager: DiscoveryAgentManager = create_discovery_agent_manager()
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.prompt_template_path = "prompts/agent_invoke_prompt.md" # Default path
        self.max_retries = max_retries

        # Ensure indexes are created for the agent manager
        self.discovery_agent_manager.create_indexes()

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from a markdown file."""
        try:
            with open(self.prompt_template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at {self.prompt_template_path}")
            return "You are a helpful AI assistant. Answer the following question: {query}\n\nAgent Context: {agent_context}"
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return "You are a helpful AI assistant. Answer the following question: {query}\n\nAgent Context: {agent_context}"

    def create_query_embedding(self, query: str) -> List[float]:
        """Creates an embedding for the given query string."""
        logger.info(f"Creating embedding for query: '{query}'")
        return self.embedding_service.create_embedding(query)

    def find_relevant_agents(self, query_embedding: List[float], limit: int = 1, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Finds relevant agents based on the query embedding."""
        logger.info(f"Searching for relevant agents with limit={limit}, threshold={similarity_threshold}")
        return self.discovery_agent_manager.search_agents_by_embedding(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold
        )

    def invoke_llm_with_agent(self, query: str, agent_context: Dict[str, Any]) -> str:
        """
        [DEPRECATED] Invokes the LLM with the user query and the context of a single relevant agent.
        This method will be refactored or removed in favor of rerank_agents_with_llm.
        """
        logger.info(f"[DEPRECATED] Invoking LLM for query: '{query}' with agent: '{agent_context.get('name')}'")
        raw_prompt_template = self._load_prompt_template() # This will now load the reranking prompt

        # The prompt template expects 'potential_agents', not 'agent_context' directly
        # For compatibility with old call, we wrap agent_context into a list for potential_agents
        potential_agents_list = [agent_context]
        
        prompt = ChatPromptTemplate.from_template(raw_prompt_template)
        chain = prompt | self.llm
        
        # We expect JSON output here, so we'll try to parse it
        try:
            response = chain.invoke({"query": query, "potential_agents": potential_agents_list})
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error invoking LLM with agent for reranking (compatibility mode): {e}")
            return "[]" # Return empty list if LLM invocation fails or parsing fails

    def rerank_agents_with_llm(self, query: str, potential_agents: List[Dict[str, Any]], session_id: Optional[str] = None, session_manager: Optional[Any] = None) -> List[str]:
        """
        Uses the LLM to filter and rerank the given potential agents based on the query.
        Returns a list of agent_ids in ranked order.
        """
        def log(event: str, data: Dict[str, Any]):
            if session_id and session_manager:
                log_entry = {"event": event, **data}
                session_manager.save_log(session_id, log_entry)

        logger.info(f"Reranking {len(potential_agents)} agents with LLM for query: '{query}'")
        log("agent_rerank_started", {"query": query, "potential_agents_count": len(potential_agents)})
        raw_prompt_template = self._load_prompt_template()
        
        prompt = ChatPromptTemplate.from_template(raw_prompt_template)
        chain = prompt | self.llm

        potential_agents_json = json.dumps(potential_agents, indent=2)
        
        log("agent_rerank_prompt", {"prompt": raw_prompt_template, "query": query, "potential_agents": potential_agents_json})

        for attempt in range(self.max_retries):
            logger.info(f"Attempt {attempt + 1}/{self.max_retries} to rerank agents with LLM.")
            try:
                response = chain.invoke({"query": query, "potential_agents": potential_agents_json})
                log("agent_rerank_llm_response", {"attempt": attempt + 1, "response": response.content.strip()})
                ranked_agent_ids = json.loads(response.content.strip())
                if isinstance(ranked_agent_ids, list) and all(isinstance(aid, str) for aid in ranked_agent_ids):
                    logger.info(f"LLM reranked agents: {ranked_agent_ids}")
                    return ranked_agent_ids
                else:
                    logger.warning(f"LLM returned malformed JSON for reranking (attempt {attempt + 1}): {response.content.strip()}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM's reranking output as JSON (attempt {attempt + 1}): {e}. Output: {response.content.strip()}")
            except Exception as e:
                logger.error(f"Error during LLM reranking (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                logger.info("Retrying LLM reranking...")
            
        logger.error(f"Failed to get valid LLM reranking output after {self.max_retries} attempts.")
        return []

    def run_workflow(self, query: str, session_id: Optional[str] = None, session_manager: Optional[Any] = None) -> List[str]:
        """
        Runs the complete workflow: embed query, find agents, rerank with LLM, and return ranked agent IDs.
        """
        def log(event: str, data: Dict[str, Any]):
            if session_id and session_manager:
                log_entry = {"event": event, **data}
                session_manager.save_log(session_id, log_entry)

        logger.info(f"Starting workflow for query: '{query}'")
        log("agent_workflow_started", {"query": query})

        query_embedding = self.create_query_embedding(query)
        relevant_agents = self.find_relevant_agents(query_embedding, limit=5)

        if not relevant_agents:
            logger.warning("No initial relevant agents found.")
            log("agent_vector_search_failed", {"reason": "No agents found."})
            return []
        log("agent_vector_search_completed", {"retrieved_agents_count": len(relevant_agents), "retrieved_agents": relevant_agents})

        # Rerank agents using the LLM
        ranked_agent_ids = self.rerank_agents_with_llm(query, relevant_agents, session_id, session_manager)

        if not ranked_agent_ids:
            logger.warning("LLM reranking returned no agents or failed.")
            log("agent_rerank_failed", {"reason": "LLM reranking failed or returned no agents."})
            return []
        
        logger.info(f"Workflow completed successfully. Ranked Agent IDs: {ranked_agent_ids}")
        log("agent_workflow_completed", {"ranked_agent_ids": ranked_agent_ids})
        return ranked_agent_ids

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    workflow = AgentWorkflow()
    

    user_query = "Platform Intelligence"
    
    
    ranked_agent_ids = workflow.run_workflow(user_query)
    print("\n--- Ranked Agent IDs ---")
    if ranked_agent_ids:
        for agent_id in ranked_agent_ids:
            print(f"- {agent_id}")
    else:
        print("No agents were ranked or found.")
    print("------------------------")