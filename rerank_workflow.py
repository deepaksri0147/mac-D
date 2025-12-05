import logging
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

async def rerank_with_llm(
    query: str,
    items: List[Dict[str, Any]],
    item_type: str, # "agent" or "tool"
    session_id: Optional[str] = None,
    session_manager: Optional[Any] = None,
    max_retries: int = 3,
    prompt_file: Optional[str] = None,
) -> List[str]:
    """
    Uses the LLM to filter and rerank a list of items (agents or tools).
    Returns a list of item IDs in ranked order.
    """
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
    
    async def log(event: str, data: Dict[str, Any]):
        if session_id and session_manager:
            log_entry = {"event": event, **data}
            await session_manager.save_log(session_id, log_entry)

    logger.info(f"Reranking {len(items)} {item_type}s with LLM for query: '{query}'")
    await log(f"{item_type}_rerank_started", {"query": query, f"potential_{item_type}s_count": len(items)})
    
    if not prompt_file:
        prompt_file = f"prompts/{item_type}_rerank_prompt.md"
        
    try:
        with open(prompt_file, "r") as f:
            raw_prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {prompt_file}")
        return []

    prompt = ChatPromptTemplate.from_template(raw_prompt_template)
    chain = prompt | llm

    items_json = json.dumps(items, indent=2)
    
    # Use a generic placeholder in the prompt, like {potential_items}
    # The prompt will need to be updated to use this generic placeholder
    prompt_input = {"query": query, "potential_items": items_json}
    
    await log(f"{item_type}_rerank_prompt", {"prompt": raw_prompt_template, "query": query, f"potential_{item_type}s": items_json})

    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries} to rerank {item_type}s with LLM.")
        try:
            response = await chain.ainvoke(prompt_input)
            await log(f"{item_type}_rerank_llm_response", {"attempt": attempt + 1, "response": response.content.strip()})
            
            result_json = json.loads(response.content.strip())
            
            id_key = f"ranked_agent_ids" if item_type == "agent" else "ranked_tool_ids"
            
            if isinstance(result_json, dict) and id_key in result_json and isinstance(result_json[id_key], list):
                ranked_ids = result_json[id_key]
                logger.info(f"LLM reranked {item_type}s: {ranked_ids}")
                return ranked_ids
            else:
                 logger.warning(f"LLM returned malformed JSON for {item_type} reranking (attempt {attempt + 1}): {response.content.strip()}")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM's {item_type} reranking output as JSON (attempt {attempt + 1}): {e}. Output: {response.content.strip()}")
        except Exception as e:
            logger.error(f"Error during LLM {item_type} reranking (attempt {attempt + 1}): {e}")

    logger.error(f"Failed to get valid LLM {item_type} reranking output after {max_retries} attempts.")
    return []