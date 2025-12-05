import os
import logging
import json
from typing import Dict, Any, List, TypedDict, Optional
import asyncio

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.messages import messages_from_dict
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from embedding_service_ollama import OllamaEmbeddingService, create_ollama_embedding_service
from session_manager import SessionManager
from api_executor import APIExecutor, create_api_executor
from rerank_workflow import rerank_with_llm
from insert_tools_chroma_mongo import (
    create_chroma_inserter,
    create_mongo_tool_inserter,
    mongo_get_tool,
    mongo_get_agent
)

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

api_executor = create_api_executor()
chroma_inserter = create_chroma_inserter()
mongo_inserter = create_mongo_tool_inserter()

@tool
def demo_tool(tool_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds a tool by its ID in MongoDB, retrieves its details, and executes an API call.
    """
    logger.info(f"Executing tool with tool_id: {tool_id} and request_body: {request_body}")
    
    try:
        tool_info = mongo_get_tool(tool_id)
        if not tool_info:
            error_message = f"Tool with tool_id '{tool_id}' not found in MongoDB."
            logger.error(error_message)
            return {"status": "error", "error": error_message}

        result = api_executor.execute_api(
            tool_info=tool_info,
            parameter_modifications=request_body
        )
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred in demo_tool: {e}", exc_info=True)
        return {"status": "error", "error": f"An unexpected error occurred: {e}"}


class Context(TypedDict):
    tools: List[Dict[str, Any]]

@dynamic_prompt
def tool_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on available tools."""
    tools = request.runtime.context.get("tools", [])
    with open("prompts/dynamic_agent_prompt.md", "r") as f:
        base_prompt = f.read()
    if tools:
        tool_info_list = []
        for tool in tools:
            tool_details = {
                "tool_id": tool.get("tool_id"),
                "name": tool.get("name"),
                "description": tool.get("description"),
                "requestBody": tool.get("requestBody"),
                "queryParameters": tool.get("queryParameters"),
                "field_descriptions": tool.get("field_descriptions"),
                "required_fields": tool.get("required_fields"),
                "authentication_required": tool.get("authentication_required", False),
                "returns_token": tool.get("returns_token", False),
            }
            tool_info_list.append(json.dumps(tool_details, indent=2))
        
        tool_info = "\n".join(tool_info_list)
        
        # Print the formatted tool information that will be passed to the agent
        print("--- Tools Provided to Agent ---")
        print(tool_info)
        print("-----------------------------")
        
        return base_prompt.replace("{tool_info}", tool_info)
    return "You are a helpful assistant."

class MultiStepToolWorkflow:
    """
    Manages a multi-step workflow for finding relevant agents, finding tools for that agent,
    and executing a sequence of tools using an LLM agent.
    """
    def __init__(self, max_retries: int = 3):
        self.embedding_service: OllamaEmbeddingService = create_ollama_embedding_service()
        self.session_manager = SessionManager()
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.tools = [demo_tool]
        self.max_retries = max_retries
        self.selected_tools = None

        self.agent = create_agent(
            self.llm,
            self.tools,
            middleware=[tool_prompt],
            context_schema=Context
        )

    def find_relevant_agents(self, query: str, limit: int = 5) -> List[str]:
        """
        Finds relevant agents using ChromaDB vector search.
        """
        logger.info(f"Searching for relevant agents with query: '{query}'")
        return chroma_inserter.chroma_find_agent(query, limit=limit)

    def find_relevant_tools(self, query: str, agent_id_list: List[str] = None, limit: int = 10) -> List[str]:
        """
        Finds relevant tool IDs using ChromaDB vector search, optionally filtered by agent_id.
        """
        logger.info(f"Searching for relevant tools with query: '{query}' and agent_ids: {agent_id_list}")
        return chroma_inserter.chroma_find_tools(query, agent_id_list=agent_id_list, limit=limit)

    def get_tool_details_from_mongo(self, tool_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieves full tool details from MongoDB for a list of tool_ids.
        """
        tool_details = []
        for tool_id in tool_ids:
            tool_info = mongo_get_tool(tool_id)
            if tool_info:
                tool_details.append(tool_info)
        return tool_details

    async def filter_tools_with_llm(self, query: str, potential_tools: List[Dict[str, Any]], session_id: Optional[str] = None) -> List[str]:
        """
        Uses the LLM to filter and select the best tools from a list of potential tools.
        Returns a list of tool_ids for the best tools.
        """
        logger.info(f"Filtering {len(potential_tools)} tools with LLM for query: '{query}'")
        if not potential_tools:
            return []

        with open("prompts/tool_filter_prompt.md", "r") as f:
            prompt_text = f.read()
            
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm
        
        filtered_tools = [
            {
                "tool_id": tool.get("tool_id"),
                "name": tool.get("name"),
                "description": tool.get("description"),
                "requestBody": tool.get("requestBody"),
                "queryParameters": tool.get("queryParameters"),
                "field_descriptions": tool.get("field_descriptions"),
            }
            for tool in potential_tools
        ]
        potential_tools_json = json.dumps(filtered_tools, indent=2)
        
        await self.session_manager.save_log(session_id, {"event": "tool_filter_prompt", "prompt": prompt_text, "query": query, "potential_tools": potential_tools_json})

        for attempt in range(self.max_retries):
            logger.info(f"Attempt {attempt + 1}/{self.max_retries} to filter tools with LLM.")
            try:
                response = await chain.ainvoke({"query": query, "potential_tools": potential_tools_json})
                await self.session_manager.save_log(session_id, {"event": "tool_filter_llm_response", "attempt": attempt + 1, "response": response.content.strip()})
                
                result_json = json.loads(response.content.strip())
                
                if isinstance(result_json, dict) and "tool_ids" in result_json and isinstance(result_json["tool_ids"], list):
                    best_tool_ids = result_json["tool_ids"]
                    logger.info(f"LLM selected tools: {best_tool_ids}")
                    return best_tool_ids
                else:
                    logger.warning(f"LLM returned malformed JSON for tool filtering (attempt {attempt + 1}): {response.content.strip()}")
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM's tool filtering output as JSON (attempt {attempt + 1}): {e}. Output: {response.content.strip()}")
            except Exception as e:
                logger.error(f"Error during LLM tool filtering (attempt {attempt + 1}): {e}")

        logger.error(f"Failed to get valid LLM tool filtering output after {self.max_retries} attempts.")
        return []

    async def execute_tool_with_agent(self, session_id: str, tools: List[Dict[str, Any]], max_steps: int = 5) -> Any:
        """
        Executes a tool or sequence of tools using a ReAct-style agent.
        The agent is expected to provide a final answer directly.
        """
        if not tools:
            logger.warning("No tools available for execution.")
            return {"error": "No tools found to execute."}

        history_dicts = self.session_manager.load_history(session_id)
        history = messages_from_dict(history_dicts)
        
        logger.info(f"--- Executing Agent (Session: {session_id}) ---")
        logger.info(f"Current history: {history}")

        try:
            # Invoke the agent with the current history
            result_dict = await self.agent.ainvoke(
                {"messages": history},
                context={"tools": tools}
            )
            
            # The agent's final answer is expected in the last message
            print("response from agent ",result_dict)
            agent_messages = result_dict['messages']
            
            # Save only the new messages from the agent
            new_messages = agent_messages[len(history):]
            for message in new_messages:
                await self.session_manager.save_message(session_id, message)
            
            final_answer = agent_messages[-1].content if agent_messages else "No response."

            # Return the final answer and the updated history
            logger.info(f"Agent provided final answer: {final_answer}")
            return {"final_answer": final_answer, "history": agent_messages}

        except Exception as e:
            logger.error(f"Error executing tool with agent: {e}")
            return {"error": str(e), "history": history}

    async def run_workflow(self, query: str, session_id: Optional[str] = None) -> Any:
        """
        Runs the complete workflow:
        1. Find relevant tools using vector search.
        2. Filter and select the best tools using an LLM.
        1. Find a relevant agent.
        2. Find relevant tools for that agent.
        3. Filter and select the best tools using an LLM.
        4. Execute the agent with the selected tools.
        """
        if session_id:
            logger.info(f"Continuing session {session_id} for query: '{query}'")
            history = self.session_manager.load_history(session_id)
        else:
            session_id = self.session_manager.create_session_id()
            logger.info(f"Starting new session {session_id} for query: '{query}'")
            history = []
            self.selected_tools = None
        
        # Add the new user message to the history and save it
        user_message = HumanMessage(content=query)
        history.append(user_message)
        await self.session_manager.save_message(session_id, user_message)
        if not self.selected_tools:
            print("--- No tools found in session, searching for new tools... ---")
            # 1. Find relevant agent(s) from ChromaDB
            await self.session_manager.save_log(session_id, {"event": "agent_search_chroma", "query": query})
            agent_ids = self.find_relevant_agents(query, limit=5) # Get top 5 agents
            if not agent_ids:
                log_entry = {"event": "agent_search_failed", "query": query}
                await self.session_manager.save_log(session_id, log_entry)
                logger.warning("No relevant agents found from vector search.")
                return {"error": "No relevant agents found for your query."}
            await self.session_manager.save_log(session_id, {"event": "agent_vector_search_completed", "retrieved_agent_ids": agent_ids})

            # 2. Rerank agents with LLM
            potential_agents = [mongo_get_agent(agent_id) for agent_id in agent_ids if mongo_get_agent(agent_id)]
            ranked_agent_ids = await rerank_with_llm(query, potential_agents, "agent", session_id, self.session_manager)
            if not ranked_agent_ids:
                logger.warning("LLM reranking failed or returned no agents. Falling back to vector search results.")
                ranked_agent_ids = agent_ids
            await self.session_manager.save_log(session_id, {"event": "agent_rerank_completed", "ranked_agent_ids": ranked_agent_ids})
            
            # 3. Find relevant tools for the top-ranked agents from ChromaDB
            await self.session_manager.save_log(session_id, {"event": "tool_search_chroma", "query": query, "agent_ids": ranked_agent_ids})
            potential_tool_ids = self.find_relevant_tools(query, agent_id_list=ranked_agent_ids, limit=10)
            if not potential_tool_ids:
                await self.session_manager.save_log(session_id, {"event": "tool_search_failed", "reason": "No potential tools found in ChromaDB for the selected agents."})
                logger.warning("No potential tools found from ChromaDB search.")
                return {"error": "No relevant tools found."}
            await self.session_manager.save_log(session_id, {"event": "tool_vector_search_completed", "retrieved_tool_ids": potential_tool_ids})
            
            # 4. Rerank tools with LLM
            potential_tools = self.get_tool_details_from_mongo(potential_tool_ids)
            ranked_tool_ids = await rerank_with_llm(query, potential_tools, "tool", session_id, self.session_manager)
            if not ranked_tool_ids:
                logger.warning("LLM tool reranking failed. Falling back to vector search results.")
                ranked_tool_ids = potential_tool_ids
            await self.session_manager.save_log(session_id, {"event": "tool_rerank_completed", "ranked_tool_ids": ranked_tool_ids})
            
            # 5. Filter tools with LLM to get the best ones to execute
            potential_tools = self.get_tool_details_from_mongo(ranked_tool_ids)
            await self.session_manager.save_log(session_id, {"event": "tool_filtering_started"})
            best_tool_ids = await self.filter_tools_with_llm(query, potential_tools, session_id=session_id)
            if not best_tool_ids:
                await self.session_manager.save_log(session_id, {"event": "tool_filtering_failed", "reason": "LLM could not select any suitable tools."})
                logger.warning("LLM could not select any suitable tools.")
                return []
            await self.session_manager.save_log(session_id, {"event": "tool_filtering_completed", "selected_tool_ids": best_tool_ids})

            # 6. Get the final selected tools' details from MongoDB
            self.selected_tools = self.get_tool_details_from_mongo(best_tool_ids)
        else:
            print("--- Reusing tools from the current session. ---")
        
        print("selected tools", self.selected_tools)
        execution_result = await self.execute_tool_with_agent(session_id, self.selected_tools)


        logger.info("Tool workflow completed.")
        return execution_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    workflow = MultiStepToolWorkflow()
    session_id = None

    print("--- Multi-Step Tool Workflow CLI ---")
    print("Type 'new' to start a new session, or 'exit' to quit.")

    async def main():
        global session_id
        while True:
            try:
                if session_id:
                    user_query = input(f"\nUser (Session: {session_id[:8]}): ")
                else:
                    user_query = input(f"\nUser: ")

                if user_query.lower() == 'exit':
                    print("Exiting workflow.")
                    break
                
                if user_query.lower() == 'new':
                    session_id = workflow.session_manager.create_session_id()
                    workflow.selected_tools = None
                    print(f"\n--- New session started: {session_id} ---")
                    continue

                if not user_query:
                    continue
                
                if not session_id:
                    session_id = workflow.session_manager.create_session_id()
                    print(f"--- New session started: {session_id} ---")

                result = await workflow.run_workflow(user_query, session_id=session_id)

                print("\n--- Agent Response ---")
                if "final_answer" in result:
                    print(result["final_answer"])
                elif "error" in result:
                    print(f"An error occurred: {result['error']}")
                
                print("----------------------")

            except KeyboardInterrupt:
                print("\nExiting workflow.")
                break
            except Exception as e:
                logger.error(f"A critical error occurred in the CLI: {e}")
                break
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting workflow.")