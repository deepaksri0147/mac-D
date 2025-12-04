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
from neo4j_enhanced_manager import EnhancedNeo4jToolManager
from agent_workflow import AgentWorkflow
from session_manager import SessionManager
from api_executor import APIExecutor, create_api_executor

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

tool_manager = EnhancedNeo4jToolManager()
api_executor = create_api_executor()

@tool
def demo_tool(tool_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds a tool by its ID in Neo4j, retrieves its details, and executes an API call.
    """
    logger.info(f"Executing tool with tool_id: {tool_id} and request_body: {request_body}")
    
    try:
        tool_info = tool_manager.get_tool_by_id(tool_id)
        if not tool_info:
            error_message = f"Tool with tool_id '{tool_id}' not found."
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
        self.agent_workflow = AgentWorkflow(max_retries=max_retries)
        self.session_manager = SessionManager()
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.tools = [demo_tool]
        self.max_retries = max_retries
        self.selected_tools = None

        tool_manager.create_indexes()

        self.agent = create_agent(
            self.llm,
            self.tools,
            middleware=[tool_prompt],
            context_schema=Context
        )


    def find_relevant_tools(self, query: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Finds relevant tools using vector search based on the user query.
        """
        logger.info(f"Creating embedding for query: '{query}'")
        query_embedding = self.embedding_service.create_embedding(query)

        logger.info(f"Searching for relevant tools with limit={limit}, threshold={similarity_threshold}")
        return tool_manager.search_tools_by_embedding(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
    
    async def filter_tools_with_llm(self, query: str, potential_tools: List[Dict[str, Any]], session_id: Optional[str] = None) -> List[str]:
        """
        Uses the LLM to filter and select the best tools from a list of potential tools.
        Returns a list of tool_ids for the best tools.
        """
        logger.info(f"Filtering {len(potential_tools)} tools with LLM for query: '{query}'")
        if not potential_tools:
            return None

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
            # 1. Find relevant agent(s)
            await self.session_manager.save_log(session_id, {"event": "agent_search", "query": query})
            ranked_agent_ids = await self.agent_workflow.run_workflow(query, session_id=session_id, session_manager=self.session_manager)
            if not ranked_agent_ids:
                log_entry = {"event": "agent_search_failed", "query": query}
                await self.session_manager.save_log(session_id, log_entry)
                logger.warning("No relevant agents found.")
                return {"error": "No relevant agents found for your query."}
            await self.session_manager.save_log(session_id, {"event": "agent_search_completed", "ranked_agent_ids": ranked_agent_ids})
            
            # For now, let's proceed with the top-ranked agent
            top_agent_id = ranked_agent_ids[0]
            logger.info(f"Proceeding with top-ranked agent: {top_agent_id}")

            # 2. Find relevant tools via vector search (now filtered by agent context)
            # Placeholder for filtering tools by agent. For now, we search all tools.
            await self.session_manager.save_log(session_id, {"event": "tool_search", "query": query})
            potential_tools = self.find_relevant_tools(query)
            if not potential_tools:
                await self.session_manager.save_log(session_id, {"event": "tool_search_failed", "reason": "No potential tools found."})
                logger.warning("No potential tools found from vector search.")
                return {"error": "No relevant tools found."}
            await self.session_manager.save_log(session_id, {"event": "tool_search_completed", "potential_tools_count": len(potential_tools)})


            # 3. Filter tools with LLM to get the best ones
            await self.session_manager.save_log(session_id, {"event": "tool_filtering_started"})
            best_tool_ids = await self.filter_tools_with_llm(query, potential_tools, session_id=session_id)
            if not best_tool_ids:
                await self.session_manager.save_log(session_id, {"event": "tool_filtering_failed", "reason": "LLM could not select any suitable tools."})
                logger.warning("LLM could not select any suitable tools.")
                return []
            await self.session_manager.save_log(session_id, {"event": "tool_filtering_completed", "selected_tool_ids": best_tool_ids})
            
            self.selected_tools = [tool for tool in potential_tools if tool.get("tool_id") in best_tool_ids]
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