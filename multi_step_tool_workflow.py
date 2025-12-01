import os
import logging
import json
from typing import Dict, Any, List, TypedDict, Optional

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
            tool_info_list.append(f"- {tool['name']} (ID: {tool['tool_id']}): {tool['description']}")
        tool_info = "\n".join(tool_info_list)
        
        # Print the formatted tool information that will be passed to the agent
        print("--- Tools Provided to Agent ---")
        print(tool_info)
        print("-----------------------------")
        
        return base_prompt.format(tool_info=tool_info)
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
    
    def filter_tools_with_llm(self, query: str, potential_tools: List[Dict[str, Any]]) -> List[str]:
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
        
        # Filter tool properties to reduce token count
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

        for attempt in range(self.max_retries):
            logger.info(f"Attempt {attempt + 1}/{self.max_retries} to filter tools with LLM.")
            try:
                response = chain.invoke({"query": query, "potential_tools": potential_tools_json})
                
                # The prompt expects a JSON object with a 'tool_ids' list
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

    def execute_tool_with_agent(self, session_id: str, tools: List[Dict[str, Any]], max_steps: int = 5) -> Any:
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
            result_dict = self.agent.invoke(
                {"messages": history},
                context={"tools": tools}
            )
            
            # The agent's final answer is expected in the last message
            print("response from agent ",result_dict)
            agent_response = result_dict['messages'][-1]

            # Save the final response to history
            self.session_manager.save_message(session_id, agent_response)
            history.append(agent_response)

            # Return the final answer
            logger.info(f"Agent provided final answer: {agent_response.content}")
            return {"final_answer": agent_response.content, "history": history}

        except Exception as e:
            logger.error(f"Error executing tool with agent: {e}")
            return {"error": str(e), "history": history}

    def run_workflow(self, query: str, session_id: Optional[str] = None) -> Any:
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
        
        # Add the new user message to the history and save it
        user_message = HumanMessage(content=query)
        history.append(user_message)
        self.session_manager.save_message(session_id, user_message)

        # 1. Find relevant agent(s)
        ranked_agent_ids = self.agent_workflow.run_workflow(query)
        if not ranked_agent_ids:
            logger.warning("No relevant agents found.")
            return {"error": "No relevant agents found for your query."}
        
        # For now, let's proceed with the top-ranked agent
        top_agent_id = ranked_agent_ids[0]
        logger.info(f"Proceeding with top-ranked agent: {top_agent_id}")

        # 2. Find relevant tools via vector search (now filtered by agent context)
        # Placeholder for filtering tools by agent. For now, we search all tools.
        potential_tools = self.find_relevant_tools(query)
        if not potential_tools:
            logger.warning("No potential tools found from vector search.")
            return {"error": "No relevant tools found."}

        # 3. Filter tools with LLM to get the best ones
        best_tool_ids = self.filter_tools_with_llm(query, potential_tools)
        if not best_tool_ids:
            logger.warning("LLM could not select any suitable tools.")
            return []
        
        selected_tools = [tool for tool in potential_tools if tool.get("tool_id") in best_tool_ids]
        print("selected tools",selected_tools)
        execution_result = self.execute_tool_with_agent(session_id, selected_tools)

        logger.info("Tool workflow completed.")
        return execution_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    workflow = MultiStepToolWorkflow()
    session_id = workflow.session_manager.create_session_id()

    print("--- Multi-Step Tool Workflow CLI ---")
    print(f"New session started: {session_id}")
    print("Type 'new' to start a new session, or 'exit' to quit.")

    while True:
        try:
            user_query = input(f"\nUser (Session: {session_id[:8]}): ")

            if user_query.lower() == 'exit':
                print("Exiting workflow.")
                break
            
            if user_query.lower() == 'new':
                session_id = workflow.session_manager.create_session_id()
                print(f"\n--- New session started: {session_id} ---")
                continue

            if not user_query:
                continue

            result = workflow.run_workflow(user_query, session_id=session_id)

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