import os
import logging
import json
from typing import Dict, Any, List, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from agent_workflow import AgentWorkflow
from discovery_agent_manager import DiscoveryAgentManager, create_discovery_agent_manager

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:30b")

@tool
def demo_tool(tool_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    A demo tool that simulates calling an API with a tool ID and request body.
    In a real scenario, this would make an HTTP request.
    """
    logger.info(f"Executing demo_tool with tool_id: {tool_id} and request_body: {request_body}")
    # Simulate an API call
    return {"status": "success", "tool_id": tool_id, "response": f"API call for {tool_id} was successful"}

class Context(TypedDict):
    tools: List[Dict[str, Any]]

@dynamic_prompt
def tool_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on available tools."""
    tools = request.runtime.context.get("tools", [])
    base_prompt = "You are a helpful assistant."
    if tools:
        tool_info = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
        return f"{base_prompt}\n\nAvailable tools:\n{tool_info}\n\nPlease use these tools to answer the user's query."
    return base_prompt

class ToolWorkflow:
    """
    Manages the workflow for finding relevant tools based on agent search results
    and executing a tool using an LLM agent.
    """
    def __init__(self):
        self.agent_workflow = AgentWorkflow()
        self.discovery_agent_manager: DiscoveryAgentManager = create_discovery_agent_manager()
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.tools = [demo_tool]
        self.agent = create_agent(
            self.llm,
            self.tools,
            middleware=[tool_prompt],
            context_schema=Context
        )

    def find_relevant_tools(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Finds tools associated with the given agent IDs.
        (This is a placeholder for the actual tool search logic)
        """
        logger.info(f"Searching for tools for agent IDs: {agent_ids}")
        # In a real implementation, this would query a database (e.g., Neo4j)
        # to find tools linked to these agent IDs.
        # For this example, we'll assume a static mapping or a simple search.
        
        # This is where you would integrate your tool search logic.
        # For now, we'll return a mock tool list if any agent IDs are present.
        if not agent_ids:
            return []
            
        # Mocking tool search result
        mock_tools = [
            {"id": "tool-123", "name": "demo_tool", "description": "A demo tool for API calls.", "agent_id": agent_ids[0]},
        ]
        logger.info(f"Found mock tools: {mock_tools}")
        return mock_tools

    def execute_tool_with_agent(self, query: str, tools: List[Dict[str, Any]]) -> Any:
        """
        Executes a tool using the LLM agent based on the user query and available tools.
        """
        if not tools:
            logger.warning("No tools available for execution.")
            return {"error": "No tools found to execute."}

        # The `create_agent` function in the constructor already has the tools.
        # We invoke the agent with the user's query.
        logger.info(f"Invoking agent for query: '{query}'")
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                context={"tools": tools}
            )
            return result
        except Exception as e:
            logger.error(f"Error executing tool with agent: {e}")
            return {"error": str(e)}

    def run_workflow(self, query: str) -> Any:
        """
        Runs the complete workflow:
        1. Find relevant agents.
        2. Find tools for those agents.
        3. Execute the best tool for the query.
        """
        logger.info(f"Starting tool workflow for query: '{query}'")

        # 1. Find relevant agents
        ranked_agent_ids = self.agent_workflow.run_workflow(query)
        if not ranked_agent_ids:
            logger.warning("No agents found for the query.")
            return {"error": "No relevant agents found."}
        
        logger.info(f"Found agent IDs: {ranked_agent_ids}")

        # 2. Find tools for those agents
        relevant_tools = self.find_relevant_tools(ranked_agent_ids)
        if not relevant_tools:
            logger.warning(f"No tools found for agent IDs: {ranked_agent_ids}")
            return {"error": "No tools found for the identified agents."}

        # 3. Execute the best tool for the query
        execution_result = self.execute_tool_with_agent(query, relevant_tools)

        logger.info("Tool workflow completed.")
        return execution_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tool_workflow = ToolWorkflow()

    user_query = "call the create schema api"
    result = tool_workflow.run_workflow(user_query)

    print("\n--- Tool Execution Result ---")
    print(result)
    print("---------------------------")