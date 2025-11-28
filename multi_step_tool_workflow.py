import os
import logging
import json
from typing import Dict, Any, List, TypedDict, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from embedding_service_ollama import OllamaEmbeddingService, create_ollama_embedding_service
from neo4j_enhanced_manager import EnhancedNeo4jToolManager
from agent_workflow import AgentWorkflow # Import AgentWorkflow

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
    with open("prompts/dynamic_agent_prompt.md", "r") as f:
        base_prompt = f.read()
    if tools:
        tool_info = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in tools])
        return base_prompt.format(tool_info=tool_info)
    return "You are a helpful assistant."

class MultiStepToolWorkflow:
    """
    Manages a multi-step workflow for finding relevant agents, finding tools for that agent,
    and executing a sequence of tools using an LLM agent.
    """
    def __init__(self, max_retries: int = 3):
        self.embedding_service: OllamaEmbeddingService = create_ollama_embedding_service()
        self.tool_manager: EnhancedNeo4jToolManager = EnhancedNeo4jToolManager()
        self.agent_workflow = AgentWorkflow(max_retries=max_retries) # Initialize AgentWorkflow
        self.llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, reasoning=True)
        self.tools = [demo_tool]
        self.max_retries = max_retries

        self.tool_manager.create_indexes()

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
        return self.tool_manager.search_tools_by_embedding(
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

    def execute_tool_with_agent(self, query: str, tools: List[Dict[str, Any]], max_steps: int = 5) -> Any:
        """
        Executes a sequence of tools using a ReAct-style agent within a manual loop.
        """
        if not tools:
            logger.warning("No tools available for execution.")
            return {"error": "No tools found to execute."}

        history = [{"role": "user", "content": query}]
        
        for i in range(max_steps):
            logger.info(f"--- Step {i + 1}/{max_steps} ---")
            logger.info(f"Current history: {history}")

            try:
                # Invoke the agent with the current history
                # Invoke the agent. The response will be a dictionary containing the new message chain.
                result_dict = self.agent.invoke(
                    {"messages": history},
                    context={"tools": tools}
                )
                
                # Extract the latest message from the agent's response
                agent_response = result_dict['messages'][-1]

                # Add the agent's response to our history
                history.append(agent_response)

                # Check if the agent's response is a final answer (AIMessage without tool calls)
                # or a tool call.
                if agent_response.tool_calls:
                    # Agent wants to call a tool. The framework has already executed it and the result
                    # is the last message in the sequence (`agent_response` is a ToolMessage).
                    # We just need to log it and continue the loop.
                    logger.info(f"Agent called tools, continuing loop.")
                elif isinstance(agent_response.content, str):
                    # This is a final answer from the agent (AIMessage with string content and no tool_calls).
                    logger.info(f"Agent provided final answer: {agent_response.content}")
                    return {"final_answer": agent_response.content, "history": history}
                else:
                    # This branch handles ToolMessage content, which is fine. We just log and continue.
                    logger.info(f"Continuing loop with new history item of type: {type(agent_response)}")


            except Exception as e:
                logger.error(f"Error executing tool with agent at step {i + 1}: {e}")
                return {"error": str(e), "history": history}

        logger.warning(f"Workflow did not complete within {max_steps} steps.")
        return {"error": "Max steps reached", "history": history}

    def run_workflow(self, query: str) -> Any:
        """
        Runs the complete workflow:
        1. Find relevant tools using vector search.
        2. Filter and select the best tools using an LLM.
        1. Find a relevant agent.
        2. Find relevant tools for that agent.
        3. Filter and select the best tools using an LLM.
        4. Execute the agent with the selected tools.
        """
        logger.info(f"Starting multi-step tool workflow for query: '{query}'")

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
        
        # Find the full details of the selected tools
        selected_tools = [tool for tool in potential_tools if tool.get("tool_id") in best_tool_ids]

        # 4. Execute the agent with the selected tools
        execution_result = self.execute_tool_with_agent(query, selected_tools)

        logger.info("Tool workflow completed.")
        return execution_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    multi_step_workflow = MultiStepToolWorkflow()

    user_query = "call the ingest data api"
    result = multi_step_workflow.run_workflow(user_query)

    print("\n--- Multi-Step Tool Execution Result ---")
    print(result)
    print("---------------------------")