#!/usr/bin/env python3
"""
Orchestrator using ChromaDB + MongoDB (instead of Neo4j)
Combines discovery from multi_step_tool_workflow with execution from main_orchestrator
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List

from agent_workflow import AgentWorkflow
from tool_workflow import ToolWorkflow
from llm_prompt_parser import create_prompt_parser
from api_executor import create_api_executor
from agent_structure import ALL_AGENTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaMongoOrchestrator:
    """
    Orchestrator that uses ChromaDB + MongoDB for discovery and executes real APIs.
    
    Flow:
    1. Discover agent using ChromaDB (agents collection)
    2. Discover tool using ChromaDB (tools collection) + MongoDB (full metadata)
    3. Parse prompt for parameter modifications
    4. Handle dependencies (e.g., generate token first)
    5. Execute API with modified parameters
    6. Return response
    """
    
    def __init__(self, max_retries: int = 3):
        self.agent_wf = AgentWorkflow(max_retries=max_retries)
        self.tool_wf = ToolWorkflow(max_retries=max_retries)
        self.prompt_parser = create_prompt_parser()
        self.api_executor = create_api_executor()
        
        # Map agent_id to agent info for getting database_name
        self.agent_map = {agent["agent_id"]: agent for agent in ALL_AGENTS}
        
        # Cache for universe_id (reuse after first creation)
        self.cached_universe_id = None
        
        logger.info("✅ ChromaMongoOrchestrator initialized successfully")
    
    def process_user_request(self, user_prompt: str) -> Dict[str, Any]:
        """
        Process a user's natural language request and execute the API
        
        Args:
            user_prompt: User's natural language request
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"📝 Processing user request: '{user_prompt}'")
        
        try:
            # Step 1: Find relevant agent using ChromaDB
            logger.info("🤖 Step 1: Searching for relevant agent in ChromaDB...")
            ranked_agent_ids = self.agent_wf.run_workflow(
                user_prompt, 
                top_k=5, 
                rerank_with_llm=True
            )
            
            if not ranked_agent_ids:
                return {
                    "success": False,
                    "error": "No matching agent found for your request",
                    "suggestion": "Try rephrasing your request or check available agents"
                }
            
            selected_agent_id = ranked_agent_ids[0]
            agent_info = self.agent_map.get(selected_agent_id)
            
            if not agent_info:
                return {
                    "success": False,
                    "error": f"Agent '{selected_agent_id}' not found in agent structure"
                }
            
            logger.info(f"✓ Found agent: {agent_info['name']} (ID: {selected_agent_id})")
            
            # Step 2: Find relevant tool using ChromaDB + MongoDB
            logger.info(f"🔍 Step 2: Searching for relevant tool in ChromaDB + MongoDB...")
            tools_result = self.tool_wf.run(
                user_prompt,
                top_k=8,
                rerank_with_llm=True
            )
            
            selected_tools = tools_result.get("selected_tools", [])
            fallback_used = tools_result.get("fallback_used", False)
            
            # Handle case where LLM selected invalid tool IDs
            # Map common variations to actual tool IDs
            tool_id_mapping = {
                "ingest_data_into_schema": "ingest_data_api",
                "ingest_data": "ingest_data_api",
                "insert_data_api": "ingest_data_api",
                "create_schema_api": "create_mongo_schema_api",
                "schema_creation_api": "create_mongo_schema_api",
            }
            
            # If no tools found, try to find tools by intent
            if not selected_tools:
                logger.warning("⚠ No tools found from LLM selection, trying intent-based fallback...")
                # Detect user intent and try appropriate tool
                user_prompt_lower = user_prompt.lower()
                if "insert data" in user_prompt_lower or "ingest data" in user_prompt_lower:
                    # User wants to ingest data
                    tool_doc = self.tool_wf.load_tool_from_mongo("ingest_data_api")
                    if tool_doc:
                        selected_tools.append(tool_doc)
                        logger.info("✓ Loaded ingest_data_api using intent-based fallback")
                elif "tidb" in user_prompt_lower and ("create schema" in user_prompt_lower or "create table" in user_prompt_lower):
                    # User wants to create TiDB schema
                    tool_doc = self.tool_wf.load_tool_from_mongo("create_tidb_schema_api")
                    if tool_doc:
                        selected_tools.append(tool_doc)
                        logger.info("✓ Loaded create_tidb_schema_api using intent-based fallback")
                elif "create schema" in user_prompt_lower or "create table" in user_prompt_lower:
                    # User wants to create schema (MongoDB by default)
                    tool_doc = self.tool_wf.load_tool_from_mongo("create_mongo_schema_api")
                    if tool_doc:
                        selected_tools.append(tool_doc)
                        logger.info("✓ Loaded create_mongo_schema_api using intent-based fallback")
                elif "create dataverse" in user_prompt_lower or "create universe" in user_prompt_lower:
                    # User wants to create dataverse
                    tool_doc = self.tool_wf.load_tool_from_mongo("create_dataverse_api")
                    if tool_doc:
                        selected_tools.append(tool_doc)
                        logger.info("✓ Loaded create_dataverse_api using intent-based fallback")
            
            if not selected_tools:
                return {
                    "success": False,
                    "error": f"No matching tool found for your request",
                    "suggestion": "Try rephrasing your request or check available tools",
                    "agent": agent_info.get('name', 'Unknown')
                }
            
            # Find the PRIMARY tool (not a dependency tool like token_generation_api)
            # Dependency tools are: token_generation_api, runrun_token_api
            # Action tools (what user actually wants): create_mongo_schema_api, create_dataverse_api, ingest_data_api, etc.
            dependency_tool_ids = {"token_generation_api", "runrun_token_api"}
            
            # Priority order: action tools first, then others
            # For "create schema" queries, prefer create_mongo_schema_api
            # For "create dataverse" queries, prefer create_dataverse_api
            # For "ingest data" queries, prefer ingest_data_api
            
            primary_tool = None
            
            # First, try to find the tool that matches the user's intent
            user_prompt_lower = user_prompt.lower()
            if "tidb" in user_prompt_lower and ("schema" in user_prompt_lower or "table" in user_prompt_lower):
                # User wants to create TiDB schema
                for t in selected_tools:
                    if t.get("tool_id") == "create_tidb_schema_api":
                        primary_tool = t
                        break
                # If not found in selected tools, try to load it directly
                if not primary_tool:
                    tidb_tool = self.tool_wf.load_tool_from_mongo("create_tidb_schema_api")
                    if tidb_tool:
                        selected_tools.append(tidb_tool)
                        primary_tool = tidb_tool
                        logger.info("✓ Loaded create_tidb_schema_api using intent-based fallback")
            elif "schema" in user_prompt_lower or "table" in user_prompt_lower:
                # User wants to create schema (MongoDB by default, or if mongo mentioned)
                if "mongo" in user_prompt_lower:
                    for t in selected_tools:
                        if t.get("tool_id") == "create_mongo_schema_api":
                            primary_tool = t
                            break
                else:
                    # Default to MongoDB schema if not specified
                    for t in selected_tools:
                        if t.get("tool_id") == "create_mongo_schema_api":
                            primary_tool = t
                            break
            elif "dataverse" in user_prompt_lower or "universe" in user_prompt_lower:
                # User wants to create dataverse
                for t in selected_tools:
                    if t.get("tool_id") == "create_dataverse_api":
                        primary_tool = t
                        break
            elif "ingest" in user_prompt_lower or "insert data" in user_prompt_lower:
                # User wants to ingest data
                for t in selected_tools:
                    if t.get("tool_id") == "ingest_data_api":
                        primary_tool = t
                        break
            
            # If no intent-specific tool found, look for any non-dependency tool
            if not primary_tool:
                for t in selected_tools:
                    tool_id = t.get("tool_id", "")
                    # Skip dependency tools - they'll be handled automatically
                    if tool_id not in dependency_tool_ids:
                        primary_tool = t
                        break
            
            # If still no primary tool found, use the last one (usually the action tool)
            # since LLM often returns dependencies first, then the action tool
            if not primary_tool:
                primary_tool = selected_tools[-1]  # Use last tool (usually the action)
                logger.warning(f"⚠ No primary tool found, using last tool: {primary_tool.get('tool_id')}")
            
            tool = primary_tool
            logger.info(f"✓ Found primary tool: {tool.get('name')} (ID: {tool.get('tool_id')})")
            logger.info(f"  Selected from {len(selected_tools)} candidate tool(s): {[t.get('tool_id') for t in selected_tools]}")
            
            # Step 3: Extract IDs from prompt FIRST (before parsing and dependencies)
            # This prevents unnecessary dependency creation if user already provided IDs
            logger.info("🔍 Step 3: Extracting IDs from prompt...")
            user_provided_schema_id = self._extract_schema_id_from_prompt(user_prompt)
            user_provided_universe_id = self._extract_universe_id_from_prompt(user_prompt)
            
            if user_provided_schema_id:
                logger.info(f"✓ User provided schema_id: {user_provided_schema_id}")
            if user_provided_universe_id:
                logger.info(f"✓ User provided universe_id: {user_provided_universe_id}")
            
            # Step 4: Parse prompt for parameter modifications
            logger.info("🧠 Step 4: Parsing prompt for parameter modifications...")
            
            # Get schema info from tool
            tool_schema = tool.get("schema", {})
            field_descriptions = tool_schema.get("field_descriptions", {})
            current_values = tool_schema.get("requestBody", {})
            
            # For data ingestion, try to extract data array directly from prompt first
            # This ensures we preserve ALL fields without LLM dropping any
            parameter_modifications = {}
            if tool.get("tool_id") == "ingest_data_api":
                extracted_data = self._extract_data_array_from_prompt(user_prompt)
                if extracted_data:
                    logger.info(f"✓ Directly extracted data array from prompt: {len(extracted_data)} record(s)")
                    if len(extracted_data) > 0 and isinstance(extracted_data[0], dict):
                        logger.info(f"  Data fields: {list(extracted_data[0].keys())}")
                    parameter_modifications["data"] = extracted_data
                    # Still use LLM for other fields (like schema_id if mentioned)
                    tool_info = {
                        "field_descriptions": field_descriptions,
                        "requestBody": current_values
                    }
                    llm_modifications = self.prompt_parser.parse_prompt_for_parameters(user_prompt, tool_info)
                    # Merge LLM modifications but keep our directly extracted data
                    if llm_modifications:
                        parameter_modifications.update({k: v for k, v in llm_modifications.items() if k != "data"})
                else:
                    logger.warning("⚠ Direct data extraction failed, falling back to LLM extraction")
                    # Fall back to LLM extraction
                    tool_info = {
                        "field_descriptions": field_descriptions,
                        "requestBody": current_values
                    }
                    if self.prompt_parser.should_use_defaults(user_prompt):
                        logger.info("✓ Using default parameters (no modifications)")
                        parameter_modifications = {}
                    else:
                        parameter_modifications = self.prompt_parser.parse_prompt_for_parameters(
                            user_prompt,
                            tool_info
                        )
                        if parameter_modifications:
                            logger.info(f"✓ Extracted modifications: {parameter_modifications}")
                            # If LLM added extra fields that weren't in original prompt, remove them
                            if "data" in parameter_modifications and isinstance(parameter_modifications["data"], list):
                                if len(parameter_modifications["data"]) > 0:
                                    first_record = parameter_modifications["data"][0]
                                    if isinstance(first_record, dict):
                                        # Try to extract original fields from prompt
                                        # Look for JSON pattern in the prompt to find original field names
                                        import re
                                        
                                        # Try to find JSON object in prompt (handle both quoted and unquoted)
                                        # Pattern 1: {"email":"...","age":30}  (simple format)
                                        # Pattern 2: "{\"email\":\"...\",\"age\":30}"  (quoted with escapes)
                                        original_fields = set()
                                        
                                        # First, try to find JSON object by looking for {" or '{ or just {
                                        json_start = user_prompt.find('{"')
                                        if json_start == -1:
                                            json_start = user_prompt.find("{'")
                                        if json_start == -1:
                                            # Try finding just { (for simple format)
                                            json_start = user_prompt.find('{')
                                        
                                        if json_start != -1:
                                            # Extract JSON object by matching braces
                                            bracket_count = 0
                                            json_end = json_start
                                            in_string = False
                                            escape_next = False
                                            
                                            for i in range(json_start, len(user_prompt)):
                                                char = user_prompt[i]
                                                if escape_next:
                                                    escape_next = False
                                                    continue
                                                if char == '\\':
                                                    escape_next = True
                                                    continue
                                                if char == '"' and not escape_next:
                                                    in_string = not in_string
                                                    continue
                                                if not in_string:
                                                    if char == '{':
                                                        bracket_count += 1
                                                    elif char == '}':
                                                        bracket_count -= 1
                                                        if bracket_count == 0:
                                                            json_end = i + 1
                                                            break
                                            
                                            if json_end > json_start:
                                                json_str = user_prompt[json_start:json_end]
                                                # Unescape if needed
                                                json_str = json_str.replace('\\"', '"').replace("\\'", "'")
                                                # Extract field names: "fieldname": or 'fieldname':
                                                field_matches = re.findall(r'["\']([^"\']+)["\']\s*:', json_str)
                                                if field_matches:
                                                    original_fields = set(field_matches)
                                                    logger.debug(f"  Found original fields in prompt: {original_fields}")
                                        
                                        # Only remove "id" and "name" if they weren't in original
                                        # IMPORTANT: Keep ALL other fields (like "email", "age", "username")
                                        for record in parameter_modifications["data"]:
                                            if isinstance(record, dict):
                                                # Check each field and remove ONLY "id" and "name" if not in original
                                                # DO NOT remove any other fields
                                                for field_name in list(record.keys()):
                                                    if field_name in ["id", "name"] and field_name not in original_fields:
                                                        del record[field_name]
                                                        logger.warning(f"  ⚠ Removed '{field_name}' field that LLM added (not in original data)")
                                                    elif field_name not in ["id", "name"]:
                                                        # Keep all other fields - they should be in original
                                                        logger.debug(f"  Keeping field '{field_name}' (not in removal list)")
                        else:
                            logger.info("✓ No modifications needed")
            else:
                # For other tools, use normal LLM extraction
                tool_info = {
                    "field_descriptions": field_descriptions,
                    "requestBody": current_values
                }
                if self.prompt_parser.should_use_defaults(user_prompt):
                    logger.info("✓ Using default parameters (no modifications)")
                    parameter_modifications = {}
                else:
                    parameter_modifications = self.prompt_parser.parse_prompt_for_parameters(
                        user_prompt,
                        tool_info
                    )
                    if parameter_modifications:
                        logger.info(f"✓ Extracted modifications: {parameter_modifications}")
                    else:
                        logger.info("✓ No modifications needed")
            
            # Add user-provided IDs to modifications if found
            # IMPORTANT: Remove any schema_id that LLM hallucinated if user didn't provide it
            if user_provided_schema_id:
                parameter_modifications["schema_id"] = user_provided_schema_id
            else:
                # Remove schema_id if LLM hallucinated it
                if "schema_id" in parameter_modifications:
                    logger.warning(f"⚠ Removing hallucinated schema_id: {parameter_modifications.pop('schema_id')}")
            
            if user_provided_universe_id:
                parameter_modifications["universes"] = [user_provided_universe_id]
            
            # Step 5: Handle dependencies
            logger.info("🔗 Step 5: Checking dependencies...")
            # Extract data for schema inference if needed
            # Use the ORIGINAL extracted data (before LLM modifications) to preserve exact structure
            extracted_data = None
            if tool.get("tool_id") == "ingest_data_api" and not user_provided_schema_id:
                # Get the original data that was directly extracted from prompt
                extracted_data = self._extract_data_array_from_prompt(user_prompt)
                if extracted_data:
                    logger.info(f"✓ Data extracted for schema inference: {len(extracted_data)} record(s)")
                    if len(extracted_data) > 0 and isinstance(extracted_data[0], dict):
                        logger.info(f"  Fields: {list(extracted_data[0].keys())}")
                else:
                    # Fallback to parameter_modifications if direct extraction failed
                    extracted_data = parameter_modifications.get("data")
                    if extracted_data:
                        logger.info(f"✓ Using data from parameter_modifications: {len(extracted_data)} record(s)")
                    else:
                        logger.warning("⚠ No data found for schema inference - will use default schema")
            
            dependency_results = self._handle_dependencies(
                tool, 
                agent_info,
                skip_schema_creation=user_provided_schema_id is not None,
                skip_dataverse_creation=user_provided_universe_id is not None,
                data_for_schema_inference=extracted_data,
                parameter_modifications=parameter_modifications
            )
            auth_token = dependency_results.get("token")
            dependency_values = dependency_results.get("values", {})
            
            # Merge dependency values into parameter modifications
            if dependency_values:
                parameter_modifications = {**parameter_modifications, **dependency_values}
                logger.info(f"✓ Added dependency values: {dependency_values}")
            
            # Step 6: Execute API
            logger.info("🚀 Step 6: Executing API call...")
            
            # Prepare tool_info for API executor
            api_tool_info = {
                "url": tool_schema.get("url", ""),
                "method": tool_schema.get("method", "POST"),
                "headers": tool_schema.get("headers", {}),
                "requestBody": current_values,
                "queryParameters": tool_schema.get("queryParameters", {}),
                "authentication_required": tool_schema.get("authentication_required", False),
                "returns_token": tool_schema.get("returns_token", False),
                "token_field": tool_schema.get("token_field", "token")
            }
            
            result = self.api_executor.execute_api(
                tool_info=api_tool_info,
                parameter_modifications=parameter_modifications,
                auth_token=auth_token
            )
            
            # Step 6: Cache token if generated
            if result.get("token"):
                self.api_executor.cache_token(tool.get("tool_id"), result["token"])
            
            # Format final response
            return self._format_response(tool, result, parameter_modifications, agent_info)
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_schema_id_from_prompt(self, user_prompt: str) -> Optional[str]:
        """Extract schema ID from user prompt using regex"""
        import re
        # Look for patterns like "schemaid is 692ceae7fd9c66658f22d724" or "schema id 692ceae7fd9c66658f22d724"
        patterns = [
            r'schemaid\s+is\s+([a-f0-9]{24})',
            r'schema\s+id\s+is\s+([a-f0-9]{24})',
            r'schemaid[:\s]+([a-f0-9]{24})',
            r'schema[:\s]+id[:\s]+([a-f0-9]{24})',
            r'with\s+schemaid\s+as\s+([a-f0-9]{24})',
            r'into\s+schema\s+([a-f0-9]{24})',
            r'schema\s+([a-f0-9]{24})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                schema_id = match.group(1)
                logger.info(f"  ✓ Extracted schema_id from prompt: {schema_id}")
                return schema_id
        
        return None
    
    def _extract_universe_id_from_prompt(self, user_prompt: str) -> Optional[str]:
        """Extract universe ID from user prompt using regex"""
        import re
        # Look for patterns like "universeid as 6911d6f568829872bc56d027"
        patterns = [
            r'universeid\s+as\s+([a-f0-9]{24})',
            r'universe\s+id\s+as\s+([a-f0-9]{24})',
            r'universeid[:\s]+([a-f0-9]{24})',
            r'universe[:\s]+id[:\s]+([a-f0-9]{24})',
            r'with\s+universeid\s+as\s+([a-f0-9]{24})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                universe_id = match.group(1)
                logger.info(f"  ✓ Extracted universe_id from prompt: {universe_id}")
                return universe_id
        
        return None
    
    def _extract_data_array_from_prompt(self, user_prompt: str) -> Optional[list]:
        """
        Extract data array directly from prompt using regex/JSON parsing
        This ensures we preserve ALL fields without LLM dropping any
        """
        import re
        import ast
        
        # Try to find JSON object or array in the prompt
        # Look for patterns like: {"email":"...","age":26,"username":"..."}
        # or: [{"email":"...","age":26}]
        
        # First, try to find JSON object/array brackets
        bracket_start = user_prompt.find('{')
        array_start = user_prompt.find('[')
        
        # Prefer array if both found
        start_char = '[' if array_start != -1 and (bracket_start == -1 or array_start < bracket_start) else '{'
        start_idx = array_start if start_char == '[' else bracket_start
        
        if start_idx != -1:
            # Find the matching closing bracket
            bracket_count = 0
            bracket_end = start_idx
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(user_prompt)):
                char = user_prompt[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char in ['{', '[']:
                        bracket_count += 1
                    elif char in ['}', ']']:
                        bracket_count -= 1
                        if bracket_count == 0:
                            bracket_end = i + 1
                            break
            
            if bracket_end > start_idx:
                json_str = user_prompt[start_idx:bracket_end]
                logger.debug(f"  Attempting to parse JSON: {json_str[:100]}...")
                
                # Try multiple parsing strategies
                # Strategy 1: Direct JSON parsing
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        logger.info(f"  ✓ Extracted data array directly from prompt (JSON): {len(data)} record(s)")
                        if len(data) > 0 and isinstance(data[0], dict):
                            fields = list(data[0].keys())
                            logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                        return data
                    elif isinstance(data, dict):
                        # Single object, wrap in array
                        logger.info(f"  ✓ Extracted data object, wrapping in array")
                        return [data]
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: Try Python literal eval
                try:
                    data = ast.literal_eval(json_str)
                    if isinstance(data, list):
                        logger.info(f"  ✓ Extracted data array (Python syntax): {len(data)} record(s)")
                        return data
                    elif isinstance(data, dict):
                        return [data]
                except (ValueError, SyntaxError):
                    pass
                
                # Strategy 3: Try fixing quotes
                try:
                    fixed_json = json_str.replace("'", '"')
                    data = json.loads(fixed_json)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                except json.JSONDecodeError:
                    pass
        
        # Try to extract from "insert data" or "ingest data" patterns
        # Handle both formats:
        # 1. insert data {"email":"...","age":30}  (simple format - preferred)
        # 2. insert data "{\"email\":\"...\",\"age\":30}"  (quoted with escaped quotes)
        ingest_keywords = ['insert data', 'ingest data']
        
        for keyword in ingest_keywords:
            keyword_lower = keyword.lower()
            prompt_lower = user_prompt.lower()
            keyword_pos = prompt_lower.find(keyword_lower)
            if keyword_pos != -1:
                # Find position after keyword
                search_start = keyword_pos + len(keyword)
                logger.debug(f"  Found '{keyword}' at position {keyword_pos}, searching from {search_start}")
                logger.debug(f"  Substring: '{user_prompt[search_start:search_start+50]}'")
                
                # Look for { or [ starting from this position (skip whitespace and quotes)
                json_start = -1
                for i in range(search_start, min(search_start + 100, len(user_prompt))):  # Limit search to first 100 chars
                    char = user_prompt[i]
                    if char in ['{', '[']:
                        json_start = i
                        logger.debug(f"  Found JSON start bracket '{char}' at position {i}")
                        break
                    elif char not in [' ', '"', "'"]:  # Skip whitespace and quotes only
                        # If we hit something else before finding { or [, this isn't the right pattern
                        break
                
                if json_start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    bracket_end = json_start
                    in_string = False
                    escape_next = False
                    
                    for i in range(json_start, len(user_prompt)):
                        char = user_prompt[i]
                        
                        if escape_next:
                            escape_next = False
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            continue
                        
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        
                        if not in_string:
                            if char in ['{', '[']:
                                bracket_count += 1
                            elif char in ['}', ']']:
                                bracket_count -= 1
                                if bracket_count == 0:
                                    bracket_end = i + 1
                                    break
                    
                    if bracket_end > json_start:
                        json_str = user_prompt[json_start:bracket_end]
                        logger.debug(f"  Extracted JSON string: {json_str}")
                        
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, list):
                                logger.info(f"  ✓ Extracted data array from pattern: {len(data)} record(s)")
                                if len(data) > 0 and isinstance(data[0], dict):
                                    fields = list(data[0].keys())
                                    logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                                return data
                            elif isinstance(data, dict):
                                logger.info(f"  ✓ Extracted data object from pattern, wrapping in array")
                                if len(data) > 0:
                                    fields = list(data.keys())
                                    logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                                return [data]
                        except json.JSONDecodeError as e:
                            logger.debug(f"  JSON parse failed: {e}, trying literal_eval...")
                            try:
                                data = ast.literal_eval(json_str)
                                if isinstance(data, list):
                                    logger.info(f"  ✓ Extracted data array (literal_eval): {len(data)} record(s)")
                                    return data
                                elif isinstance(data, dict):
                                    logger.info(f"  ✓ Extracted data object (literal_eval), wrapping in array")
                                    return [data]
                            except (ValueError, SyntaxError) as e2:
                                logger.debug(f"  literal_eval also failed: {e2}")
                                pass
        
        # Try patterns without quotes (direct JSON) - simpler format
        # Pattern: insert data {"email":"...","age":30}
        direct_patterns = [
            r'insert\s+data\s+(\{.+\})',
            r'ingest\s+data\s+(\{.+\})',
        ]
        
        for pattern in direct_patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                data_part = match.group(1).strip()
                # Find the complete JSON object by matching braces
                bracket_count = 0
                json_end = len(data_part)
                in_string = False
                escape_next = False
                
                for i, char in enumerate(data_part):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                
                json_str = data_part[:json_end]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        logger.info(f"  ✓ Extracted data array (direct pattern): {len(data)} record(s)")
                        return data
                    elif isinstance(data, dict):
                        logger.info(f"  ✓ Extracted data object (direct pattern), wrapping in array")
                        return [data]
                except json.JSONDecodeError:
                    pass
        
        logger.warning("  ⚠ Could not extract data from prompt using any method")
        return None
    
    def _handle_dependencies(
        self,
        tool: Dict[str, Any],
        agent_info: Dict[str, Any],
        skip_schema_creation: bool = False,
        skip_dataverse_creation: bool = False,
        data_for_schema_inference: Optional[List[Dict[str, Any]]] = None,
        parameter_modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle tool dependencies (e.g., generate token, create dataverse, etc.)
        
        Args:
            tool: Tool that may have dependencies
            agent_info: Agent information
            
        Returns:
            Dictionary with 'token' and 'values' (extracted from dependency responses)
        """
        result = {"token": None, "values": {}}
        
        tool_schema = tool.get("schema", {})
        tool_id = tool.get("tool_id", "")
        
        # Get dependencies from tool (may be stored as JSON string in MongoDB)
        dependencies = tool.get("dependencies", [])
        # Handle case where dependencies might be a JSON string
        if isinstance(dependencies, str):
            try:
                import json
                dependencies = json.loads(dependencies)
            except:
                dependencies = []
        
        # For schema creation, we always need token and dataverse
        # Even if dependencies aren't explicitly listed, infer them
        if tool_id == "create_mongo_schema_api" or tool_id == "create_tidb_schema_api":
            if "token_generation_api" not in dependencies:
                dependencies.append("token_generation_api")
            if "create_dataverse_api" not in dependencies and not skip_dataverse_creation:
                dependencies.append("create_dataverse_api")
            logger.info(f"  Schema creation tool ({tool_id}) - ensuring dependencies: {dependencies}")
        
        # For data ingestion, we need token (but not schema creation if schema_id provided)
        if tool_id == "ingest_data_api":
            if "token_generation_api" not in dependencies:
                dependencies.append("token_generation_api")
            # Only add schema creation if schema_id not provided
            if "create_mongo_schema_api" not in dependencies and not skip_schema_creation:
                dependencies.append("create_mongo_schema_api")
            logger.info(f"  Data ingestion tool - ensuring dependencies: {dependencies}")
        
        # Check if tool requires authentication
        requires_auth = tool_schema.get("authentication_required", False)
        
        if not requires_auth and not dependencies:
            logger.info("  ✓ No authentication required and no dependencies")
            return result
        
        # If we have dependencies or need auth, process them
        if not dependencies and requires_auth:
            logger.warning("  ⚠ Tool requires auth but has no dependencies defined")
            return result
        
        if not dependencies:
            logger.warning("  ⚠ Tool requires auth but has no dependencies defined")
            return result
        
        logger.info(f"  Dependencies found: {dependencies}")
        
        # Find token generation dependency
        current_token = None
        token_dep_id = None
        
        for dep_id in dependencies:
            # Find dependency tool in MongoDB
            dep_tool = self.tool_wf.load_tool_from_mongo(dep_id)
            if dep_tool and dep_tool.get("schema", {}).get("returns_token", False):
                token_dep_id = dep_id
                cached_token = self.api_executor.get_cached_token(dep_id)
                if cached_token:
                    logger.info(f"  ✓ Using cached token from {dep_id}")
                    current_token = cached_token
                else:
                    logger.info(f"  🔑 Generating token using {dep_id}...")
                    dep_schema = dep_tool.get("schema", {})
                    dep_tool_info = {
                        "url": dep_schema.get("url", ""),
                        "method": dep_schema.get("method", "POST"),
                        "headers": dep_schema.get("headers", {}),
                        "requestBody": dep_schema.get("requestBody", {}),
                        "queryParameters": dep_schema.get("queryParameters", {}),
                        "authentication_required": dep_schema.get("authentication_required", False),
                        "returns_token": dep_schema.get("returns_token", False),
                        "token_field": dep_schema.get("token_field", "token")
                    }
                    token_result = self.api_executor.execute_api(
                        tool_info=dep_tool_info,
                        parameter_modifications={},
                        auth_token=None
                    )
                    if token_result and token_result.get("success"):
                        current_token = token_result.get("token")
                        if current_token:
                            self.api_executor.cache_token(dep_id, current_token)
                            logger.info("  ✓ Token generated and cached")
                break
        
        if not current_token:
            logger.warning("  ⚠ No token available - dependencies requiring auth may fail")
        
        # Handle dataverse creation FIRST (before schema creation) if needed
        # This ensures we have universe_id for schema creation
        needs_dataverse = False
        for dep_id in dependencies:
            if dep_id == "create_dataverse_api":
                if skip_dataverse_creation:
                    continue
                elif self.cached_universe_id:
                    logger.info(f"  ✓ Using cached universe_id: {self.cached_universe_id}")
                    result["values"]["universes"] = [self.cached_universe_id]
                else:
                    needs_dataverse = True
                    break
        
        # Create dataverse first if needed
        if needs_dataverse and current_token:
            logger.info(f"  🔧 Creating dataverse (required for schema creation)...")
            dataverse_tool = self.tool_wf.load_tool_from_mongo("create_dataverse_api")
            if dataverse_tool:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataverse_modifications = {"name": f"Dataverse_{timestamp}"}
                
                dataverse_schema = dataverse_tool.get("schema", {})
                dataverse_tool_info = {
                    "url": dataverse_schema.get("url", ""),
                    "method": dataverse_schema.get("method", "POST"),
                    "headers": dataverse_schema.get("headers", {}),
                    "requestBody": dataverse_schema.get("requestBody", {}),
                    "queryParameters": dataverse_schema.get("queryParameters", {}),
                    "authentication_required": dataverse_schema.get("authentication_required", False),
                    "returns_token": dataverse_schema.get("returns_token", False),
                    "token_field": dataverse_schema.get("token_field", "token")
                }
                
                dataverse_result = self.api_executor.execute_api(
                    tool_info=dataverse_tool_info,
                    parameter_modifications=dataverse_modifications,
                    auth_token=current_token
                )
                
                if dataverse_result and dataverse_result.get("success"):
                    extracted_values = self._extract_dependency_values(
                        "create_dataverse_api",
                        dataverse_result.get("response", {})
                    )
                    if extracted_values:
                        result["values"].update(extracted_values)
                        universe_id = extracted_values.get("universes", [None])[0] if extracted_values.get("universes") else None
                        if universe_id:
                            self.cached_universe_id = universe_id
                            logger.info(f"  ✓ Dataverse created and universe_id cached: {universe_id}")
        
        # Handle other dependencies (schema creation, etc.)
        for dep_id in dependencies:
            if dep_id == token_dep_id:
                continue  # Skip token generation, already handled
            
            # Skip schema creation if user provided schema_id
            if dep_id in ["create_mongo_schema_api", "create_tidb_schema_api"] and skip_schema_creation:
                logger.info(f"  ⏭ Skipping {dep_id} (schema_id provided by user)")
                continue
            
            # Skip dataverse creation (already handled above)
            if dep_id == "create_dataverse_api":
                continue
            
            dep_tool = self.tool_wf.load_tool_from_mongo(dep_id)
            if not dep_tool:
                logger.error(f"  ✗ Dependency tool not found: {dep_id}")
                continue
            
            dep_schema = dep_tool.get("schema", {})
            if dep_schema.get("authentication_required", False):
                if not current_token:
                    logger.warning(f"  ⚠ Dependency {dep_id} needs token but none available")
                    continue
                
                logger.info(f"  🔧 Executing dependency: {dep_id}...")
                
                dep_tool_info = {
                    "url": dep_schema.get("url", ""),
                    "method": dep_schema.get("method", "POST"),
                    "headers": dep_schema.get("headers", {}),
                    "requestBody": dep_schema.get("requestBody", {}),
                    "queryParameters": dep_schema.get("queryParameters", {}),
                    "authentication_required": dep_schema.get("authentication_required", False),
                    "returns_token": dep_schema.get("returns_token", False),
                    "token_field": dep_schema.get("token_field", "token")
                }
                
                # Prepare dependency modifications
                dep_modifications = {}
                if dep_id == "create_dataverse_api":
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dep_modifications = {"name": f"Dataverse_{timestamp}"}
                elif dep_id == "create_mongo_schema_api" or dep_id == "create_tidb_schema_api":
                    # Determine database type
                    is_tidb = dep_id == "create_tidb_schema_api"
                    db_type = "TIDB" if is_tidb else "MONGO"
                    
                    # Get schema name and attributes from parameter_modifications if provided
                    # Check if we're creating schema as part of the main tool execution
                    # (not as a dependency for data ingestion)
                    schema_name = None
                    attributes = []
                    primary_key = []
                    
                    # Check if attributes are in parameter_modifications
                    # (this happens when user explicitly requests schema creation)
                    if parameter_modifications:
                        if "name" in parameter_modifications:
                            schema_name = parameter_modifications["name"]
                        if "attributes" in parameter_modifications:
                            # Convert LLM format to API format
                            llm_attributes = parameter_modifications["attributes"]
                            if isinstance(llm_attributes, list):
                                for attr in llm_attributes:
                                    if isinstance(attr, dict):
                                        # Convert LLM format: {"name": "username", "type": "string", "required": True}
                                        # to API format: {"name": "username", "type": {"type": "string"}, "required": True}
                                        attr_type = attr.get("type")
                                        if isinstance(attr_type, str):
                                            api_type = {"type": attr_type}
                                        elif isinstance(attr_type, dict):
                                            api_type = attr_type
                                        else:
                                            api_type = {"type": "string"}
                                        
                                        api_attr = {
                                            "name": attr.get("name"),
                                            "type": api_type,
                                            "required": attr.get("required", True)
                                        }
                                        attributes.append(api_attr)
                                logger.info(f"  ✓ Using attributes from prompt: {len(attributes)} attribute(s)")
                        if "primaryKey" in parameter_modifications:
                            primary_key = parameter_modifications["primaryKey"]
                            logger.info(f"  ✓ Using primary key from prompt: {primary_key}")
                    
                    # If no attributes from parameter_modifications, infer from data
                    if not attributes and data_for_schema_inference and len(data_for_schema_inference) > 0:
                        # Get first data record to infer structure
                        first_record = data_for_schema_inference[0]
                        if isinstance(first_record, dict):
                            logger.info(f"  🔍 Inferring schema attributes from data structure...")
                            logger.info(f"  📊 Data sample fields: {list(first_record.keys())}")
                            for field_name, field_value in first_record.items():
                                # Determine type from value
                                if isinstance(field_value, str):
                                    field_type = "string"
                                elif isinstance(field_value, (int, float)):
                                    field_type = "number"
                                elif isinstance(field_value, bool):
                                    field_type = "boolean"
                                else:
                                    field_type = "string"  # default
                                
                                attributes.append({
                                    "name": field_name,
                                    "type": {"type": field_type},
                                    "required": True
                                })
                            
                            # Use first field as primary key (use actual first field, not "id" unless it exists)
                            primary_key = [list(first_record.keys())[0]]
                            
                            logger.info(f"  ✓ Inferred {len(attributes)} attributes: {[a['name'] for a in attributes]}")
                            logger.info(f"  ✓ Primary key: {primary_key}")
                    
                    # Generate schema name if not provided
                    if not schema_name:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        schema_name = f"Schema_{timestamp}"
                    
                    # Use LLM to generate schema request body if we have attributes or data
                    if attributes:
                        logger.info(f"  📦 Using provided attributes for schema generation: {len(attributes)} attribute(s)")
                        dep_modifications = self._generate_schema_request_body(
                            schema_name=schema_name,
                            attributes=attributes,
                            primary_key=primary_key if primary_key else [attributes[0]["name"]] if attributes else ["id"],
                            data_sample=None,
                            db_type=db_type
                        )
                        # Use cached universe_id if available
                        if self.cached_universe_id and "universes" in dep_modifications:
                            dep_modifications["universes"] = [self.cached_universe_id]
                            logger.info(f"  ✓ Using cached universe_id in schema creation: {self.cached_universe_id}")
                    elif data_for_schema_inference and len(data_for_schema_inference) > 0:
                        logger.info(f"  📦 Using data for schema generation: {len(data_for_schema_inference)} record(s)")
                        dep_modifications = self._generate_schema_request_body(
                            schema_name=schema_name,
                            attributes=attributes,
                            primary_key=primary_key,
                            data_sample=data_for_schema_inference[0] if data_for_schema_inference else None,
                            db_type=db_type
                        )
                        # Use cached universe_id if available
                        if self.cached_universe_id and "universes" in dep_modifications:
                            dep_modifications["universes"] = [self.cached_universe_id]
                            logger.info(f"  ✓ Using cached universe_id in schema creation: {self.cached_universe_id}")
                    else:
                        logger.warning(f"  ⚠ No attributes or data available for schema inference!")
                        # Fallback to defaults with unique name
                        dep_modifications = {"name": schema_name}
                        # Use cached universe_id if available
                        if self.cached_universe_id:
                            dep_modifications["universes"] = [self.cached_universe_id]
                            logger.info(f"  ✓ Using cached universe_id in schema creation: {self.cached_universe_id}")
                
                dep_result = self.api_executor.execute_api(
                    tool_info=dep_tool_info,
                    parameter_modifications=dep_modifications,
                    auth_token=current_token
                )
                
                if dep_result and dep_result.get("success"):
                    # Extract values from response
                    extracted_values = self._extract_dependency_values(
                        dep_id,
                        dep_result.get("response", {})
                    )
                    if extracted_values:
                        result["values"].update(extracted_values)
                        logger.info(f"  ✓ Extracted values from {dep_id}: {extracted_values}")
                    
                    # Cache universe_id after dataverse creation
                    if dep_id == "create_dataverse_api":
                        universe_id = extracted_values.get("universes", [None])[0] if extracted_values.get("universes") else None
                        if universe_id:
                            self.cached_universe_id = universe_id
                            logger.info(f"  ✓ Cached universe_id for future use: {universe_id}")
                else:
                    logger.error(f"  ✗ Dependency {dep_id} execution failed")
        
        result["token"] = current_token
        return result
    
    def _extract_dependency_values(self, dep_id: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract values from dependency response"""
        extracted = {}
        
        if dep_id == "create_dataverse_api":
            dataverse_id = response.get("id") or response.get("dataverseID")
            if dataverse_id:
                extracted["universes"] = [dataverse_id]
        
        if dep_id in ["create_mongo_schema_api", "create_tidb_schema_api"]:
            schema_id = response.get("id") or response.get("schemaId")
            if schema_id:
                extracted["schema_id"] = schema_id
        
        return extracted
    
    def _generate_schema_request_body(
        self,
        schema_name: str,
        attributes: List[Dict[str, Any]],
        primary_key: List[str],
        data_sample: Optional[Dict[str, Any]] = None,
        db_type: str = "MONGO"
    ) -> Dict[str, Any]:
        """
        Generate schema request body using LLM from data structure
        
        Args:
            schema_name: Name for the schema
            attributes: Inferred attributes from data
            primary_key: Primary key fields
            data_sample: Sample data record
            
        Returns:
            Schema request body dictionary
        """
        from datetime import datetime
        import requests
        
        # Use cached universe_id if available, otherwise we need to create dataverse first
        universes = [self.cached_universe_id] if self.cached_universe_id else ["UNIVERSE_ID_PLACEHOLDER"]
        
        # If we have attributes inferred, use LLM to generate complete schema body
        if attributes and data_sample:
            logger.info(f"  🤖 Using LLM to generate schema request body from data structure...")
            
            # Build LLM prompt
            # Determine database type for prompt
            db_name = db_type if db_type in ["MONGO", "TIDB"] else "MONGO"
            
            # Build system prompt - use string formatting to avoid nested f-string issues
            pi_features_example = '{"COHORTS": ["' + db_name + '"]}'
            
            # Build system prompt using string concatenation to avoid f-string nesting issues
            system_prompt = """You are a schema generation assistant. Generate a complete """ + db_name + """ schema request body in JSON format based on the provided data structure.

The schema request body must include:
- name: Schema name (string)
- description: Description of the schema (string)
- universes: Array with universe ID (use the provided universe_id or "UNIVERSE_ID_PLACEHOLDER" if not available)
- tags: Object with "BLUE" key containing ["SCHEMA"] array
- primaryDb: \"""" + db_name + """\" (string) - MUST be \"""" + db_name + """\"
- piFeatures: Object with COHORTS, CONTEXT, BIGQUERY keys, each containing """ + pi_features_example + """ - MUST use \"""" + db_name + """\" not "MONGO"
- attributes: Array of attribute objects, each with:
  - name: Field name (string)
  - type: Object with "type" key (string: "string", "number", "boolean")
  - required: true (boolean)
- primaryKey: Array of field names (strings)
- dataReadAccess: "PUBLIC" (string)
- dataWriteAccess: "PUBLIC" (string)
- metadataReadAccess: "PUBLIC" (string)
- metadataWriteAccess: "PUBLIC" (string)
- visibility: "PUBLIC" (string)

IMPORTANT RULES:
1. Generate attributes based on ALL fields in the data sample - DO NOT add fields that don't exist
2. DO NOT add an "id" field if it doesn't exist in the data sample
3. Determine field types from the data values:
   - String values → type: "string"
   - Numeric values (int/float) → type: "number"
   - Boolean values → type: "boolean"
4. Use the first field as primary key (use the actual first field from data, not "id" unless it exists)
5. Include ONLY the fields that exist in the data sample - do not add extra fields
6. Description MUST be a non-empty string (use a meaningful description like "Schema for storing user data" or similar)
7. Return ONLY valid JSON, no markdown, no explanations

Example data: {"email": "test@example.com", "age": 25, "username": "testuser"}
Example response (for """ + db_name + """):
{
  "name": "Schema_20251201_070848",
  "description": "Schema created from data ingestion",
  "universes": ["UNIVERSE_ID_PLACEHOLDER"],
  "tags": {"BLUE": ["SCHEMA"]},
  "primaryDb": \"""" + db_name + """\",
  "piFeatures": {
    "COHORTS": {"COHORTS": [\"""" + db_name + """\"]},
    "CONTEXT": {"CONTEXT": [\"""" + db_name + """\"]},
    "BIGQUERY": {"BIGQUERY": [\"""" + db_name + """\"]}
  },
  "attributes": [
    {"name": "email", "type": {"type": "string"}, "required": true},
    {"name": "age", "type": {"type": "number"}, "required": true},
    {"name": "username", "type": {"type": "string"}, "required": true}
  ],
  "primaryKey": ["email"],
  "dataReadAccess": "PUBLIC",
  "dataWriteAccess": "PUBLIC",
  "metadataReadAccess": "PUBLIC",
  "metadataWriteAccess": "PUBLIC",
  "visibility": "PUBLIC"
}"""

            user_prompt = f"""Generate a {db_name} schema request body with:
- Schema name: {schema_name}
- Universe ID: {universes[0]}
- Database type: {db_name} (IMPORTANT: use "{db_name}" in primaryDb and piFeatures, NOT "MONGO")
- Data sample: {json.dumps(data_sample)}
- Inferred attributes: {json.dumps(attributes)}
- Primary key: {json.dumps(primary_key)}

Generate the complete schema request body JSON based on this data structure. Remember to use "{db_name}" for primaryDb and in all piFeatures values."""

            try:
                # Call LLM
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
                model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
                api_url = f"{ollama_url}/api/chat"
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False
                }
                
                response = requests.post(api_url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Extract JSON from LLM response
                llm_response = result.get("message", {}).get("content", "")
                
                # Try to parse JSON from response (might be wrapped in markdown)
                import re
                json_match = re.search(r'\{[\s\S]*\}', llm_response)
                if json_match:
                    schema_body = json.loads(json_match.group(0))
                    # Ensure universe_id is set correctly
                    if self.cached_universe_id:
                        schema_body["universes"] = [self.cached_universe_id]
                    # Ensure name is set
                    schema_body["name"] = schema_name
                    # Ensure description is not empty (use default if empty)
                    if not schema_body.get("description") or schema_body.get("description") == "":
                        schema_body["description"] = f"Schema created from data ingestion at {datetime.now().isoformat()}"
                    # Ensure db_type is correct
                    schema_body["primaryDb"] = db_type
                    if "piFeatures" in schema_body:
                        for feature_key in schema_body["piFeatures"]:
                            if isinstance(schema_body["piFeatures"][feature_key], dict):
                                for sub_key in schema_body["piFeatures"][feature_key]:
                                    if isinstance(schema_body["piFeatures"][feature_key][sub_key], list):
                                        schema_body["piFeatures"][feature_key][sub_key] = [db_type]
                    logger.info(f"  ✓ LLM generated schema request body with {len(schema_body.get('attributes', []))} attributes for {db_type}")
                    return schema_body
                else:
                    logger.warning("  ⚠ Could not parse JSON from LLM response, using inferred attributes")
            except Exception as e:
                logger.warning(f"  ⚠ LLM generation failed: {e}, using inferred attributes")
        
        # Fallback: use inferred attributes directly
        if attributes:
            # Ensure description is meaningful
            description = f"Schema created from data ingestion at {datetime.now().isoformat()}"
            if data_sample:
                field_names = list(data_sample.keys()) if isinstance(data_sample, dict) else []
                if field_names:
                    description = f"Schema for storing data with fields: {', '.join(field_names)}"
            
            # Determine database type (default to MONGO if not specified)
            db_name = db_type if db_type in ["MONGO", "TIDB"] else "MONGO"
            
            schema_body = {
                "name": schema_name,
                "description": description,
                "universes": universes,
                "tags": {
                    "BLUE": ["SCHEMA"]
                },
                "primaryDb": db_name,
                "piFeatures": {
                    "COHORTS": {
                        "COHORTS": [db_name]
                    },
                    "CONTEXT": {
                        "CONTEXT": [db_name]
                    },
                    "BIGQUERY": {
                        "BIGQUERY": [db_name]
                    }
                },
                "attributes": attributes,
                "primaryKey": primary_key,
                "dataReadAccess": "PUBLIC",
                "dataWriteAccess": "PUBLIC",
                "metadataReadAccess": "PUBLIC",
                "metadataWriteAccess": "PUBLIC",
                "visibility": "PUBLIC"
            }
            
            logger.info(f"  ✓ Generated schema request body with {len(attributes)} attributes (fallback) for {db_name}")
            return schema_body
        
        # Final fallback: basic structure
        db_name = db_type if db_type in ["MONGO", "TIDB"] else "MONGO"
        logger.warning(f"  ⚠ No attributes inferred, using default schema structure for {db_name}")
        return {
            "name": schema_name,
            "description": f"Schema created at {datetime.now().isoformat()}",
            "universes": universes,
            "tags": {"BLUE": ["SCHEMA"]},
            "primaryDb": db_name,
            "piFeatures": {
                "COHORTS": {"COHORTS": [db_name]},
                "CONTEXT": {"CONTEXT": [db_name]},
                "BIGQUERY": {"BIGQUERY": [db_name]}
            },
            "attributes": [
                {
                    "name": "id",
                    "type": {"type": "string"},
                    "required": True
                }
            ],
            "primaryKey": ["id"],
            "dataReadAccess": "PUBLIC",
            "dataWriteAccess": "PUBLIC",
            "metadataReadAccess": "PUBLIC",
            "metadataWriteAccess": "PUBLIC",
            "visibility": "PUBLIC"
        }
    
    def _format_response(
        self,
        tool: Dict[str, Any],
        result: Dict[str, Any],
        modifications: Dict[str, Any],
        agent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the final response for the user"""
        return {
            "success": result["success"],
            "agent": {
                "id": agent.get("agent_id"),
                "name": agent.get("name"),
                "database": agent.get("database_name")
            },
            "tool_used": {
                "id": tool.get("tool_id"),
                "name": tool.get("name"),
                "url": result.get("url")
            },
            "modifications_applied": modifications,
            "api_response": result.get("response"),
            "status_code": result.get("status_code"),
            "token": result.get("token"),
            "error": result.get("error")
        }


def create_chroma_mongo_orchestrator() -> ChromaMongoOrchestrator:
    """Factory function to create orchestrator"""
    return ChromaMongoOrchestrator()


# CLI Interface
def main():
    """Main CLI interface"""
    print("=" * 80)
    print("🤖 ChromaDB + MongoDB Orchestrator - Natural Language to API Execution")
    print("=" * 80)
    print("\nInitializing...")
    
    orchestrator = create_chroma_mongo_orchestrator()
    
    print("✅ Ready to process requests!")
    print("\nExample prompts:")
    print("  • 'generate token'")
    print("  • 'create schema named Users with attributes email as string and age as number'")
    print("  • 'create a dataverse named Test Universe'")
    print("\nType 'exit' to quit\n")
    
    try:
        while True:
            user_input = input("📝 Your request: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            print("\n" + "-" * 80)
            result = orchestrator.process_user_request(user_input)
            print("-" * 80)
            
            # Display result
            if result["success"]:
                print("\n✅ SUCCESS")
                
                if result.get("agent"):
                    print(f"\n🤖 Agent: {result['agent']['name']}")
                
                print(f"\n🔧 Tool Used: {result['tool_used']['name']}")
                print(f"🌐 URL: {result['tool_used']['url']}")
                
                if result["modifications_applied"]:
                    print(f"\n📝 Modifications Applied:")
                    for key, value in result["modifications_applied"].items():
                        print(f"   • {key}: {value}")
                
                print(f"\n📊 Response (Status {result['status_code']}):")
                print(json.dumps(result["api_response"], indent=2))
                
                if result.get("token"):
                    print(f"\n🔑 Token Generated: {result['token'][:50]}...")
            else:
                print("\n❌ FAILED")
                # Handle different error formats
                if isinstance(result, dict):
                    if result.get("agent"):
                        agent = result.get("agent")
                        if isinstance(agent, dict):
                            print(f"Agent: {agent.get('name', 'Unknown')}")
                        else:
                            print(f"Agent: {agent}")
                    elif result.get("agent_name"):
                        print(f"Agent: {result['agent_name']}")
                    
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    if result.get('suggestion'):
                        print(f"Suggestion: {result['suggestion']}")
                else:
                    print(f"Error: {result}")
            
            print("\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")


if __name__ == "__main__":
    main()

