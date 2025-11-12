#!/usr/bin/env python3
"""
Main Orchestrator - Ties everything together
Handles the complete flow from prompt to API execution
"""

import logging
import re
import json
import ast
from typing import Dict, Any, Optional, List
from neo4j_enhanced_manager import create_neo4j_manager
from embedding_service_ollama import create_ollama_embedding_service
from llm_prompt_parser import create_prompt_parser
from api_executor import create_api_executor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIOrchestrator:
    """
    Main orchestrator that handles the complete pipeline:
    1. User prompt → Search Neo4j for relevant tool
    2. Parse prompt for parameter modifications
    3. Handle dependencies (e.g., generate token first)
    4. Execute API with modified parameters
    5. Return response
    """
    
    def __init__(self):
        self.neo4j_manager = create_neo4j_manager()
        self.embedding_service = create_ollama_embedding_service()
        self.prompt_parser = create_prompt_parser()
        self.api_executor = create_api_executor()
        
        # Ensure Neo4j indexes exist
        self.neo4j_manager.create_indexes()
        
        logger.info("✅ Orchestrator initialized successfully")
    
    def process_user_request(self, user_prompt: str) -> Dict[str, Any]:
        """
        Process a user's natural language request
        
        Args:
            user_prompt: User's natural language request
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"📝 Processing user request: '{user_prompt}'")
        
        try:
            # Step 1: Find relevant tool using semantic search
            logger.info("🔍 Step 1: Searching for relevant API tool...")
            tool = self._find_relevant_tool(user_prompt)
            
            if not tool:
                return {
                    "success": False,
                    "error": "No matching API tool found for your request",
                    "suggestion": "Try rephrasing your request or check available tools"
                }
            
            logger.info(f"✓ Found tool: {tool['name']} (ID: {tool['tool_id']})")
            logger.info(f"  Similarity score: {tool['similarity_score']:.3f}")
            
            # Step 2: Parse prompt for parameter modifications
            logger.info("🧠 Step 2: Parsing prompt for parameter modifications...")
            
            # Step 2.1: For data ingestion, try to extract data array directly from prompt first
            # This ensures we preserve ALL fields without LLM dropping any
            if tool["tool_id"] == "ingest_data_api":
                extracted_data = self._extract_data_array_from_prompt(user_prompt)
                if extracted_data:
                    logger.info(f"✓ Directly extracted data array from prompt: {len(extracted_data)} records")
                    # Use direct extraction instead of LLM for data field
                    parameter_modifications = {"data": extracted_data}
                    # Still use LLM for other fields (like schema_id if mentioned)
                    llm_modifications = self.prompt_parser.parse_prompt_for_parameters(user_prompt, tool)
                    # Merge LLM modifications but keep our directly extracted data
                    if llm_modifications:
                        parameter_modifications.update({k: v for k, v in llm_modifications.items() if k != "data"})
                else:
                    # Fall back to LLM extraction
                    if self.prompt_parser.should_use_defaults(user_prompt):
                        logger.info("✓ Using default parameters (no modifications)")
                        parameter_modifications = {}
                    else:
                        parameter_modifications = self.prompt_parser.parse_prompt_for_parameters(
                            user_prompt, 
                            tool
                        )
                        if parameter_modifications:
                            parameter_modifications = self._cleanup_modifications(parameter_modifications, user_prompt)
                            logger.info(f"✓ Extracted modifications: {parameter_modifications}")
                        else:
                            logger.info("✓ No modifications needed")
            else:
                # For other tools, use normal LLM extraction
                if self.prompt_parser.should_use_defaults(user_prompt):
                    logger.info("✓ Using default parameters (no modifications)")
                    parameter_modifications = {}
                else:
                    parameter_modifications = self.prompt_parser.parse_prompt_for_parameters(
                        user_prompt, 
                        tool
                    )
                    if parameter_modifications:
                        # Clean up invalid name fields (if LLM extracted entire prompt as name)
                        parameter_modifications = self._cleanup_modifications(parameter_modifications, user_prompt)
                        logger.info(f"✓ Extracted modifications: {parameter_modifications}")
                    else:
                        logger.info("✓ No modifications needed")
            
            # Step 2.5: Extract IDs from prompt if provided (before handling dependencies)
            # IMPORTANT: Only trust IDs found in the actual prompt text, not LLM hallucinations
            user_provided_universe_id = self._extract_universe_id_from_prompt(user_prompt, parameter_modifications)
            user_provided_schema_id = self._extract_schema_id_from_prompt(user_prompt, parameter_modifications)
            
            # Step 2.6: Validate and clean up schemaId from modifications
            # If LLM extracted a schemaId but it's not in the prompt, it's a hallucination - remove it
            if "schema_id" in parameter_modifications or "schemaId" in parameter_modifications:
                extracted_schema_id = parameter_modifications.get("schema_id") or parameter_modifications.get("schemaId")
                # Only keep it if it was actually found in the prompt text
                if not user_provided_schema_id:
                    # LLM hallucinated it - remove it
                    logger.warning(f"  ⚠ Removing hallucinated schemaId from LLM response: {extracted_schema_id}")
                    parameter_modifications.pop("schema_id", None)
                    parameter_modifications.pop("schemaId", None)
                    user_provided_schema_id = None
            
            # Step 2.7: Special handling for data ingestion - infer schema from data if needed
            if tool["tool_id"] == "ingest_data_api" and not user_provided_schema_id:
                # Validate data extraction - check if all fields are preserved
                if "data" in parameter_modifications:
                    data = parameter_modifications["data"]
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        fields_in_data = list(data[0].keys())
                        logger.info(f"  ✓ Data extraction validated: {len(fields_in_data)} fields found: {fields_in_data}")
                
                # Extract data object from modifications to infer schema
                schema_attributes = self._infer_schema_from_data(parameter_modifications)
                if schema_attributes:
                    inferred_field_names = [attr.get("name") for attr in schema_attributes]
                    logger.info(f"✓ Inferred schema attributes from data: {len(schema_attributes)} fields: {inferred_field_names}")
                    # Store for later schema creation
                    parameter_modifications["_inferred_attributes"] = schema_attributes
                else:
                    logger.warning("  ⚠ Could not infer schema from data - data may be missing or invalid")
            
            # Step 3: Handle dependencies
            logger.info("🔗 Step 3: Checking dependencies...")
            dependency_results = self._handle_dependencies(
                tool, 
                skip_dataverse_creation=user_provided_universe_id is not None,
                skip_schema_creation=user_provided_schema_id is not None,
                inferred_schema_attributes=parameter_modifications.get("_inferred_attributes")
            )
            auth_token = dependency_results.get("token")
            dependency_values = dependency_results.get("values", {})
            
            # Use user-provided universe ID if available, otherwise use dependency value
            if user_provided_universe_id:
                dependency_values["universes"] = [user_provided_universe_id]
                logger.info(f"✓ Using user-provided universe ID: {user_provided_universe_id}")
            elif dependency_values:
                logger.info(f"✓ Added dependency values: {dependency_values}")
            
            # Use user-provided schema ID if available, otherwise use dependency value
            if user_provided_schema_id:
                dependency_values["schema_id"] = user_provided_schema_id
                logger.info(f"✓ Using user-provided schema ID: {user_provided_schema_id}")
            elif "schema_id" in dependency_values:
                logger.info(f"✓ Using schema ID from dependency: {dependency_values['schema_id']}")
            
            # Remove internal fields from modifications
            parameter_modifications.pop("_inferred_attributes", None)
            
            # Merge dependency values into parameter modifications
            # (e.g., dataverse ID from create_dataverse_api, schema_id from create_mongo_schema_api)
            if dependency_values:
                parameter_modifications = {**parameter_modifications, **dependency_values}
            
            # Generate unique name if not provided (to avoid conflicts)
            # Skip for data ingestion - it doesn't need a name field
            if tool["tool_id"] != "ingest_data_api" and "name" not in parameter_modifications:
                unique_name = self._generate_unique_name_for_tool(tool["tool_id"])
                if unique_name:
                    parameter_modifications["name"] = unique_name
                    logger.info(f"✓ Generated unique name: {unique_name}")
            
            # Step 4: Execute API
            logger.info("🚀 Step 4: Executing API call...")
            result = self.api_executor.execute_api(
                tool_info=tool,
                parameter_modifications=parameter_modifications,
                auth_token=auth_token
            )
            
            # Step 5: Cache token if generated
            if result.get("token"):
                self.api_executor.cache_token(tool["tool_id"], result["token"])
            
            # Format final response
            return self._format_response(tool, result, parameter_modifications)
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _find_relevant_tool(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant tool for the user's request
        
        Args:
            user_prompt: User's request
            
        Returns:
            Tool information or None
        """
        # Create embedding for user prompt
        query_embedding = self.embedding_service.create_embedding(user_prompt)
        
        # Search Neo4j
        tools = self.neo4j_manager.search_tools_by_embedding(
            query_embedding=query_embedding,
            limit=1,
            similarity_threshold=0.6  # Adjust based on your needs
        )
        
        return tools[0] if tools else None
    
    def _extract_data_array_from_prompt(self, user_prompt: str) -> Optional[list]:
        """
        Extract data array directly from prompt using regex/JSON parsing
        This ensures we preserve ALL fields without LLM dropping any
        
        Args:
            user_prompt: User's natural language request
            
        Returns:
            Data array if found, None otherwise
        """
        # json and re are already imported at module level
        
        # Try to find JSON array pattern in the prompt
        # Look for patterns like: [{"id": "123", "name": "John", "age": 30}]
        # First, try to find array brackets and extract the complete JSON array
        bracket_start = user_prompt.find('[')
        if bracket_start != -1:
            # Find the matching closing bracket by counting brackets
            bracket_count = 0
            bracket_end = bracket_start
            in_string = False
            escape_next = False
            
            for i in range(bracket_start, len(user_prompt)):
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
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            bracket_end = i + 1
                            break
            
            if bracket_end > bracket_start:
                json_str = user_prompt[bracket_start:bracket_end]
                logger.debug(f"  Attempting to parse JSON: {json_str[:100]}...")
                
                # Try multiple parsing strategies in order
                # Strategy 1: Direct JSON parsing (most common case)
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"  ✓ Extracted data array directly from prompt (JSON): {len(data)} record(s)")
                        if isinstance(data[0], dict):
                            fields = list(data[0].keys())
                            logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                        return data
                except json.JSONDecodeError:
                    pass
                
                # Strategy 2: Try Python literal eval (handles single quotes and Python syntax)
                try:
                    data = ast.literal_eval(json_str)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"  ✓ Extracted data array directly from prompt (Python syntax): {len(data)} record(s)")
                        if isinstance(data[0], dict):
                            fields = list(data[0].keys())
                            logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                        return data
                except (ValueError, SyntaxError):
                    pass
                
                # Strategy 3: Try replacing single quotes with double quotes (if JSON incorrectly uses single quotes)
                try:
                    fixed_json = json_str.replace("'", '"')
                    data = json.loads(fixed_json)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"  ✓ Extracted data array directly from prompt (fixed quotes): {len(data)} record(s)")
                        if isinstance(data[0], dict):
                            fields = list(data[0].keys())
                            logger.info(f"  ✓ Fields in data ({len(fields)} total): {fields}")
                        return data
                except json.JSONDecodeError:
                    pass
                
                logger.warning(f"  ⚠ Could not parse JSON array from prompt: {json_str[:100]}...")
        
        # If no JSON array found, try to extract from "ingest data: ..." pattern
        # Look for the part after "ingest data:" or "insert data:"
        ingest_patterns = [
            r'ingest\s+data\s*:\s*(.+)',
            r'insert\s+data\s*:\s*(.+)',
            r'data\s*:\s*(.+)',
        ]
        
        for pattern in ingest_patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                data_part = match.group(1).strip()
                # Try to parse as JSON
                try:
                    data = json.loads(data_part)
                    if isinstance(data, list):
                        logger.info(f"  ✓ Extracted data array from ingest pattern: {len(data)} record(s)")
                        return data
                except:
                    # If it starts with [, try to find the complete array
                    if data_part.startswith('['):
                        # Try to find the matching closing bracket
                        bracket_count = 0
                        end_idx = 0
                        for i, char in enumerate(data_part):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > 0:
                            try:
                                data = json.loads(data_part[:end_idx])
                                if isinstance(data, list):
                                    logger.info(f"  ✓ Extracted data array (found complete array): {len(data)} record(s)")
                                    return data
                            except:
                                pass
        
        return None
    
    def _cleanup_modifications(self, modifications: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """
        Clean up modifications to remove invalid values
        
        Args:
            modifications: Extracted parameter modifications
            user_prompt: Original user prompt
            
        Returns:
            Cleaned modifications
        """
        cleaned = modifications.copy()
        
        # Remove name field if it's the entire prompt (LLM extraction error)
        if "name" in cleaned:
            name_value = cleaned["name"]
            if isinstance(name_value, str):
                # Check if name is suspiciously long or contains common prompt words
                name_lower = name_value.lower()
                prompt_lower = user_prompt.lower()
                
                # Remove if:
                # 1. Name is too long (> 50 chars is suspicious for a schema name)
                # 2. Name contains prompt phrases like "create table", "with attributes", etc.
                # 3. Name is very similar to the prompt (more than 80% match)
                suspicious_phrases = ["create table", "create schema", "with attributes", "put primary key", "in mongo"]
                is_suspicious = (
                    len(name_value) > 50 or
                    any(phrase in name_lower for phrase in suspicious_phrases) or
                    (len(prompt_lower) > 0 and len(name_lower) / len(prompt_lower) > 0.8)
                )
                
                if is_suspicious:
                    logger.warning(f"  ⚠ Removing invalid name field: {name_value[:50]}... (likely extraction error)")
                    del cleaned["name"]
        
        # Remove any keys with invalid formats (like "attributes[0].type")
        keys_to_remove = [key for key in cleaned.keys() if "[" in key and "]" in key]
        for key in keys_to_remove:
            logger.warning(f"  ⚠ Removing invalid key format: {key}")
            del cleaned[key]
        
        return cleaned
    
    def _extract_universe_id_from_prompt(self, user_prompt: str, parameter_modifications: Dict[str, Any]) -> Optional[str]:
        """
        Extract universe ID from user prompt or parameter modifications
        
        Args:
            user_prompt: User's natural language request
            parameter_modifications: Already extracted parameter modifications
            
        Returns:
            Universe ID if found, None otherwise
        """
        # Check if already extracted in modifications
        if "universes" in parameter_modifications:
            universes = parameter_modifications["universes"]
            if isinstance(universes, list) and len(universes) > 0:
                universe_id = universes[0]
                if universe_id and universe_id != "UNIVERSE_ID_PLACEHOLDER":
                    return universe_id
        
        # Try to extract from prompt using regex
        # Look for patterns like "universeid as 6911d6f568829872bc56d027" or "universe id 6911d6f568829872bc56d027"
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
                logger.info(f"  ✓ Extracted universe ID from prompt: {universe_id}")
                return universe_id
        
        return None
    
    def _extract_schema_id_from_prompt(self, user_prompt: str, parameter_modifications: Dict[str, Any]) -> Optional[str]:
        """
        Extract schema ID from user prompt text ONLY (not from LLM modifications)
        This prevents trusting hallucinated schemaIds from the LLM
        
        Args:
            user_prompt: User's natural language request
            parameter_modifications: Already extracted parameter modifications (for reference only)
            
        Returns:
            Schema ID if found in prompt text, None otherwise
        """
        # ONLY extract from prompt text using regex - don't trust LLM modifications
        # Look for patterns like "schemaid as 67dcf66d5ccb2c54260fb156" or "schema id 67dcf66d5ccb2c54260fb156"
        patterns = [
            r'schemaid\s+as\s+([a-f0-9]{24})',
            r'schema\s+id\s+as\s+([a-f0-9]{24})',
            r'schemaid[:\s]+([a-f0-9]{24})',
            r'schema[:\s]+id[:\s]+([a-f0-9]{24})',
            r'with\s+schemaid\s+as\s+([a-f0-9]{24})',
            r'into\s+schema\s+([a-f0-9]{24})',
            r'schema\s+([a-f0-9]{24})',
            r'ingest\s+data\s+into\s+schema\s+([a-f0-9]{24})',  # "ingest data into schema 123..."
            r'insert\s+data\s+into\s+schema\s+([a-f0-9]{24})',  # "insert data into schema 123..."
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE)
            if match:
                schema_id = match.group(1)
                logger.info(f"  ✓ Extracted schema ID from prompt text: {schema_id}")
                return schema_id
        
        # No schemaId found in prompt text
        return None
    
    def _infer_schema_from_data(self, parameter_modifications: Dict[str, Any]) -> Optional[list]:
        """
        Infer schema attributes from data object in parameter modifications
        
        Args:
            parameter_modifications: Parameter modifications that may contain data array
            
        Returns:
            List of attribute definitions or None
        """
        # Get data array from modifications
        data = parameter_modifications.get("data")
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        # Use first data object to infer schema
        sample_record = data[0]
        if not isinstance(sample_record, dict):
            return None
        
        # Infer attributes from sample record
        attributes = []
        for field_name, field_value in sample_record.items():
            # Determine type from value
            if isinstance(field_value, str):
                # Check if it's a UUID format
                if len(field_value) == 36 and field_value.count('-') == 4:
                    attr_type = "string"  # UUID as string
                else:
                    attr_type = "string"
            elif isinstance(field_value, (int, float)):
                attr_type = "number"
            elif isinstance(field_value, bool):
                attr_type = "boolean"
            elif isinstance(field_value, list):
                attr_type = "string"  # Arrays stored as strings or handled separately
            else:
                attr_type = "string"  # Default to string
            
            # Determine if required (id is usually required)
            required = (field_name == "id")
            
            attributes.append({
                "name": field_name,
                "type": {"type": attr_type},
                "required": required
            })
        
        # Set primary key (usually "id" if exists)
        if "id" in sample_record:
            logger.info(f"  ✓ Inferred {len(attributes)} attributes with primary key: id")
        else:
            logger.info(f"  ✓ Inferred {len(attributes)} attributes (no id field found)")
        
        return attributes
    
    def _handle_dependencies(
        self, 
        tool: Dict[str, Any], 
        skip_dataverse_creation: bool = False,
        skip_schema_creation: bool = False,
        inferred_schema_attributes: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Handle tool dependencies (e.g., generate token, create dataverse, etc.)
        
        Args:
            tool: Tool that may have dependencies
            
        Returns:
            Dictionary with 'token' and 'values' (extracted from dependency responses)
        """
        result = {"token": None, "values": {}}
        
        if not tool.get("authentication_required", False):
            logger.info("  ✓ No authentication required")
            return result
        
        # Get dependencies
        dependencies = self.neo4j_manager.get_tool_dependencies(tool["tool_id"])
        
        if not dependencies:
            logger.warning("  ⚠ Tool requires auth but has no dependencies defined")
            return result
        
        logger.info(f"  Dependencies found: {dependencies}")
        
        # First, handle token generation (must come first)
        current_token = None
        token_dep_id = None
        
        # Find token generation dependency
        for dep_id in dependencies:
            dep_tool = self._get_tool_by_id(dep_id)
            if dep_tool and dep_tool.get("returns_token", False):
                token_dep_id = dep_id
                cached_token = self.api_executor.get_cached_token(dep_id)
                if cached_token:
                    logger.info(f"  ✓ Using cached token from {dep_id}")
                    current_token = cached_token
                else:
                    logger.info(f"  🔑 Generating token using {dep_id}...")
                    token_result = self._execute_dependency(dep_tool, {}, None)
                    if token_result and token_result.get("success"):
                        current_token = token_result.get("token")
                        if current_token:
                            self.api_executor.cache_token(dep_id, current_token)
                            logger.info("  ✓ Token generated and cached")
                break
        
        if not current_token:
            logger.warning("  ⚠ No token available - dependencies requiring auth may fail")
        
        # Then, handle other dependencies that need the token
        # If we need to create schema for data ingestion, we also need dataverse (if not skipped)
        needs_dataverse = (
            "create_mongo_schema_api" in dependencies and 
            not skip_schema_creation and 
            not skip_dataverse_creation
        )
        
        # Process dataverse creation first if needed (before schema creation)
        if needs_dataverse and "create_dataverse_api" not in dependencies:
            # Check if dataverse tool exists and create it
            dataverse_tool = self._get_tool_by_id("create_dataverse_api")
            if dataverse_tool and current_token:
                logger.info(f"  🔧 Creating dataverse (required for schema creation)...")
                dep_modifications = self._generate_unique_dataverse_name()
                dep_result = self._execute_dependency(dataverse_tool, dep_modifications, current_token)
                if dep_result and dep_result.get("success"):
                    extracted_values = self._extract_dependency_values("create_dataverse_api", dep_result.get("response", {}))
                    if extracted_values:
                        result["values"].update(extracted_values)
                        logger.info(f"  ✓ Extracted values from create_dataverse_api: {extracted_values}")
        
        for dep_id in dependencies:
            if dep_id == token_dep_id:
                continue  # Skip token generation, already handled
            
            dep_tool = self._get_tool_by_id(dep_id)
            if not dep_tool:
                logger.error(f"  ✗ Dependency tool not found: {dep_id}")
                continue
            
            # Execute dependencies that require authentication
            if dep_tool.get("authentication_required", False):
                # Skip dataverse creation if user provided universe ID or if explicitly skipped
                if dep_id == "create_dataverse_api" and skip_dataverse_creation:
                    logger.info(f"  ⏭ Skipping {dep_id} (universe ID provided by user)")
                    continue
                
                # Skip dataverse if already created above
                if dep_id == "create_dataverse_api" and "universes" in result["values"]:
                    logger.info(f"  ⏭ Skipping {dep_id} (already created)")
                    continue
                
                # Skip schema creation if user provided schema ID or if explicitly skipped
                if dep_id == "create_mongo_schema_api" and skip_schema_creation:
                    logger.info(f"  ⏭ Skipping {dep_id} (schema ID provided by user)")
                    continue
                
                if not current_token:
                    logger.warning(f"  ⚠ Dependency {dep_id} needs token but none available")
                    continue
                
                logger.info(f"  🔧 Executing dependency: {dep_id}...")
                
                # Prepare dependency modifications
                dep_modifications = {}
                if dep_id == "create_dataverse_api":
                    dep_modifications = self._generate_unique_dataverse_name()
                elif dep_id == "create_mongo_schema_api":
                    # Initialize dep_modifications
                    dep_modifications = {}
                    
                    # Use inferred attributes if available (for data ingestion without schemaId)
                    if inferred_schema_attributes:
                        # Determine primary key (prefer "id", otherwise use first field)
                        primary_key_field = None
                        for attr in inferred_schema_attributes:
                            if attr.get("name") == "id":
                                primary_key_field = "id"
                                break
                        
                        if not primary_key_field and len(inferred_schema_attributes) > 0:
                            # Use first field as primary key if no "id" field
                            primary_key_field = inferred_schema_attributes[0].get("name")
                            logger.warning(f"  ⚠ No 'id' field found, using '{primary_key_field}' as primary key")
                        
                        dep_modifications.update({
                            "attributes": inferred_schema_attributes,
                            "primaryKey": [primary_key_field] if primary_key_field else []
                        })
                        logger.info(f"  ✓ Using inferred attributes for schema creation: {len(inferred_schema_attributes)} fields, primaryKey: {dep_modifications['primaryKey']}")
                    
                    # Use dataverse ID if available from previous dependency
                    if "universes" in result["values"]:
                        dep_modifications["universes"] = result["values"]["universes"]
                        logger.info(f"  ✓ Using dataverse ID for schema creation: {result['values']['universes']}")
                
                dep_result = self._execute_dependency(dep_tool, dep_modifications, current_token)
                
                if dep_result and dep_result.get("success"):
                    # Extract values from response based on dependency type
                    extracted_values = self._extract_dependency_values(dep_id, dep_result.get("response", {}))
                    if extracted_values:
                        result["values"].update(extracted_values)
                        logger.info(f"  ✓ Extracted values from {dep_id}: {extracted_values}")
                else:
                    logger.error(f"  ✗ Dependency {dep_id} execution failed")
                    if dep_result:
                        error_msg = dep_result.get("response", {}).get("errorMessage", "Unknown error")
                        logger.error(f"  Error: {error_msg}")
        
        result["token"] = current_token
        return result
    
    def _get_tool_by_id(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tool information by tool_id
        
        Args:
            tool_id: Tool ID to search for
            
        Returns:
            Tool information or None
        """
        # Search for tool using embedding (semantic search)
        query_embedding = self.embedding_service.create_embedding(tool_id)
        tools = self.neo4j_manager.search_tools_by_embedding(
            query_embedding=query_embedding,
            limit=10  # Get more results to find exact match
        )
        
        # Find exact match by tool_id
        for tool in tools:
            if tool.get("tool_id") == tool_id:
                return tool
        
        logger.error(f"  ✗ Tool not found: {tool_id}")
        return None
    
    def _execute_dependency(
        self, 
        dep_tool: Dict[str, Any], 
        parameter_modifications: Dict[str, Any],
        auth_token: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a dependency tool
        
        Args:
            dep_tool: Dependency tool information
            parameter_modifications: Parameter modifications for dependency
            auth_token: Authentication token if needed
            
        Returns:
            Execution result
        """
        return self.api_executor.execute_api(
            tool_info=dep_tool,
            parameter_modifications=parameter_modifications,
            auth_token=auth_token
        )
    
    def _extract_dependency_values(self, dep_id: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract values from dependency response that are needed by the main tool
        
        Args:
            dep_id: Dependency tool ID
            response: Response from dependency execution
            
        Returns:
            Dictionary of extracted values to pass to main tool
        """
        extracted = {}
        
        # Extract dataverse ID from create_dataverse_api response
        if dep_id == "create_dataverse_api":
            # Response should have "id" or "dataverseID" field
            dataverse_id = response.get("id") or response.get("dataverseID")
            if dataverse_id:
                # Schema creation needs this in "universes" array
                extracted["universes"] = [dataverse_id]
                logger.info(f"  ✓ Extracted dataverse ID: {dataverse_id}")
        
        # Extract schema ID from create_mongo_schema_api response
        if dep_id == "create_mongo_schema_api":
            # Response should have "id" or "schemaId" field
            schema_id = response.get("id") or response.get("schemaId")
            if schema_id:
                # Data ingestion needs this as schema_id
                extracted["schema_id"] = schema_id
                logger.info(f"  ✓ Extracted schema ID: {schema_id}")
        
        # Add more extraction logic for other dependencies as needed
        
        return extracted
    
    def _generate_unique_dataverse_name(self) -> Dict[str, Any]:
        """
        Generate a unique dataverse name with timestamp to avoid conflicts
        
        Returns:
            Dictionary with name modification
        """
        unique_name = self._generate_unique_name_for_tool("create_dataverse_api")
        if unique_name:
            return {"name": unique_name}
        return {}
    
    def _generate_unique_name_for_tool(self, tool_id: str) -> Optional[str]:
        """
        Generate a unique name with timestamp for a tool to avoid conflicts
        
        Args:
            tool_id: Tool ID to generate name for
            
        Returns:
            Unique name with timestamp or None
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate prefix based on tool type
        if tool_id == "create_dataverse_api":
            return f"Dataverse_{timestamp}"
        elif tool_id == "create_mongo_schema_api":
            return f"Schema_{timestamp}"
        else:
            # Generic name for other tools
            return f"Resource_{timestamp}"
    
    def _format_response(
        self, 
        tool: Dict[str, Any], 
        result: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the final response for the user"""
        
        return {
            "success": result["success"],
            "tool_used": {
                "id": tool["tool_id"],
                "name": tool["name"],
                "url": result.get("url")
            },
            "modifications_applied": modifications,
            "api_response": result.get("response"),
            "status_code": result.get("status_code"),
            "token": result.get("token"),  # Include if token was generated
            "error": result.get("error")
        }
    
    def close(self):
        """Clean up resources"""
        self.neo4j_manager.close()
        logger.info("✓ Orchestrator closed")


def create_orchestrator() -> APIOrchestrator:
    """Factory function to create orchestrator"""
    return APIOrchestrator()


# ============================================================================
# CLI Interface for testing
# ============================================================================

def main():
    """Main CLI interface"""
    print("=" * 80)
    print("🤖 API Orchestrator - Natural Language to API Execution")
    print("=" * 80)
    print("\nInitializing...")
    
    orchestrator = create_orchestrator()
    
    print("✅ Ready to process requests!")
    print("\nExample prompts:")
    print("  • 'generate token'")
    print("  • 'use requestType as SUBTENANT and password as Gaian1234'")
    print("  • 'create a dataverse named Test Universe'")
    print("  • 'get me a token'")
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
                print(f"\n🔧 Tool Used: {result['tool_used']['name']}")
                print(f"🌐 URL: {result['tool_used']['url']}")
                
                if result["modifications_applied"]:
                    print(f"\n📝 Modifications Applied:")
                    for key, value in result["modifications_applied"].items():
                        print(f"   • {key}: {value}")
                
                print(f"\n📊 Response (Status {result['status_code']}):")
                import json
                print(json.dumps(result["api_response"], indent=2))
                
                if result.get("token"):
                    print(f"\n🔑 Token Generated: {result['token'][:50]}...")
            else:
                print("\n❌ FAILED")
                print(f"Error: {result.get('error')}")
                if result.get('suggestion'):
                    print(f"Suggestion: {result['suggestion']}")
            
            print("\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
