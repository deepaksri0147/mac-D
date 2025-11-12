#!/usr/bin/env python3
"""
LLM-Based Prompt Parser using Ollama
Extracts parameter modifications from natural language
"""

import os
import json
import logging
import re
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OllamaPromptParser:
    """Parse natural language prompts to extract API parameter changes"""
    
    def __init__(
        self, 
        base_url: str = None,
        model: str = None
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama-keda.mobiusdtaas.ai")
        self.model = model or os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:latest")
        self.api_url = f"{self.base_url}/api/chat"
    
    def parse_prompt_for_parameters(
        self, 
        user_prompt: str,
        tool_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse user prompt and extract parameter modifications
        
        Args:
            user_prompt: Natural language request from user
            tool_info: Tool information including field_descriptions and requestBody
            
        Returns:
            Dictionary of parameters to modify
        """
        
        # Build context for LLM
        field_descriptions = tool_info.get("field_descriptions", {})
        current_values = tool_info.get("requestBody", {})
        
        # Create LLM prompt
        llm_prompt = self._build_extraction_prompt(
            user_prompt, 
            field_descriptions, 
            current_values
        )
        
        try:
            # Call Ollama using chat API for better JSON support
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": llm_prompt
                        }
                    ],
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {}
            
            result = response.json()
            llm_response = result.get("message", {}).get("content", "{}")
            
            # Clean up response (remove markdown code blocks if present)
            llm_response = llm_response.strip()
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:]
            elif llm_response.startswith("```"):
                llm_response = llm_response[3:]
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()
            
            # Parse JSON response
            try:
                modifications = json.loads(llm_response)
                logger.info(f"✓ Extracted parameters: {modifications}")
                return modifications.get("modifications", {})
            except json.JSONDecodeError as e:
                logger.error(f"✗ Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Raw response: {llm_response}")
                # Try to extract JSON from response if it's embedded in text
                # Look for JSON object with "modifications" key
                json_pattern = r'\{[^{}]*(?:"modifications"[^{}]*\{[^{}]*\}[^{}]*)*\}'
                json_match = re.search(json_pattern, llm_response, re.DOTALL)
                if json_match:
                    try:
                        modifications = json.loads(json_match.group())
                        logger.info(f"✓ Extracted parameters (from text): {modifications}")
                        return modifications.get("modifications", {})
                    except json.JSONDecodeError:
                        # Try a more aggressive extraction - find the first complete JSON object
                        start_idx = llm_response.find('{')
                        if start_idx != -1:
                            brace_count = 0
                            end_idx = start_idx
                            for i in range(start_idx, len(llm_response)):
                                if llm_response[i] == '{':
                                    brace_count += 1
                                elif llm_response[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                            if end_idx > start_idx:
                                try:
                                    json_str = llm_response[start_idx:end_idx]
                                    modifications = json.loads(json_str)
                                    logger.info(f"✓ Extracted parameters (aggressive): {modifications}")
                                    return modifications.get("modifications", {})
                                except:
                                    pass
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Error calling Ollama API: {e}")
            return {}
    
    def _build_extraction_prompt(
        self, 
        user_prompt: str,
        field_descriptions: Dict[str, str],
        current_values: Dict[str, Any]
    ) -> str:
        """Build the prompt for LLM to extract parameters"""
        
        prompt = f"""You are an API parameter extraction assistant. 

USER REQUEST: "{user_prompt}"

AVAILABLE FIELDS AND THEIR DESCRIPTIONS:
{json.dumps(field_descriptions, indent=2)}

CURRENT DEFAULT VALUES:
{json.dumps(current_values, indent=2)}

TASK:
Extract which fields the user wants to modify and their new values from the user request.
Return ONLY a JSON object in this exact format (no other text):

{{
  "modifications": {{
    "field_name": "new_value"
  }}
}}

RULES:
1. Only include fields that the user explicitly wants to change
2. Use exact field names from AVAILABLE FIELDS (match case exactly)
3. Convert values to appropriate types (string, number, boolean, etc.)
4. If user mentions a name after "named" or "name is", extract it as the "name" field value
5. If user doesn't specify any changes, return empty modifications: {{"modifications": {{}}}}
6. Be case-sensitive with field names
7. Return ONLY valid JSON, no markdown, no explanations
8. Preserve the exact value the user provides (don't modify case or formatting)

EXAMPLES:
- "use requestType as SUBTENANT" → {{"modifications": {{"requestType": "SUBTENANT"}}}}
- "change password to Gaian1234" → {{"modifications": {{"password": "Gaian1234"}}}}
- "set userName to test@example.com and requestType to USER" → {{"modifications": {{"userName": "test@example.com", "requestType": "USER"}}}}
- "create table in mongo with attributes name as string and id as number put primary key as the id" → {{"modifications": {{"attributes": [{{"name": "name", "type": "string", "required": true}}, {{"name": "id", "type": "number", "required": true}}], "primaryKey": ["id"]}}}}
- "create schema named Users with attributes email as string, age as number" → {{"modifications": {{"name": "Users", "attributes": [{{"name": "email", "type": "string", "required": true}}, {{"name": "age", "type": "number", "required": true}}]}}}}
- "create table with universeid as 6911d6f568829872bc56d027" → {{"modifications": {{"universes": ["6911d6f568829872bc56d027"]}}}}
- "ingest data: [{{"id": "123", "name": "John", "age": 30}}]" → {{"modifications": {{"data": [{{"id": "123", "name": "John", "age": 30}}]}}}}
- "ingest data: [{{"id": "1", "name": "Test"}}]" → {{"modifications": {{"data": [{{"id": "1", "name": "Test"}}]}}}}
- "insert data into schema 67dcf66d5ccb2c54260fb156 with data: [{{"id": "1", "name": "Test"}}]" → {{"modifications": {{"schema_id": "67dcf66d5ccb2c54260fb156", "data": [{{"id": "1", "name": "Test"}}]}}}}
- "ingest data into schema 6911e00e19be331b9bebdf93 with data: [{{"id": "1", "name": "Test"}}]" → {{"modifications": {{"schema_id": "6911e00e19be331b9bebdf93", "data": [{{"id": "1", "name": "Test"}}]}}}}
- "generate token" → {{"modifications": {{}}}}

CRITICAL RULES FOR DATA INGESTION:
- For "ingest data" or "insert data" requests: Extract the COMPLETE data array from the user request
- CRITICAL: Preserve EVERY SINGLE FIELD in the data objects - DO NOT drop any fields like "age", "email", etc.
- If user provides: [{{"id": "123", "name": "John", "age": 30}}], you MUST extract ALL THREE fields: id, name, AND age
- DO NOT extract schema_id unless the user explicitly provides it in the prompt (like "schema 123" or "schemaid as 123")
- If user only provides data without schema_id, return ONLY the data field - do NOT add schema_id
- When extracting data arrays, copy the ENTIRE JSON structure exactly as provided - preserve all fields, all values, all types
- Example: "ingest data: [{{"id": "1", "name": "John", "age": 30}}]" → {{"modifications": {{"data": [{{"id": "1", "name": "John", "age": 30}}]}}}}
- Example: "ingest data: [{{"id": "123", "name": "John", "age": 30, "email": "john@example.com"}}]" → {{"modifications": {{"data": [{{"id": "123", "name": "John", "age": 30, "email": "john@example.com"}}]}}}}
- DO NOT extract schema_id from examples or default values - only from the actual user request
- REMEMBER: If the data has 3 fields, extract ALL 3. If it has 10 fields, extract ALL 10. Never drop any fields.

CRITICAL RULES FOR SCHEMA CREATION:
- If user says "create table" or "create schema", DO NOT use the entire prompt as the name field
- Extract a meaningful name from the prompt OR leave name field out (use default)
- For attributes: extract as array of objects with name, type (string/number/boolean), and required (usually true)
- For universeid/universe ID: extract and put in "universes" array as ["ID_VALUE"]
- For primaryKey: extract as array of field names like ["id", "name"]
- NEVER use the entire user request as the name field value
- NEVER extract schema_id for data ingestion unless explicitly mentioned in user request

IMPORTANT: Return ONLY the JSON object. Do not include any explanation, markdown, or additional text. Start with {{ and end with }}.

Now extract from the USER REQUEST above:"""
        
        return prompt
    
    def should_use_defaults(self, user_prompt: str) -> bool:
        """
        Determine if user wants to use default values (no modifications)
        
        Args:
            user_prompt: User's request
            
        Returns:
            True if should use defaults, False otherwise
        """
        default_keywords = [
            "just", "default", "normal", "regular", "standard",
            "generate token", "get token", "login", "authenticate"
        ]
        
        prompt_lower = user_prompt.lower()
        
        # Check for modification keywords
        modification_keywords = [
            "change", "modify", "update", "set", "use", "with", "as"
        ]
        
        has_modification = any(keyword in prompt_lower for keyword in modification_keywords)
        has_default_intent = any(keyword in prompt_lower for keyword in default_keywords)
        
        # If has modification keywords, don't use defaults
        if has_modification and not has_default_intent:
            return False
        
        # If only default keywords, use defaults
        if has_default_intent and not has_modification:
            return True
        
        # Default: parse to be sure
        return False


def create_prompt_parser() -> OllamaPromptParser:
    """Factory function to create prompt parser"""
    return OllamaPromptParser()
