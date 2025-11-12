#!/usr/bin/env python3
"""
API Executor - Makes HTTP requests with modified parameters
Handles token generation and injection for dependent APIs
"""

import logging
import requests
import json
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class APIExecutor:
    """Execute API calls with parameter modifications"""
    
    def __init__(self):
        self.token_cache = {}  # Cache tokens for reuse
    
    def execute_api(
        self, 
        tool_info: Dict[str, Any],
        parameter_modifications: Dict[str, Any] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an API call with optional parameter modifications
        
        Args:
            tool_info: Tool configuration from Neo4j
            parameter_modifications: Dictionary of fields to modify
            auth_token: Optional authentication token for APIs that require it
            
        Returns:
            Dictionary with status, response data, and any generated token
        """
        
        try:
            # Extract API details
            url = tool_info["url"]
            method = tool_info.get("method", "POST")
            headers = deepcopy(tool_info.get("headers", {}))
            request_body = deepcopy(tool_info.get("requestBody", {}))
            
            # Apply parameter modifications
            if parameter_modifications:
                # Parse JSON strings in modifications (LLM might return JSON as strings)
                parsed_modifications = self._parse_json_strings(parameter_modifications)
                
                # Handle URL parameters (like {schema_id}) - extract and replace in URL
                url_parameters = {}
                if "schema_id" in parsed_modifications:
                    url_parameters["schema_id"] = parsed_modifications.pop("schema_id")
                elif "schemaId" in parsed_modifications:
                    url_parameters["schema_id"] = parsed_modifications.pop("schemaId")
                
                # Replace URL parameters in the URL template
                if url_parameters:
                    for param_name, param_value in url_parameters.items():
                        url = url.replace(f"{{{param_name}}}", str(param_value))
                        logger.info(f"✓ Replaced URL parameter {{{param_name}}} with: {param_value}")
                
                # Deep merge for nested structures (like attributes array)
                request_body = self._deep_merge(request_body, parsed_modifications)
                logger.info(f"✓ Applied modifications: {parsed_modifications}")
            
            # Add authentication token if required
            if tool_info.get("authentication_required", False):
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                    logger.info("✓ Added authentication token to headers")
                else:
                    logger.warning("⚠ API requires authentication but no token provided")
            
            # Log request details
            logger.info(f"🚀 Executing {method} request to: {url}")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Body: {json.dumps(request_body, indent=2)}")
            
            # Log request body for debugging (especially for errors)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Request body: {json.dumps(request_body, indent=2)}")
            
            # Make API call
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=request_body, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=request_body, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=request_body, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, json=request_body, timeout=30)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}"
                }
            
            # Parse response
            response_data = self._parse_response(response)
            
            # Extract token if this API returns one
            extracted_token = None
            if tool_info.get("returns_token", False):
                token_field = tool_info.get("token_field", "token")
                extracted_token = self._extract_token(response_data, token_field)
                if extracted_token:
                    logger.info(f"✓ Token extracted from response: {extracted_token[:20]}...")
            
            # Return result
            result = {
                "success": response.status_code in [200, 201],
                "status_code": response.status_code,
                "response": response_data,
                "token": extracted_token,
                "url": url,
                "method": method
            }
            
            if result["success"]:
                logger.info(f"✅ API call successful (status: {response.status_code})")
            else:
                logger.error(f"❌ API call failed (status: {response.status_code})")
                logger.error(f"Response: {json.dumps(response_data, indent=2)}")
                logger.error(f"Request body was: {json.dumps(request_body, indent=2)}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("❌ API request timed out")
            return {
                "success": False,
                "error": "Request timed out after 30 seconds"
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_response(self, response: requests.Response) -> Any:
        """Parse API response based on content type"""
        try:
            # Try to parse as JSON
            return response.json()
        except json.JSONDecodeError:
            # Return text if not JSON
            return response.text
    
    def _extract_token(self, response_data: Any, token_field: str) -> Optional[str]:
        """
        Extract token from response data
        
        Args:
            response_data: Parsed response data
            token_field: Field name containing the token
            
        Returns:
            Extracted token or None
        """
        if isinstance(response_data, dict):
            # Direct field access
            if token_field in response_data:
                return response_data[token_field]
            
            # Nested access (e.g., "data.token")
            if "." in token_field:
                parts = token_field.split(".")
                current = response_data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return None
                return current if isinstance(current, str) else None
            
            # Common token field names
            common_fields = ["token", "access_token", "accessToken", "authToken", "bearer"]
            for field in common_fields:
                if field in response_data:
                    return response_data[field]
        
        return None
    
    def cache_token(self, tool_id: str, token: str):
        """Cache a token for later use"""
        self.token_cache[tool_id] = token
        logger.info(f"✓ Token cached for tool: {tool_id}")
    
    def get_cached_token(self, tool_id: str) -> Optional[str]:
        """Retrieve cached token"""
        return self.token_cache.get(tool_id)
    
    def _parse_json_strings(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON strings in modifications dictionary and normalize attribute types
        
        Args:
            modifications: Dictionary that may contain JSON strings
            
        Returns:
            Dictionary with parsed JSON values and normalized types
        """
        parsed = {}
        for key, value in modifications.items():
            if isinstance(value, str):
                # Try to parse as JSON if it looks like JSON
                value_stripped = value.strip()
                if (value_stripped.startswith('[') or value_stripped.startswith('{')):
                    try:
                        parsed[key] = json.loads(value)
                        logger.debug(f"  Parsed JSON string for field '{key}'")
                    except json.JSONDecodeError:
                        # Not valid JSON, keep as string
                        parsed[key] = value
                else:
                    parsed[key] = value
            elif isinstance(value, dict):
                # Recursively parse nested dictionaries
                parsed[key] = self._parse_json_strings(value)
            elif isinstance(value, list):
                # Parse list items if they are JSON strings
                parsed_list = []
                for item in value:
                    if isinstance(item, str) and item.strip().startswith(('{', '[')):
                        try:
                            parsed_list.append(json.loads(item))
                        except json.JSONDecodeError:
                            parsed_list.append(item)
                    else:
                        parsed_list.append(item)
                parsed[key] = parsed_list
            else:
                parsed[key] = value
        
        # Normalize attribute types if this is the attributes field
        if "attributes" in parsed and isinstance(parsed["attributes"], list):
            parsed["attributes"] = self._normalize_attribute_types(parsed["attributes"])
        
        return parsed
    
    def _normalize_attribute_types(self, attributes: list) -> list:
        """
        Normalize attribute type format from simple string to nested object format
        
        Converts: {"type": "string"} → {"type": {"type": "string"}}
        
        Args:
            attributes: List of attribute dictionaries
            
        Returns:
            List with normalized type formats
        """
        normalized = []
        for attr in attributes:
            if isinstance(attr, dict) and "type" in attr:
                type_value = attr["type"]
                # If type is a string, convert to nested format
                if isinstance(type_value, str):
                    # Create a copy to avoid modifying the original
                    attr_normalized = deepcopy(attr)
                    attr_normalized["type"] = {"type": type_value}
                    logger.debug(f"  Normalized attribute type: {type_value} → {{\"type\": \"{type_value}\"}}")
                    normalized.append(attr_normalized)
                else:
                    # Type is already an object or other format, keep as is
                    normalized.append(attr)
            else:
                # Not a valid attribute dict, keep as is
                normalized.append(attr)
        return normalized
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with updates taking precedence
        
        Args:
            base: Base dictionary
            updates: Dictionary with updates
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in updates.items():
            # Skip invalid keys (like "attributes[0].type" which are LLM extraction errors)
            if "[" in key and "]" in key:
                logger.warning(f"  ⚠ Skipping invalid key format: {key} (likely LLM extraction error)")
                continue
            
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # Replace list entirely (for arrays like attributes, primaryKey, universes)
                result[key] = value
                logger.debug(f"  Replaced array '{key}' with new value")
            else:
                # Replace or add the value
                result[key] = value
        
        return result


def create_api_executor() -> APIExecutor:
    """Factory function to create API executor"""
    return APIExecutor()
