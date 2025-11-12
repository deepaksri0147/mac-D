#!/usr/bin/env python3
"""
API Tool Template - Copy and modify this for your 28 APIs

This template shows the complete structure with all fields explained.
Fill in your API details and add to neo4j_tool_structure.py
"""

# ============================================================================
# TEMPLATE - Copy and modify for each new API
# ============================================================================

NEW_API_TEMPLATE = {
    # ========== BASIC INFORMATION ==========
    "tool_id": "unique_api_identifier",  # REQUIRED: Unique ID (use snake_case)
    "name": "Human Readable API Name",    # REQUIRED: Display name
    "description": "Detailed description of what this API does. Include purpose, use cases, and important notes.",  # REQUIRED
    
    # ========== SEARCH KEYWORDS ==========
    "keywords": [
        "keyword1",      # Terms users might search for
        "keyword2",      # Include synonyms
        "action_verb",   # Like "create", "update", "delete"
        "domain_term"    # Domain-specific terms
    ],
    
    # ========== API SCHEMA ==========
    "schema": {
        # --- Connection Details ---
        "url": "https://api.example.com/v1/endpoint",  # REQUIRED: Full API URL
        "method": "POST",  # REQUIRED: GET, POST, PUT, DELETE, PATCH
        
        # --- Headers ---
        "headers": {
            "Content-Type": "application/json",
            # Add "Authorization": "Bearer {token}" if authentication_required=True
        },
        
        # --- Request Body (default values) ---
        "requestBody": {
            "fieldName1": "defaultValue1",
            "fieldName2": 123,
            "nestedField": {
                "subField": "value"
            },
            "arrayField": ["item1", "item2"]
        },
        
        # --- Field Descriptions (CRITICAL for LLM) ---
        "field_descriptions": {
            "fieldName1": "Clear description of this field. Include data type, valid values, format requirements, and examples.",
            "fieldName2": "Numeric field. Explanation of what this number represents. Range: 0-100.",
            "nestedField.subField": "Use dot notation for nested fields.",
            "arrayField": "Array of strings. Each item represents X. Maximum 10 items."
        },
        
        # --- Required Fields ---
        "required_fields": [
            "fieldName1",  # List all mandatory fields
            "fieldName2"
        ],
        
        # --- Authentication ---
        "authentication_required": False,  # True if needs Bearer token
        
        # --- Token Generation ---
        "returns_token": False,  # True if this API returns an auth token
        "token_field": "token"   # Field name in response containing the token
                                 # Can be nested: "data.access_token"
    },
    
    # ========== DEPENDENCIES ==========
    "dependencies": [
        # "token_generation_api",  # Uncomment if needs authentication
        # "other_api_tool_id"      # Add if depends on other APIs
    ],
    
    # ========== EXAMPLE PROMPTS ==========
    "example_prompts": [
        "natural language request 1",
        "how user might ask for this",
        "alternative phrasing",
        "with specific parameters mentioned"
    ]
}


# ============================================================================
# REAL EXAMPLES - Based on your APIs
# ============================================================================

# Example 1: Token Generation (No dependencies, returns token)
TOKEN_API = {
    "tool_id": "token_generation_api",
    "name": "Token Generation API",
    "description": "Generates authentication token for PI Artifacts. This is the login API that returns a bearer token used for subsequent API calls.",
    "keywords": ["token", "login", "authentication", "authorize", "bearer"],
    "schema": {
        "url": "https://igs.gov-cloud.ai/mobius-iam-service/v1.0/login",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "requestBody": {
            "userName": "aidtaas@gaiansolutions.com",
            "password": "Gaian@123",
            "productId": "c2255be4-ddf6-449e-a1e0-b4f7f9a2b636",
            "requestType": "TENANT"
        },
        "field_descriptions": {
            "userName": "Email address for login. Must be valid email format.",
            "password": "User password. Case-sensitive alphanumeric with special chars.",
            "productId": "UUID of the product. Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "requestType": "Authentication scope. Valid values: TENANT, SUBTENANT, USER. TENANT for tenant-level access, SUBTENANT for sub-tenant, USER for user-level."
        },
        "required_fields": ["userName", "password", "productId", "requestType"],
        "authentication_required": False,
        "returns_token": True,
        "token_field": "token"
    },
    "dependencies": [],
    "example_prompts": [
        "generate token",
        "login and get token",
        "authenticate",
        "use requestType as SUBTENANT",
        "change password to Gaian1234"
    ]
}

# Example 2: API that requires authentication
DATA_FETCH_API = {
    "tool_id": "fetch_data_api",
    "name": "Fetch Data from Schema",
    "description": "Retrieves data from a specific schema/table. Requires authentication token.",
    "keywords": ["fetch", "get", "retrieve", "data", "query", "read"],
    "schema": {
        "url": "https://ig.gov-cloud.ai/pi-entity-service/v1.0/data",
        "method": "GET",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"  # Will be replaced with actual token
        },
        "requestBody": {
            "schemaId": "SCHEMA_ID_PLACEHOLDER",
            "filters": {},
            "limit": 100,
            "offset": 0
        },
        "field_descriptions": {
            "schemaId": "ID of the schema to query. Obtained from create_mongo_schema_api response.",
            "filters": "JSON object with filter conditions. Format: {fieldName: value}",
            "limit": "Maximum number of records to return. Range: 1-1000. Default: 100",
            "offset": "Number of records to skip (for pagination). Default: 0"
        },
        "required_fields": ["schemaId"],
        "authentication_required": True,
        "returns_token": False
    },
    "dependencies": ["token_generation_api"],
    "example_prompts": [
        "fetch data from schema",
        "get records",
        "retrieve data with limit 50",
        "query schema with filters"
    ]
}

# Example 3: Complex API with nested fields
USER_MANAGEMENT_API = {
    "tool_id": "create_user_api",
    "name": "Create User",
    "description": "Creates a new user in the system with specified roles and permissions.",
    "keywords": ["create", "user", "add user", "new user", "register"],
    "schema": {
        "url": "https://ig.gov-cloud.ai/user-service/v1.0/users",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"
        },
        "requestBody": {
            "email": "user@example.com",
            "firstName": "John",
            "lastName": "Doe",
            "role": "VIEWER",
            "permissions": {
                "read": True,
                "write": False,
                "delete": False
            },
            "metadata": {
                "department": "Engineering",
                "employeeId": ""
            }
        },
        "field_descriptions": {
            "email": "User's email address. Must be unique and valid email format.",
            "firstName": "User's first name. 2-50 characters.",
            "lastName": "User's last name. 2-50 characters.",
            "role": "User role. Valid values: ADMIN, EDITOR, VIEWER. ADMIN has full access, EDITOR can modify, VIEWER can only read.",
            "permissions.read": "Boolean. Allow user to read data. Default: true",
            "permissions.write": "Boolean. Allow user to write data. Default: false",
            "permissions.delete": "Boolean. Allow user to delete data. Default: false",
            "metadata.department": "Optional. User's department name.",
            "metadata.employeeId": "Optional. Company employee ID."
        },
        "required_fields": ["email", "firstName", "lastName", "role"],
        "authentication_required": True,
        "returns_token": False
    },
    "dependencies": ["token_generation_api"],
    "example_prompts": [
        "create a new user",
        "add user with email john@example.com",
        "register user as ADMIN",
        "create user with write permissions"
    ]
}


# ============================================================================
# FIELD DESCRIPTION GUIDELINES
# ============================================================================

"""
Good Field Descriptions Should Include:

1. **Purpose**: What this field is used for
2. **Data Type**: String, number, boolean, array, object
3. **Valid Values**: List of acceptable values or ranges
4. **Format**: Expected format (email, UUID, date, etc.)
5. **Constraints**: Min/max length, allowed characters
6. **Default**: What happens if not specified
7. **Examples**: Sample valid values

GOOD EXAMPLES:
✅ "userName: Email address for authentication. Must be valid email format. Example: user@example.com"
✅ "limit: Maximum records to return. Integer between 1-1000. Default: 100"
✅ "status: Order status. Valid values: PENDING, PROCESSING, COMPLETED, CANCELLED"

BAD EXAMPLES:
❌ "userName: username"  (too vague)
❌ "limit: limit"  (not descriptive)
❌ "status: status field"  (no valid values)
"""


# ============================================================================
# QUICK CHECKLIST FOR ADDING NEW API
# ============================================================================

"""
Before inserting a new API, verify:

□ tool_id is unique and descriptive
□ name is human-readable
□ description explains the API's purpose clearly
□ keywords include search terms users might use
□ url is correct and complete
□ method is specified (POST/GET/PUT/DELETE)
□ requestBody has sensible default values
□ field_descriptions explain EVERY field clearly
□ required_fields lists all mandatory parameters
□ authentication_required is set correctly
□ dependencies include token_generation_api if needed
□ example_prompts show various ways to invoke this API
□ returns_token is True only for login/auth APIs
□ token_field matches the response structure

Test after insertion:
□ Semantic search finds this tool
□ LLM can extract modifications
□ API executes successfully
□ Token injection works (if needed)
"""


# ============================================================================
# USAGE
# ============================================================================

"""
1. Copy the template above
2. Fill in all fields for your API
3. Add to neo4j_tool_structure.py or create separate file
4. Insert using:
   
   from insert_tools import insert_tool
   insert_tool(YOUR_NEW_API)
   
5. Test:
   
   from main_orchestrator import create_orchestrator
   orch = create_orchestrator()
   result = orch.process_user_request("your test prompt")
   print(result)
"""
