#!/usr/bin/env python3
"""
Enhanced Tool Storage Structure for Neo4j with Field Descriptions
"""

# Enhanced Tool Schema with Field Descriptions
TOKEN_GENERATION_TOOL_ENHANCED = {
    "tool_id": "token_generation_api",
    "name": "Token Generation API",
    "description": "Generates authentication token for PI Artifacts. This is the login API that returns a bearer token used for subsequent API calls.",
    "keywords": [
        "token", "login", "authentication", "authorization", "bearer token"
    ],
    "schema": {
        "url": "https://igs.gov-cloud.ai/mobius-iam-service/v1.0/login",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json"
        },
        "requestBody": {
            "userName": "aidtaas@gaiansolutions.com",
            "password": "Gaian@123",
            "productId": "c2255be4-ddf6-449e-a1e0-b4f7f9a2b636",
            "requestType": "TENANT"
        },
        "field_descriptions": {
            "userName": "Email address of the user attempting to login. Must be a valid email format.",
            "password": "User's password for authentication. Case-sensitive alphanumeric string.",
            "productId": "UUID of the product/application being accessed. Must be a valid UUID format.",
            "requestType": "Type of authentication request. Valid values: TENANT, SUBTENANT, USER. Determines the scope of access."
        },
        "required_fields": ["userName", "password", "productId", "requestType"],
        "authentication_required": False,
        "returns_token": True,
        "token_field": "token"  # Field name in response that contains the token
    },
    "dependencies": [],
    "example_prompts": [
        "generate token",
        "login to the system",
        "get authentication token",
        "use requestType as SUBTENANT",
        "change password to Gaian1234"
    ]
}

DATAVERSE_CREATION_TOOL_ENHANCED = {
    "tool_id": "create_dataverse_api",
    "name": "Create Dataverse/Universe",
    "description": "Creates a new dataverse (universe) in PI. A dataverse is a logical container for organizing schemas and data.",
    "keywords": [
        "dataverse", "universe", "create universe", "create dataverse"
    ],
    "schema": {
        "url": "https://ig.gov-cloud.ai/pi-dataverse-service/v1.0/dataverses",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"  # Will be replaced dynamically
        },
        "requestBody": {
            "name": "Default Dataverse",
            "description": "Default dataverse description",
            "icon": "iconURL",
            "tags": {
                "RED": ["Universe"]
            },
            "universeReadAccess": "PRIVATE",
            "universeWriteAccess": "PRIVATE",
            "visibility": "PRIVATE"
        },
        "field_descriptions": {
            "name": "Name of the dataverse. Must be unique within the tenant.",
            "description": "Detailed description of the dataverse purpose and contents.",
            "icon": "URL to an icon image representing this dataverse.",
            "tags": "Categorization tags for the dataverse. Format: {color: [tag_list]}",
            "universeReadAccess": "Read access level. Valid values: PUBLIC, PRIVATE, RESTRICTED",
            "universeWriteAccess": "Write access level. Valid values: PUBLIC, PRIVATE, RESTRICTED",
            "visibility": "Overall visibility of the dataverse. Valid values: PUBLIC, PRIVATE"
        },
        "required_fields": ["name", "description"],
        "authentication_required": True,
        "returns_token": False
    },
    "dependencies": ["token_generation_api"],
    "example_prompts": [
        "create a dataverse named XXM Factory",
        "create universe for testing",
        "make a new dataverse"
    ]
}

SCHEMA_CREATION_TOOL_ENHANCED = {
    "tool_id": "create_mongo_schema_api",
    "name": "MongoDB Schema Creation",
    "description": "Creates a new entity schema/table in MongoDB. Defines the structure and attributes for storing data in PI.",
    "keywords": [
        "schema", "table", "mongo", "mongodb", "create schema", "entity"
    ],
    "schema": {
        "url": "https://ig.gov-cloud.ai/pi-entity-service-dbaas/v1.0/schemas",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"
        },
        "requestBody": {
            "name": "Default Schema",
            "description": "Default schema description",
            "universes": ["UNIVERSE_ID_PLACEHOLDER"],
            "tags": {
                "BLUE": ["SCHEMA"]
            },
            "primaryDb": "MONGO",
            "piFeatures": {
                "COHORTS": {"COHORTS": ["MONGO"]},
                "CONTEXT": {"CONTEXT": ["MONGO"]},
                "BIGQUERY": {"BIGQUERY": ["MONGO"]}
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
        },
        "field_descriptions": {
            "name": "Name of the schema/table. Must be unique within the universe.",
            "description": "Description of what data this schema stores.",
            "universes": "Array of universe IDs where this schema should be created. Use dataverse ID from create_dataverse_api.",
            "tags": "Categorization tags. Format: {color: [tag_list]}",
            "primaryDb": "Primary database type. Valid values: MONGO, POSTGRES, MYSQL",
            "piFeatures": "PI features that can access this schema. Maps features to database types.",
            "attributes": "Array of column/field definitions. Each attribute object must have: name (string), type (object with 'type' field, e.g., {\"type\": \"string\"} or {\"type\": \"number\"}), and required (boolean). Example: [{\"name\": \"id\", \"type\": {\"type\": \"string\"}, \"required\": true}]",
            "primaryKey": "Array of field names that form the primary key. Must exist in attributes.",
            "dataReadAccess": "Who can read data. Values: PUBLIC, PRIVATE, RESTRICTED",
            "dataWriteAccess": "Who can write data. Values: PUBLIC, PRIVATE, RESTRICTED",
            "metadataReadAccess": "Who can read schema metadata. Values: PUBLIC, PRIVATE, RESTRICTED",
            "metadataWriteAccess": "Who can modify schema metadata. Values: PUBLIC, PRIVATE, RESTRICTED",
            "visibility": "Overall schema visibility. Values: PUBLIC, PRIVATE"
        },
        "required_fields": ["name", "description", "universes", "tags", "primaryDb", "piFeatures", "attributes", "primaryKey"],
        "authentication_required": True,
        "returns_token": False
    },
    "dependencies": ["token_generation_api", "create_dataverse_api"],
    "example_prompts": [
        "create a schema for driver data",
        "create mongodb table",
        "define new schema with user fields"
    ]
}


DATA_INGESTION_TOOL_ENHANCED = {
    "tool_id": "ingest_data_api",
    "name": "Ingest Data into Schema",
    "description": "Ingests/inserts data records into a MongoDB schema. Allows bulk insertion of multiple data instances into an existing schema/table.",
    "keywords": [
        "ingest", "insert", "data", "instances", "bulk insert", "add data", "populate schema"
    ],
    "schema": {
        "url": "https://ig.gov-cloud.ai/pi-entity-instances-service/v2.0/schemas/{schema_id}/instances",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"
        },
        "requestBody": {
            "data": [
                {
                    "id": "123",
                    "name": "Deepak"
                }
            ]
        },
        "field_descriptions": {
            "data": "Array of data objects to be inserted. Each object represents one record/instance to be added to the schema.",
            "schema_id": "The unique identifier (ID) of the schema where data will be ingested. This should be obtained from the schema creation API response. Example: '67dcf66d5ccb2c54260fb156'",
            "id": "Unique identifier for the data record. Must be unique within the schema.",
            "name": "Name of the data record."
        },
        "required_fields": ["data"],
        "authentication_required": True,
        "returns_token": False,
        "url_parameters": {
            "schema_id": "Required path parameter - ID of the target schema for data ingestion. If not provided in prompt, a new schema will be created automatically from the data structure."
        }
    },
    "dependencies": ["token_generation_api", "create_mongo_schema_api"],
    "example_prompts": [
        "ingest data into schema",
        "add records to the schema",
        "insert data into schema 67dcf66d5ccb2c54260fb156",
        "bulk insert data with id and name fields",
        "populate the schema with data: [{\"id\": \"123\", \"name\": \"John\"}]",
        "add new instances to the table",
        "ingest data: [{\"id\": \"1\", \"name\": \"Test\", \"age\": 30}]"
    ]
}



# All tools list for batch insertion
ALL_TOOLS_ENHANCED = [
    TOKEN_GENERATION_TOOL_ENHANCED,
    DATAVERSE_CREATION_TOOL_ENHANCED,
    SCHEMA_CREATION_TOOL_ENHANCED,
    DATA_INGESTION_TOOL_ENHANCED
]
