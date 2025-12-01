import chromadb
from chromadb.config import Settings
import json

# Tool definitions
DATAVERSE_CREATION_TOOL_ENHANCED = {
    "tool_id": "create_dataverse_api",
    "name": "Create Dataverse/Universe",
    "agent_id": "pi_agent",
    "description": "Creates a new dataverse (universe) in PI. A dataverse is a logical container for organizing schemas and data.",
    "keywords": [
        "dataverse", "universe", "create universe", "create dataverse"
    ],
    "schema": {
        "url": "https://ig.gov-cloud.ai/pi-dataverse-service/v1.0/dataverses",
        "method": "POST",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {token}"
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
    "agent_id": "pi_agent",
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
            "attributes": "Array of column/field definitions. Each attribute object must have: name (string), type (object with 'type' field), and required (boolean).",
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

# Function to create searchable text from tool
def create_tool_document(tool):
    """Create a comprehensive text document from tool data for better retrieval"""
    doc = f"""
Tool ID: {tool['tool_id']}
Name: {tool['name']}
Description: {tool['description']}
Keywords: {', '.join(tool['keywords'])}
Example Prompts: {', '.join(tool['example_prompts'])}

Primary Database: {tool['schema']['requestBody'].get('primaryDb', 'N/A')}
Method: {tool['schema']['method']}
URL: {tool['schema']['url']}

Field Descriptions:
"""
    for field, desc in tool['schema']['field_descriptions'].items():
        doc += f"- {field}: {desc}\n"
    
    return doc.strip()

# Connect to ChromaDB Docker instance with Ollama embeddings
chroma_client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(
        allow_reset=True,
        anonymized_telemetry=False
    )
)

# Create or get collection with Ollama embedding function
# Replace 'nomic-embed-text' with your preferred Ollama embedding model
collection = chroma_client.get_or_create_collection(
    name="tools_collection",
    embedding_function=chromadb.utils.embedding_functions.OllamaEmbeddingFunction(
        url="http://ollama-keda.mobiusdtaas.ai//api/embeddings",
        model_name="nomic-embed-text"  # Change this to your Ollama model
    ),
    metadata={"description": "PI Agent tools for dataverse and schema creation"}
)

# Prepare documents
tools = [DATAVERSE_CREATION_TOOL_ENHANCED, SCHEMA_CREATION_TOOL_ENHANCED]
documents = [create_tool_document(tool) for tool in tools]
metadatas = [
    {
        "tool_id": tool["tool_id"],
        "name": tool["name"],
        "agent_id": tool["agent_id"]
    }
    for tool in tools
]
ids = [tool["tool_id"] for tool in tools]

# Insert data into ChromaDB (upsert to avoid duplicates)
collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("✓ Tools inserted into ChromaDB successfully!\n")

# Test query
test_query = """create table in mongo with name as "mongod" and description as the 
"table in mongo" with attributes modelname as string and modelid as number 
put primary key as the modelid and with universeid as 6911eb8a68829872bc56d02b 
with tags as "BLUE" """

results = collection.query(
    query_texts=[test_query],
    n_results=1
)

print("Query:", test_query)
print("\n" + "="*80)
print("Results:")
print("="*80)
print(f"\nTool ID: {results['metadatas'][0][0]['tool_id']}")
print(f"Tool Name: {results['metadatas'][0][0]['name']}")
print(f"Distance: {results['distances'][0][0]:.4f}")
print(f"\nMatched Document Preview:\n{results['documents'][0][0][:300]}...")

# Additional test queries
print("\n" + "="*80)
print("Additional Test Queries:")
print("="*80)

test_queries = [
    "create a new universe for my project",
    "I need to create a schema with user information",
    "make a dataverse called TestUniverse"
]

for query in test_queries:
    result = collection.query(query_texts=[query], n_results=1)
    tool_id = result['metadatas'][0][0]['tool_id']
    print(f"\nQuery: {query}")
    print(f"→ Tool ID: {tool_id}")




































# import chromadb

# chroma_client = chromadb.Client()

# # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
# collection = chroma_client.get_or_create_collection(name="my_collection")


# student_info = """
# Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
# is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
# in her free time in hopes of working at a tech company after graduating from the University of Washington.
# """

# club_info = """
# The university chess club provides an outlet for students to come together and enjoy playing
# the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
# the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
# participate in tournaments, analyze famous chess matches, and improve members' skills.
# """

# university_info = """
# The University of Washington, founded in 1861 in Seattle, is a public research university
# with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
# As the flagship institution of the six public universities in Washington state,
# UW encompasses over 500 buildings and 20 million square feet of space,
# including one of the largest library systems in the world.
# """



# # switch `add` to `upsert` to avoid adding the same documents every time
# collection.add(
#     documents = [student_info, club_info, university_info],
#     metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
#     ids = ["id1", "id2", "id3"]
# )

# results = collection.query(
#     query_texts=["What is the student name?"],
#     n_results=2
# )



# print(results)
