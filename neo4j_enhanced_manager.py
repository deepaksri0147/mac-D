#!/usr/bin/env python3
"""
Enhanced Neo4j Tool Manager with Field Descriptions Support
"""

import os
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class EnhancedNeo4jToolManager:
    """Manages API tools in Neo4j with embeddings and field descriptions"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = "neo4j"):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.database = database
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def create_indexes(self):
        """Create necessary indexes and constraints"""
        session_kwargs = {"database": self.database} if self.database else {}
        with self.driver.session(**session_kwargs) as session:
            # Vector index for embeddings
            try:
                session.run("""
                    CREATE VECTOR INDEX tool_embeddings IF NOT EXISTS
                    FOR (t:Tool)
                    ON t.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("✓ Vector index created/verified")
            except Exception as e:
                logger.warning(f"Vector index warning: {e}")
            
            # Unique constraint on tool_id
            try:
                session.run("""
                    CREATE CONSTRAINT tool_id_unique IF NOT EXISTS
                    FOR (t:Tool)
                    REQUIRE t.tool_id IS UNIQUE
                """)
                logger.info("✓ Unique constraint on tool_id created")
            except Exception as e:
                logger.warning(f"Constraint warning: {e}")
    
    def insert_tool(self, tool: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Insert or update a tool with its embedding
        
        Args:
            tool: Tool dictionary with all fields including field_descriptions
            embedding: Vector embedding for semantic search
            
        Returns:
            True if successful
        """
        try:
            session_kwargs = {"database": self.database} if self.database else {}
            with self.driver.session(**session_kwargs) as session:
                # Get dependencies list
                dependencies = tool.get("dependencies", [])
                
                # Merge tool node (including dependencies as property)
                result = session.run("""
                    MERGE (t:Tool {tool_id: $tool_id})
                    SET t.name = $name,
                        t.description = $description,
                        t.keywords = $keywords,
                        t.url = $url,
                        t.method = $method,
                        t.headers = $headers,
                        t.requestBody = $requestBody,
                        t.query_parameters = $queryParameters,
                        t.field_descriptions = $field_descriptions,
                        t.required_fields = $required_fields,
                        t.authentication_required = $authentication_required,
                        t.returns_token = $returns_token,
                        t.token_field = $token_field,
                        t.example_prompts = $example_prompts,
                        t.dependencies = $dependencies,
                        t.embedding = $embedding,
                        t.updated_at = datetime()
                    RETURN t.tool_id as tool_id
                """, 
                    tool_id=tool["tool_id"],
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    keywords=tool.get("keywords", []),
                    url=tool["schema"].get("url", ""),
                    method=tool["schema"].get("method", "POST"),
                    headers=str(tool["schema"].get("headers", {})),
                    requestBody=str(tool["schema"].get("requestBody", {})),
                    queryParameters=str(tool["schema"].get("query_parameters", {})),
                    field_descriptions=str(tool["schema"].get("field_descriptions", {})),
                    required_fields=tool["schema"].get("required_fields", []),
                    authentication_required=tool["schema"].get("authentication_required", False),
                    returns_token=tool["schema"].get("returns_token", False),
                    token_field=tool["schema"].get("token_field", "token"),
                    example_prompts=tool.get("example_prompts", []),
                    dependencies=dependencies,  # Store as property
                    embedding=embedding
                )
                
                tool_id = result.single()["tool_id"]
                
                # Create dependency relationships (if dependency tools exist)
                if dependencies:
                    logger.info(f"  Creating dependency relationships for {tool_id}: {dependencies}")
                    for dep_id in dependencies:
                        try:
                            # Check if dependency tool exists first
                            dep_exists = session.run("""
                                MATCH (d:Tool {tool_id: $dep_id})
                                RETURN d.tool_id as dep_id
                            """, dep_id=dep_id).single()
                            
                            if dep_exists:
                                # Create relationship
                                session.run("""
                                    MATCH (t:Tool {tool_id: $tool_id})
                                    MATCH (d:Tool {tool_id: $dep_id})
                                    MERGE (t)-[:DEPENDS_ON]->(d)
                                """, tool_id=tool["tool_id"], dep_id=dep_id)
                                logger.info(f"    ✓ Created dependency: {tool_id} → {dep_id}")
                            else:
                                logger.warning(f"    ⚠ Dependency tool '{dep_id}' not found - relationship will be created when dependency is inserted")
                                # Create relationship anyway (will work when dependency is inserted later)
                                session.run("""
                                    MATCH (t:Tool {tool_id: $tool_id})
                                    OPTIONAL MATCH (d:Tool {tool_id: $dep_id})
                                    WITH t, d
                                    WHERE d IS NOT NULL
                                    MERGE (t)-[:DEPENDS_ON]->(d)
                                """, tool_id=tool["tool_id"], dep_id=dep_id)
                        except Exception as e:
                            logger.warning(f"    ⚠ Error creating dependency relationship {tool_id} → {dep_id}: {e}")
                
                logger.info(f"✓ Tool '{tool_id}' inserted/updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"✗ Error inserting tool: {e}")
            return False
    
    def search_tools_by_embedding(
        self, 
        query_embedding: List[float], 
        limit: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tools using vector similarity
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching tools with similarity scores
        """
        try:
            session_kwargs = {"database": self.database} if self.database else {}
            print(session_kwargs)
            with self.driver.session(**session_kwargs) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('tool_embeddings', $limit, $query_embedding)
                    YIELD node, score
                    WHERE score >= $similarity_threshold
                    RETURN node.tool_id as tool_id,
                           node.name as name,
                           node.description as description,
                           node.url as url,
                           node.method as method,
                           node.headers as headers,
                           node.requestBody as requestBody,
                           node.queryParameters as queryParameters,
                           node.field_descriptions as field_descriptions,
                           node.required_fields as required_fields,
                           node.authentication_required as authentication_required,
                           node.returns_token as returns_token,
                           node.token_field as token_field,
                           score
                    ORDER BY score DESC
                """, 
                    query_embedding=query_embedding, 
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                tools = []
                for record in result:
                    tools.append({
                        "tool_id": record["tool_id"],
                        "name": record["name"],
                        "description": record["description"],
                        "url": record["url"],
                        "method": record["method"],
                        "headers": eval(record["headers"]) if record["headers"] else {},
                        "requestBody": eval(record["requestBody"]) if record["requestBody"] else {},
                        "queryParameters": eval(record["queryParameters"]) if record.get("queryParameters") else {},
                        "field_descriptions": eval(record["field_descriptions"]) if record["field_descriptions"] else {},
                        "required_fields": record["required_fields"],
                        "authentication_required": record["authentication_required"],
                        "returns_token": record["returns_token"],
                        "token_field": record["token_field"],
                        "similarity_score": record["score"]
                    })
                
                return tools
                
        except Exception as e:
            logger.error(f"✗ Error searching tools: {e}")
            return []

    def get_tool_by_id(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single tool by its tool_id.
        """
        try:
            session_kwargs = {"database": self.database} if self.database else {}
            with self.driver.session(**session_kwargs) as session:
                result = session.run("""
                    MATCH (node:Tool {tool_id: $tool_id})
                    RETURN node.tool_id as tool_id,
                           node.name as name,
                           node.description as description,
                           node.url as url,
                           node.method as method,
                           node.headers as headers,
                           node.requestBody as requestBody,
                           node.queryParameters as queryParameters,
                           node.field_descriptions as field_descriptions,
                           node.required_fields as required_fields,
                           node.authentication_required as authentication_required,
                           node.returns_token as returns_token,
                           node.token_field as token_field
                """, tool_id=tool_id)
                
                record = result.single()
                if not record:
                    return None

                return {
                    "tool_id": record["tool_id"],
                    "name": record["name"],
                    "description": record["description"],
                    "url": record["url"],
                    "method": record["method"],
                    "headers": eval(record["headers"]) if record["headers"] else {},
                    "requestBody": eval(record["requestBody"]) if record["requestBody"] else {},
                    "queryParameters": eval(record["queryParameters"]) if record.get("queryParameters") else {},
                    "field_descriptions": eval(record["field_descriptions"]) if record["field_descriptions"] else {},
                    "required_fields": record["required_fields"],
                    "authentication_required": record["authentication_required"],
                    "returns_token": record["returns_token"],
                    "token_field": record["token_field"],
                }
        except Exception as e:
            logger.error(f"✗ Error getting tool by ID '{tool_id}': {e}")
            return None
    
    def get_tool_dependencies(self, tool_id: str) -> List[str]:
        """
        Get list of tool IDs that this tool depends on
        First tries to get from relationships, falls back to property if relationships don't exist
        """
        try:
            session_kwargs = {"database": self.database} if self.database else {}
            with self.driver.session(**session_kwargs) as session:
                # First, try to get from relationships
                result = session.run("""
                    MATCH (t:Tool {tool_id: $tool_id})-[:DEPENDS_ON]->(d:Tool)
                    RETURN d.tool_id as dep_id
                    ORDER BY d.tool_id
                """, tool_id=tool_id)
                
                deps_from_relationships = [record["dep_id"] for record in result]
                
                # If we have relationships, return them
                if deps_from_relationships:
                    return deps_from_relationships
                
                # Fallback: get from node property
                result = session.run("""
                    MATCH (t:Tool {tool_id: $tool_id})
                    RETURN t.dependencies as dependencies
                """, tool_id=tool_id)
                
                record = result.single()
                if record and record.get("dependencies"):
                    deps = record["dependencies"]
                    # If dependencies is stored as string, try to parse it
                    if isinstance(deps, str):
                        try:
                            import ast
                            deps = ast.literal_eval(deps)
                        except:
                            pass
                    return deps if isinstance(deps, list) else []
                
                return []
                
        except Exception as e:
            logger.error(f"✗ Error getting dependencies: {e}")
            return []
    
    def rebuild_dependency_relationships(self) -> bool:
        """
        Rebuild all dependency relationships from node properties
        Useful after inserting all tools to ensure relationships are created
        even if tools were inserted out of order
        
        Returns:
            True if successful
        """
        try:
            session_kwargs = {"database": self.database} if self.database else {}
            with self.driver.session(**session_kwargs) as session:
                # Get all tools with dependencies
                result = session.run("""
                    MATCH (t:Tool)
                    WHERE t.dependencies IS NOT NULL AND size(t.dependencies) > 0
                    RETURN t.tool_id as tool_id, t.dependencies as dependencies
                """)
                
                relationships_created = 0
                for record in result:
                    tool_id = record["tool_id"]
                    dependencies = record["dependencies"]
                    
                    # Handle string dependencies
                    if isinstance(dependencies, str):
                        try:
                            import ast
                            dependencies = ast.literal_eval(dependencies)
                        except:
                            continue
                    
                    if not isinstance(dependencies, list):
                        continue
                    
                    for dep_id in dependencies:
                        try:
                            # Create relationship if dependency tool exists
                            rel_result = session.run("""
                                MATCH (t:Tool {tool_id: $tool_id})
                                MATCH (d:Tool {tool_id: $dep_id})
                                MERGE (t)-[r:DEPENDS_ON]->(d)
                                RETURN r
                            """, tool_id=tool_id, dep_id=dep_id)
                            
                            if rel_result.single():
                                relationships_created += 1
                                logger.info(f"  ✓ Rebuilt dependency: {tool_id} → {dep_id}")
                        except Exception as e:
                            logger.warning(f"  ⚠ Could not create relationship {tool_id} → {dep_id}: {e}")
                
                logger.info(f"✓ Rebuilt {relationships_created} dependency relationships")
                return True
                
        except Exception as e:
            logger.error(f"✗ Error rebuilding dependency relationships: {e}")
            return False


def create_neo4j_manager(database: str = None) -> EnhancedNeo4jToolManager:
    """
    Factory function to create Neo4j manager
    
    Args:
        database: Optional database name (e.g., "piagent", "runrun")
                  If None, uses default database
    """
    return EnhancedNeo4jToolManager(database=database)
