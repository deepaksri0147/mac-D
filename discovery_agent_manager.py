#!/usr/bin/env python3
"""
Discovery Agent Manager - Manages the discovery agent database
This database stores agent information and helps route user requests to the right agent
"""

import os
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class DiscoveryAgentManager:
    """Manages agents in the discovery agent database"""
    
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
        with self.driver.session(database=self.database) as session:
            # Vector index for agent embeddings
            try:
                session.run("""
                    CREATE VECTOR INDEX agent_embeddings IF NOT EXISTS
                    FOR (a:Agent)
                    ON a.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("✓ Vector index for agents created/verified")
            except Exception as e:
                logger.warning(f"Vector index warning: {e}")
            
            # Unique constraint on agent_id
            try:
                session.run("""
                    CREATE CONSTRAINT agent_id_unique IF NOT EXISTS
                    FOR (a:Agent)
                    REQUIRE a.agent_id IS UNIQUE
                """)
                logger.info("✓ Unique constraint on agent_id created")
            except Exception as e:
                logger.warning(f"Constraint warning: {e}")
        
        self._migrate_agent_properties() # Call the migration method here
    
    def _migrate_agent_properties(self):
        """
        Ensures all existing Agent nodes have a 'tools' property,
        setting it to an empty list if missing.
        """
        with self.driver.session(database=self.database) as session:
            try:
                session.run("""
                    MATCH (a:Agent)
                    WHERE NOT EXISTS(a.tools)
                    SET a.tools = []
                """)
                logger.info("✓ Migrated existing Agent nodes to include 'tools' property")
            except Exception as e:
                logger.warning(f"Migration warning for 'tools' property: {e}")

    def insert_agent(self, agent: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Insert or update an agent with its embedding
        
        Args:
            agent: Agent dictionary with all fields
            embedding: Vector embedding for semantic search
            
        Returns:
            True if successful
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Merge agent node
                result = session.run("""
                    MERGE (a:Agent {agent_id: $agent_id})
                    SET a.name = $name,
                        a.description = $description,
                        a.keywords = $keywords,
                        a.database_name = $database_name,
                        a.tools = $tools,
                        a.example_prompts = $example_prompts,
                        a.embedding = $embedding,
                        a.updated_at = datetime()
                    RETURN a.agent_id as agent_id
                """,
                    agent_id=agent["agent_id"],
                    name=agent.get("name", ""),
                    description=agent.get("description", ""),
                    keywords=agent.get("keywords", []),
                    database_name=agent.get("database_name", ""),
                    tools=agent.get("tools", []),
                    example_prompts=agent.get("example_prompts", []),
                    embedding=embedding
                )
                
                agent_id = result.single()["agent_id"]
                logger.info(f"✓ Agent '{agent_id}' inserted/updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"✗ Error inserting agent: {e}")
            return False
    
    def search_agents_by_embedding(
        self, 
        query_embedding: List[float], 
        limit: int = 3,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for similar agents using vector similarity
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching agents with similarity scores
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('agent_embeddings', $limit, $query_embedding)
                    YIELD node, score
                    WHERE score >= $similarity_threshold
                    RETURN node.agent_id as agent_id,
                           node.name as name,
                           node.description as description,
                           node.database_name as database_name,
                           node.tools as tools,
                           score
                    ORDER BY score DESC
                """, 
                    query_embedding=query_embedding, 
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                agents = []
                for record in result:
                    agents.append({
                        "agent_id": record["agent_id"],
                        "name": record["name"],
                        "description": record["description"],
                        "database_name": record["database_name"],
                        "tools": record["tools"],
                        "similarity_score": record["score"]
                    })
                
                return agents
                
        except Exception as e:
            logger.error(f"✗ Error searching agents: {e}")
            return []
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information by agent_id
        
        Args:
            agent_id: Agent ID to search for
            
        Returns:
            Agent information or None
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (a:Agent {agent_id: $agent_id})
                    RETURN a.agent_id as agent_id,
                           a.name as name,
                           a.description as description,
                           a.database_name as database_name,
                           a.tools as tools
                """, agent_id=agent_id)
                
                record = result.single()
                if record:
                    return {
                        "agent_id": record["agent_id"],
                        "name": record["name"],
                        "description": record["description"],
                        "database_name": record["database_name"],
                        "tools": record["tools"]
                    }
                return None
                
        except Exception as e:
            logger.error(f"✗ Error getting agent: {e}")
            return None


def create_discovery_agent_manager() -> DiscoveryAgentManager:
    """Factory function to create discovery agent manager"""
    return DiscoveryAgentManager()

