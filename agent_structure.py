#!/usr/bin/env python3
"""
Agent Structure - Defines agents and their tool mappings
Each agent has its own Neo4j database containing its tools
"""

# PI Agent - Handles PI (Platform Intelligence) related operations
PI_AGENT = {
    "agent_id": "pi_agent",
    "name": "PI Agent",
    "description": "Platform Intelligence Agent that handles dataverse creation, schema management, and data ingestion operations in the PI platform. This agent manages the complete data lifecycle from creating logical containers (dataverses) to defining data structures (schemas) and ingesting data records.",
    "keywords": [
        "pi", "platform intelligence", "dataverse", "universe", "schema", "mongo", "mongodb",
        "data ingestion", "create dataverse", "create schema", "ingest data", "insert data",
        "token generation", "authentication", "login"
    ],
    "database_name": "piagent",  # Neo4j database name for this agent
    "tools": [
        "token_generation_api",
        "create_dataverse_api",
        "create_mongo_schema_api",
        "ingest_data_api"
    ],
    "example_prompts": [
        "generate token",
        "create a dataverse",
        "create a schema",
        "ingest data into schema",
        "create universe for testing",
        "insert data into mongo schema",
        "login to the system"
    ]
}

# RunRun Agent - Handles security and vulnerability scanning operations
RUNRUN_AGENT = {
    "agent_id": "runrun_agent",
    "name": "RunRun Agent",
    "description": "RunRun Agent specializes in security operations including vulnerability scanning, security checks, and DevSecOps workflows. This agent executes security scans on servers and deployments using Camunda workflow engine, providing comprehensive security analysis and vulnerability detection.",
    "keywords": [
        "runrun", "vulnerability", "security", "scan", "check", "devsecops",
        "vulnerability scan", "security check", "server security", "security analysis",
        "vulnerability test", "scan deployment", "runrun token", "runrun login"
    ],
    "database_name": "runrun",  # Neo4j database name for this agent
    "tools": [
        "runrun_token_api",
        "vulnerability_check_api"
    ],
    "example_prompts": [
        "run vulnerability check",
        "scan for security vulnerabilities",
        "execute vulnerability test",
        "check server for vulnerabilities",
        "perform security scan",
        "run devsecops security check",
        "generate runrun token",
        "get runrun token"
    ]
}

# All agents list
ALL_AGENTS = [
    PI_AGENT,
    RUNRUN_AGENT
]

