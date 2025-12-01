#!/usr/bin/env python3
"""
mongo_inserter.py

Mongo helper for storing canonical tool documents.

Exposes:
  - MongoToolInserter with:
      - tools: pymongo Collection (tools)
      - insert_tool(tool: dict) -> bool  (upsert on tool_id)
      - find_tool(tool_id: str) -> dict|None
  - create_mongo_tool_inserter() factory
"""

import logging
from typing import Any, Dict, Optional

from pymongo import MongoClient, ReturnDocument

logger = logging.getLogger(__name__)


class MongoToolInserter:
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "agentProd"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.tools = self.db["tools"]
        logger.info(f"Connected to MongoDB @{uri}, DB: {db_name}")

        # Ensure an index on tool_id for upsert / lookup performance
        try:
            self.tools.create_index("tool_id", unique=True)
        except Exception as e:
            logger.warning(f"Could not create index on 'tool_id': {e}")

    def insert_tool(self, tool: Dict[str, Any]) -> bool:
        """
        Upsert tool document by tool_id.
        Returns True if success, False otherwise.
        """
        if "tool_id" not in tool:
            logger.error("Tool must contain 'tool_id'")
            return False

        try:
            # Prepare document (do not store Mongo _id passed by user)
            tool_doc = dict(tool)
            tool_doc.pop("_id", None)

            res = self.tools.find_one_and_update(
                {"tool_id": tool_doc["tool_id"]},
                {"$set": tool_doc},
                upsert=True,
                return_document=ReturnDocument.AFTER
            )
            logger.info(f"Upserted tool in Mongo: {tool_doc['tool_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert tool in Mongo: {e}", exc_info=True)
            return False

    def find_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = self.tools.find_one({"tool_id": tool_id})
            if doc:
                doc.pop("_id", None)
            return doc
        except Exception as e:
            logger.error(f"Error finding tool {tool_id}: {e}")
            return None


def create_mongo_tool_inserter(uri: str = "mongodb://localhost:27017", db_name: str = "agentProd") -> MongoToolInserter:
    return MongoToolInserter(uri=uri, db_name=db_name)


# -------------------
# Quick test / demo
# -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    inserter = create_mongo_tool_inserter()
    sample_tool = {
        "tool_id": "token_generation_api",
        "name": "Token Generation API",
        "description": "Generates authentication token for PI Artifacts. Example.",
        "keywords": ["token", "login"],
        "schema": {"url": "https://example.com/login", "method": "POST"}
    }
    ok = inserter.insert_tool(sample_tool)
    print("Inserted tool into Mongo:", ok)
    print("Read back:", inserter.find_tool("token_generation_api"))





















# #!/usr/bin/env python3
# """
# mongo_inserter.py
# -----------------
# Stores ONLY tool metadata into MongoDB.
# Does NOT store embeddings here.
# """

# from pymongo import MongoClient
# import logging
# from typing import Dict, Any

# logger = logging.getLogger(__name__)

# class MongoToolInserter:
#     def __init__(self, uri="mongodb://localhost:27017", db_name="agentProd"):
#         self.client = MongoClient(uri)
#         self.db = self.client[db_name]
#         self.tools = self.db["tools"]
#         logger.info("MongoDB connected")

#     def insert_tool(self, tool: Dict[str, Any]) -> bool:
#         try:
#             tool_id = tool.get("tool_id")
#             if not tool_id:
#                 raise ValueError("tool_id missing")

#             existing = self.tools.find_one({"tool_id": tool_id})
#             if existing:
#                 logger.info(f"Updating existing tool {tool_id}")
#                 self.tools.update_one({"tool_id": tool_id}, {"$set": tool})
#             else:
#                 logger.info(f"Inserting new tool {tool_id}")
#                 self.tools.insert_one(tool)

#             return True
#         except Exception as e:
#             logger.error(f"Error inserting tool: {e}")
#             return False


# def create_mongo_tool_inserter():
#     return MongoToolInserter()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     sample_tool = {
#         "tool_id": "token_generation_api",
#         "name": "Token Generation API",
#         "description": "Creates token",
#         "schema": {},
#         "keywords": ["login", "token"],
#         "example_prompts": ["generate token"]
#     }

#     inserter = create_mongo_tool_inserter()
#     inserter.insert_tool(sample_tool)
