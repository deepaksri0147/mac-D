#!/usr/bin/env python3
"""
Ollama Embedding Service
Generates embeddings for semantic search using Ollama
"""

import os
import logging
import requests
from typing import List

logger = logging.getLogger(__name__)


class OllamaEmbeddingService:
    """Generate embeddings using Ollama's embedding models"""
    
    def __init__(
        self, 
        base_url: str = None,
        model: str = None
    ):
        """
        Initialize Ollama embedding service
        
        Args:
            base_url: Ollama server URL
            model: Embedding model name (e.g., nomic-embed-text)
        """
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", 
            "http://ollama-keda.mobiusdtaas.ai"
        )
        self.model = model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", 
            "nomic-embed-text"
        )
        self.api_url = f"{self.base_url}/api/embeddings"
        
        logger.info(f"Ollama Embedding Service initialized")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Model: {self.model}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise Exception(f"Failed to generate embedding: {response.status_code}")
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                raise Exception("Empty embedding returned from Ollama")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def create_tool_embedding(self, tool: dict) -> List[float]:
        """
        Create an embedding for a tool by combining its key fields
        
        Args:
            tool: Tool dictionary with name, description, keywords, etc.
            
        Returns:
            Embedding vector
        """
        # Combine relevant fields for better semantic search
        embedding_text_parts = [
            tool.get("name", ""),
            tool.get("description", ""),
            " ".join(tool.get("keywords", [])),
            " ".join(tool.get("example_prompts", []))
        ]
        
        embedding_text = " ".join(filter(None, embedding_text_parts))
        
        logger.debug(f"Creating embedding for: {embedding_text[:100]}...")
        return self.create_embedding(embedding_text)
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama server
        
        Returns:
            True if connection successful
        """
        try:
            test_embedding = self.create_embedding("test")
            logger.info(f"✓ Ollama connection successful (embedding dim: {len(test_embedding)})")
            return True
        except Exception as e:
            logger.error(f"✗ Ollama connection failed: {e}")
            return False


def create_ollama_embedding_service(
    base_url: str = None,
    model: str = None
) -> OllamaEmbeddingService:
    """
    Factory function to create OllamaEmbeddingService
    
    Args:
        base_url: Ollama server URL (defaults to env or default server)
        model: Embedding model name (defaults to env or nomic-embed-text)
    """
    return OllamaEmbeddingService(base_url=base_url, model=model)


# Test the service if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Ollama Embedding Service...")
    service = create_ollama_embedding_service()
    
    if service.test_connection():
        print("✅ Service is working!")
        
        # Test with sample text
        sample = "generate authentication token"
        embedding = service.create_embedding(sample)
        print(f"\nSample embedding for: '{sample}'")
        print(f"Dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    else:
        print("❌ Service test failed!")
