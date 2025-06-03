from crewai_tools import RagTool

# Create a RAG tool with custom configuration
config = {
    "app": {
        "name": "custom_app",
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    },
    "vectordb": {
        "provider": "elasticsearch",
        "config": {
            "collection_name": "my-collection",
            "cloud_id": "deployment-name:xxxx",
            "api_key": "your-key",
            "verify_certs": False
        }
    },
    "chunker": {
        "chunk_size": 400,
        "chunk_overlap": 100,
        "length_function": "len",
        "min_chunk_size": 0
    }
}

rag_tool = RagTool(config=config, summarize=True)
