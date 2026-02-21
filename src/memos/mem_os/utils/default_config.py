"""
Default configuration utilities for MemOS.
Provides simplified configuration generation for users.
"""

import os
from typing import Literal

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube


def get_default_config(
    openai_api_key: str,
    openai_api_base: str = "https://api.openai.com/v1",
    text_mem_type: Literal["tree_text", "general_text"] = "general_text",
    user_id: str = "default_user",
    **kwargs,
) -> MOSConfig:
    """
    Generate a default MOS configuration with minimal user input.

    Args:
        openai_api_key (str): OpenAI API key
        openai_api_base (str): OpenAI API base URL, defaults to "https://api.openai.com/v1"
        text_mem_type (str): Type of text memory, either "tree_text" or "general_text"
        user_id (str): User ID for the configuration
        **kwargs: Additional configuration overrides

    Returns:
        MOSConfig: Complete MOS configuration object

    Example:
        ```python
        config = get_default_config(
            openai_api_key="sk-...",
            openai_api_base="https://api.openai.com/v1",
            text_mem_type="general_text"
        )
        mos = MOS(config)
        ```
    """

    # Base OpenAI configuration
    openai_config = {
        "model_name_or_path": kwargs.get("model_name", "gpt-4o-mini"),
        "temperature": kwargs.get("temperature", 0.8),
        "max_tokens": kwargs.get("max_tokens", 1024),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 50),
        "remove_think_prefix": True,
        "api_key": openai_api_key,
        "api_base": openai_api_base,
    }

    # Embedder configuration — supports ollama or universal_api backends
    embedder_backend = kwargs.get("embedder_backend", os.getenv("MOS_EMBEDDER_BACKEND", "universal_api"))
    if embedder_backend == "ollama":
        embedder_config = {
            "backend": "ollama",
            "config": {
                "model_name_or_path": kwargs.get("embedder_model", "bge-m3"),
                "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            },
        }
    else:
        embedder_config = {
            "backend": "universal_api",
            "config": {
                "provider": "openai",
                "api_key": openai_api_key,
                "model_name_or_path": kwargs.get("embedder_model", "text-embedding-3-large"),
                "base_url": openai_api_base,
            },
        }

    # Base configuration
    config_dict = {
        "user_id": user_id,
        "chat_model": {
            "backend": "openai",
            "config": openai_config,
        },
        "mem_reader": {
            "backend": "simple_struct",
            "config": {
                "llm": {
                    "backend": "openai",
                    "config": openai_config,
                },
                "embedder": embedder_config,
                "chunker": {
                    "backend": "sentence",
                    "config": {
                        "tokenizer_or_token_counter": "gpt2",
                        "chunk_size": kwargs.get("chunk_size", 512),
                        "chunk_overlap": kwargs.get("chunk_overlap", 128),
                        "min_sentences_per_chunk": 1,
                    },
                },
            },
        },
        "enable_textual_memory": True,
        "enable_activation_memory": kwargs.get("enable_activation_memory", False),
        "top_k": kwargs.get("top_k", 5),
        "max_turns_window": kwargs.get("max_turns_window", 20),
        "enable_mem_scheduler": kwargs.get("enable_mem_scheduler", False),
    }

    # Note: text_mem configuration is handled in get_default_cube_config
    # MOSConfig doesn't have text_mem field, it's only in MemCube config

    # Add scheduler configuration if enabled
    if config_dict.get("enable_mem_scheduler", False):
        config_dict["mem_scheduler"] = {
            "backend": "general_scheduler",
            "config": {
                "top_k": kwargs.get("scheduler_top_k", 10),
                "top_n": kwargs.get("scheduler_top_n", 5),
                "act_mem_update_interval": kwargs.get("scheduler_act_mem_update_interval", 300),
                "context_window_size": kwargs.get("scheduler_context_window_size", 5),
                "thread_pool_max_workers": kwargs.get("scheduler_thread_pool_max_workers", 10),
                "consume_interval_seconds": kwargs.get("scheduler_consume_interval_seconds", 0.01),
                "enable_parallel_dispatch": kwargs.get("scheduler_enable_parallel_dispatch", True),
                "enable_activation_memory": True,
            },
        }

    # Add activation memory if enabled
    if config_dict.get("enable_activation_memory", False):
        config_dict["act_mem"] = {
            "backend": "kv_cache",
            "config": {
                "memory_filename": kwargs.get(
                    "activation_memory_filename", "activation_memory.pickle"
                ),
                "extractor_llm": {
                    "backend": "openai",
                    "config": openai_config,
                },
            },
        }

    return MOSConfig(**config_dict)


def get_default_cube_config(
    openai_api_key: str,
    openai_api_base: str = "https://api.openai.com/v1",
    text_mem_type: Literal["tree_text", "general_text"] = "general_text",
    user_id: str = "default_user",
    **kwargs,
) -> GeneralMemCubeConfig:
    """
    Generate a default MemCube configuration with minimal user input.

    Args:
        openai_api_key (str): OpenAI API key
        openai_api_base (str): OpenAI API base URL, defaults to "https://api.openai.com/v1"
        text_mem_type (str): Type of text memory, either "tree_text" or "general_text"
        user_id (str): User ID for the configuration
        **kwargs: Additional configuration overrides

    Returns:
        GeneralMemCubeConfig: Complete MemCube configuration object
    """

    # Base OpenAI configuration
    openai_config = {
        "model_name_or_path": kwargs.get("model_name", "gpt-4o-mini"),
        "temperature": kwargs.get("temperature", 0.8),
        "max_tokens": kwargs.get("max_tokens", 1024),
        "top_p": kwargs.get("top_p", 0.9),
        "top_k": kwargs.get("top_k", 50),
        "remove_think_prefix": True,
        "api_key": openai_api_key,
        "api_base": openai_api_base,
    }

    # Embedder configuration — supports ollama or universal_api backends
    embedder_backend = kwargs.get("embedder_backend", os.getenv("MOS_EMBEDDER_BACKEND", "universal_api"))
    if embedder_backend == "ollama":
        embedder_config = {
            "backend": "ollama",
            "config": {
                "model_name_or_path": kwargs.get("embedder_model", "bge-m3"),
                "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            },
        }
    else:
        embedder_config = {
            "backend": "universal_api",
            "config": {
                "provider": "openai",
                "api_key": openai_api_key,
                "model_name_or_path": kwargs.get("embedder_model", "text-embedding-3-large"),
                "base_url": openai_api_base,
            },
        }

    # Build LLM config for tree_text (supports ollama or openai backends)
    llm_backend = kwargs.get("llm_backend", os.getenv("MOS_LLM_BACKEND", "openai"))
    if llm_backend == "ollama":
        llm_model = kwargs.get("llm_model", os.getenv("MOS_LLM_MODEL", "qwen3:0.6b"))
        llm_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        tree_llm_config = {
            "backend": "ollama",
            "config": {
                "model_name_or_path": llm_model,
                "api_base": llm_api_base,
                "temperature": kwargs.get("llm_temperature", 0.1),
                "max_tokens": kwargs.get("llm_max_tokens", 256),
                "remove_think_prefix": True,
            },
        }
    else:
        tree_llm_config = {"backend": "openai", "config": openai_config}

    # Configure text memory based on type
    if text_mem_type == "tree_text":
        # Tree text memory requires Neo4j configuration
        # NOTE: Neo4j Community Edition does NOT support multiple databases.
        # It only has one default database named 'neo4j'.
        # If you are using Community Edition:
        # 1. Set 'use_multi_db' to False (default)
        # 2. Set 'db_name' to 'neo4j' (default)
        # 3. Set 'auto_create' to False to avoid 'CREATE DATABASE' permission errors.
        db_name = f"memos{user_id.replace('-', '').replace('_', '')}"
        if not kwargs.get("use_multi_db", False):
            db_name = kwargs.get("neo4j_db_name", "neo4j")

        neo4j_config = {
            "uri": kwargs.get("neo4j_uri", os.getenv("NEO4J_URI", "bolt://localhost:7687")),
            "user": kwargs.get("neo4j_user", os.getenv("NEO4J_USER", "neo4j")),
            "db_name": db_name,
            "password": kwargs.get("neo4j_password", os.getenv("NEO4J_PASSWORD", "12345678")),
            "auto_create": kwargs.get("neo4j_auto_create", False),
            "use_multi_db": kwargs.get("use_multi_db", False),
            "embedding_dimension": kwargs.get("vector_dimension", int(os.getenv("EMBEDDING_DIMENSION", "1024"))),
        }
        if not kwargs.get("use_multi_db", False):
            neo4j_config["user_name"] = f"memos{user_id.replace('-', '').replace('_', '')}"

        # Vec config for Qdrant — required by neo4j-community backend for vector storage
        qdrant_host = kwargs.get("qdrant_host", os.getenv("QDRANT_HOST"))
        qdrant_port = kwargs.get("qdrant_port", os.getenv("QDRANT_PORT"))
        if qdrant_host or qdrant_port:
            neo4j_config["vec_config"] = {
                "backend": "qdrant",
                "config": {
                    "collection_name": kwargs.get("qdrant_collection", "neo4j_vec_db"),
                    "vector_dimension": kwargs.get("vector_dimension", int(os.getenv("EMBEDDING_DIMENSION", "1024"))),
                    "distance_metric": "cosine",
                    "host": qdrant_host or "localhost",
                    "port": int(qdrant_port) if qdrant_port else 6333,
                    "path": os.getenv("QDRANT_PATH"),
                    "url": os.getenv("QDRANT_URL"),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                },
            }

        # Search strategy: "fast" = vector-only (1 LLM call via reasoner)
        search_strategy = kwargs.get("search_strategy", {
            "fast_graph": True,
            "bm25": False,
            "cot": False,
            "fulltext": False,
        })

        # Memory size limits
        memory_size = kwargs.get("memory_size", {
            "WorkingMemory": 20,
            "LongTermMemory": 1500,
            "UserMemory": 10000,
        })

        text_mem_config = {
            "backend": "tree_text",
            "config": {
                "extractor_llm": tree_llm_config,
                "dispatcher_llm": tree_llm_config,
                "graph_db": {
                    "backend": kwargs.get("graph_db_backend", os.getenv("NEO4J_BACKEND", "neo4j-community")),
                    "config": neo4j_config,
                },
                "embedder": embedder_config,
                "reorganize": kwargs.get("enable_reorganize", False),
                "search_strategy": search_strategy,
                "memory_size": memory_size,
                "mode": kwargs.get("tree_mode", "sync"),
            },
        }

    elif text_mem_type == "general_text":
        # General text memory with file storage
        text_mem_config = {
            "backend": "general_text",
            "config": {
                "cube_id": kwargs.get("cube_id", f"{user_id}_cube"),
                "memory_filename": kwargs.get("memory_filename", "textual_memory.json"),
                "extractor_llm": {"backend": "openai", "config": openai_config},
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": kwargs.get("collection_name", f"{user_id}_collection"),
                        "vector_dimension": kwargs.get("vector_dimension", 3072),
                        "distance_metric": "cosine",
                        **({"host": kwargs["qdrant_host"]} if "qdrant_host" in kwargs else {}),
                        **({"port": kwargs["qdrant_port"]} if "qdrant_port" in kwargs else {}),
                    },
                },
                "embedder": embedder_config,
            },
        }

    # Configure activation memory if enabled
    act_mem_config = {}
    if kwargs.get("enable_activation_memory", False):
        act_mem_config = {
            "backend": "kv_cache",
            "config": {
                "memory_filename": kwargs.get(
                    "activation_memory_filename", "activation_memory.pickle"
                ),
                "extractor_llm": {
                    "backend": "openai",
                    "config": openai_config,
                },
            },
        }

    # Create MemCube configuration
    cube_config_dict = {
        "user_id": user_id,
        "cube_id": kwargs.get("cube_id", f"{user_id}_default_cube"),
        "text_mem": text_mem_config,
        "act_mem": act_mem_config,
        "para_mem": {},  # Empty parametric memory by default
    }

    return GeneralMemCubeConfig.model_validate(cube_config_dict)


def get_default(
    openai_api_key: str,
    openai_api_base: str = "https://api.openai.com/v1",
    text_mem_type: Literal["tree_text", "general_text"] = "general_text",
    user_id: str = "default_user",
    **kwargs,
) -> tuple[MOSConfig, GeneralMemCube]:
    """
    Generate both MOS configuration and default MemCube with minimal user input.

    This is the main convenience function for getting started with MemOS.

    Args:
        openai_api_key (str): OpenAI API key
        openai_api_base (str): OpenAI API base URL, defaults to "https://api.openai.com/v1"
        text_mem_type (str): Type of text memory, either "tree_text" or "general_text"
        user_id (str): User ID for the configuration
        **kwargs: Additional configuration overrides

    Returns:
        Tuple[MOSConfig, GeneralMemCube]: Complete MOS configuration and MemCube instance

    Example:
        ```python
        mos_config, default_cube = get_default(
            openai_api_key="sk-...",
            text_mem_type="general_text"
        )
        memory = MOS(mos_config)
        memory.register_mem_cube(default_cube)
        ```
    """

    # Generate MOS configuration
    mos_config = get_default_config(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        text_mem_type=text_mem_type,
        user_id=user_id,
        **kwargs,
    )

    # Generate MemCube configuration
    cube_config = get_default_cube_config(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        text_mem_type=text_mem_type,
        user_id=user_id,
        **kwargs,
    )

    # Create MemCube instance
    default_cube = GeneralMemCube(cube_config)

    return mos_config, default_cube


def get_simple_config(
    openai_api_key: str,
    openai_api_base: str = "https://api.openai.com/v1",
    text_mem_type: Literal["tree_text", "general_text"] = "general_text",
    user_id: str = "default_user",
) -> MOSConfig:
    """
    Get a minimal configuration with only essential parameters.

    This is the simplest way to get started with MemOS.

    Args:
        openai_api_key (str): OpenAI API key
        openai_api_base (str): OpenAI API base URL
        text_mem_type (str): Type of text memory
        user_id (str): User ID

    Returns:
        MOSConfig: Basic MOS configuration

    Example:
        ```python
        config = get_simple_config(
            openai_api_key="sk-...",
            text_mem_type="general_text"
        )
        mos = MOS(config)
        ```
    """
    return get_default_config(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        text_mem_type=text_mem_type,
        user_id=user_id,
    )
