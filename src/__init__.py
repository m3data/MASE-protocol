"""
MASE - Many Agent Socratic Exploration

Multi-model orchestrator for polyphonic AI dialogue experiments.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A
"""

__version__ = "0.2.0"

# Phase 1: Core infrastructure
from .ollama_client import OllamaClient, ResponseMetadata
from .agents import (
    Agent,
    AgentConfig,
    EnsembleConfig,
    AgentLoader,
    load_ensemble
)
from .embedding_service import (
    EmbeddingService,
    get_embedding_service,
    embed
)

# Phase 2: Orchestration
from .session_logger import SessionLogger, SessionRecord, TurnRecord
from .orchestrator import (
    DialogueOrchestrator,
    TurnSelector,
    run_session
)

__all__ = [
    # Version
    "__version__",
    # Ollama client
    "OllamaClient",
    "ResponseMetadata",
    # Agents
    "Agent",
    "AgentConfig",
    "EnsembleConfig",
    "AgentLoader",
    "load_ensemble",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "embed",
    # Session logging
    "SessionLogger",
    "SessionRecord",
    "TurnRecord",
    # Orchestration
    "DialogueOrchestrator",
    "TurnSelector",
    "run_session",
]
