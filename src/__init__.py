"""
MASE - Many Agent Socratic Exploration

Multi-model orchestrator for polyphonic AI dialogue experiments.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A
"""

__version__ = "0.4.0"

# Phase 1: Core infrastructure
from .ollama_client import OllamaClient, ResponseMetadata, ModelWarmthManager
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

# Phase 3: Metrics & Experiments
from .metrics import (
    MetricsResult,
    semantic_curvature,
    dfa_alpha,
    entropy_shift,
    semantic_velocity,
    compute_metrics,
    compute_metrics_from_session
)
from .experiment import (
    ExperimentRunner,
    PairResult,
    ConditionResult,
    ExperimentResult,
    run_matched_pair
)
from .resume import (
    find_checkpoints,
    find_incomplete_pairs,
    analyze_checkpoint,
    print_status,
    CheckpointInfo
)

__all__ = [
    # Version
    "__version__",
    # Ollama client
    "OllamaClient",
    "ResponseMetadata",
    "ModelWarmthManager",
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
    # Metrics
    "MetricsResult",
    "semantic_curvature",
    "dfa_alpha",
    "entropy_shift",
    "semantic_velocity",
    "compute_metrics",
    "compute_metrics_from_session",
    # Experiments
    "ExperimentRunner",
    "PairResult",
    "ConditionResult",
    "ExperimentResult",
    "run_matched_pair",
    # Resume utilities
    "find_checkpoints",
    "find_incomplete_pairs",
    "analyze_checkpoint",
    "print_status",
    "CheckpointInfo",
]
