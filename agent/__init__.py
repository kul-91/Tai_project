"""Agent package for SentinelAI."""

from agent.orchestrator import AgentOrchestrator
from agent.selector import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    LLMToolSelector,
    ModelConnectionError,
    ModelOutputError,
    ToolSelectionError,
)
from agent.tool_registry import (
    Tool,
    ToolArgumentError,
    ToolError,
    ToolExecutionLimitError,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
    create_default_registry,
)

__all__ = [
    "AgentOrchestrator",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OLLAMA_BASE_URL",
    "LLMToolSelector",
    "ModelConnectionError",
    "ModelOutputError",
    "Tool",
    "ToolArgumentError",
    "ToolError",
    "ToolExecutionLimitError",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSelectionError",
    "UnknownToolError",
    "create_default_registry",
]
