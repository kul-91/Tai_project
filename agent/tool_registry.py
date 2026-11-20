"""Secure Tool Registry for SentinelAI.

Enforces strict explicit tool registration, parameter validation, and safe invocation.
Arbitrary execution, shell commands, dynamic registration, and unauthorized access
are strictly prevented. All filesystem operations are gated by security.Policy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from security.policy import Policy
from tools import filesystem


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class ToolError(Exception):
    """Base exception for all tool registry and execution errors."""
    pass


class UnknownToolError(ToolError, KeyError, ValueError):
    """Raised when an unregistered, missing, or invalid tool is invoked."""
    pass


class ToolArgumentError(ToolError, ValueError):
    """Raised when tool arguments fail validation (missing, unexpected, or wrong type)."""
    pass


class ToolExecutionLimitError(ToolError, RuntimeError):
    """Raised when the maximum tool call limit per request is exceeded."""
    pass


# ─── Tool Parameter Specification ─────────────────────────────────────────────

@dataclass
class ToolParameter:
    """Specification of a tool parameter for strict validation."""
    name: str
    type_: Union[type, tuple[type, ...]]
    required: bool = True
    default: Any = None
    description: str = ""

    @property
    def type_name(self) -> str:
        """Human-readable representation of acceptable types."""
        if isinstance(self.type_, tuple):
            return " | ".join(t.__name__ for t in self.type_)
        return self.type_.__name__


# ─── Tool Result ──────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Predictable return container for tool executions."""
    tool_name: str
    output: Any
    sources: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "sources": self.sources,
            "error": self.error,
        }


# ─── Tool Definition ──────────────────────────────────────────────────────────

class Tool:
    """Represents an explicitly registered tool with strict parameter validation."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: Optional[Sequence[ToolParameter]] = None,
    ) -> None:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if not callable(func):
            raise TypeError(f"Tool '{name}' must have a callable function.")

        self.name = name.strip()
        self.description = description.strip()
        self.func = func
        self.parameters: dict[str, ToolParameter] = {}

        if parameters:
            for p in parameters:
                if not isinstance(p, ToolParameter):
                    raise TypeError(f"Parameter '{p}' must be a ToolParameter instance.")
                self.parameters[p.name] = p

    def validate_arguments(self, arguments: Any) -> dict[str, Any]:
        """Validates arguments against the tool's parameter specification.

        Raises:
            ToolArgumentError: On missing required args, unexpected args, or wrong types.
        """
        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"Arguments for tool '{self.name}' must be a dictionary, got {type(arguments).__name__}."
            )

        validated: dict[str, Any] = {}

        # 1. Check for unexpected arguments
        for arg_name in arguments:
            if arg_name not in self.parameters:
                allowed = sorted(list(self.parameters.keys()))
                raise ToolArgumentError(
                    f"Unexpected argument '{arg_name}' for tool '{self.name}'. Allowed arguments: {allowed}"
                )

        # 2. Check each defined parameter
        for param_name, param in self.parameters.items():
            if param_name in arguments:
                val = arguments[param_name]

                # Check for null bytes in string arguments
                if isinstance(val, str) and "\0" in val:
                    raise ToolArgumentError(f"Argument '{param_name}' contains illegal null bytes.")

                # Check type
                if val is not None:
                    expected = param.type_
                    # Allow int where float is expected
                    is_valid_type = False
                    if expected is float and isinstance(val, (int, float)):
                        is_valid_type = True
                    elif isinstance(expected, tuple):
                        is_valid_type = any(
                            isinstance(val, t) or (t is float and isinstance(val, (int, float)))
                            for t in expected
                        )
                    else:
                        is_valid_type = isinstance(val, expected)

                    if not is_valid_type:
                        raise ToolArgumentError(
                            f"Argument '{param_name}' for tool '{self.name}' must be of type "
                            f"{param.type_name}, got {type(val).__name__}."
                        )
                elif param.required:
                    raise ToolArgumentError(
                        f"Required argument '{param_name}' for tool '{self.name}' cannot be None."
                    )

                validated[param_name] = val
            elif param.required:
                raise ToolArgumentError(
                    f"Missing required argument '{param_name}' for tool '{self.name}'."
                )
            else:
                validated[param_name] = param.default

        return validated

    def __call__(self, **kwargs: Any) -> Any:
        validated_args = self.validate_arguments(kwargs)
        return self.func(**validated_args)


# ─── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """Secure registry holding explicitly allowed tools.

    Enforces that only registered tools can be invoked, rejects unknown tools,
    validates parameters, and ensures unauthorized requests fail closed.
    """

    def __init__(self, policy: Optional[Policy] = None) -> None:
        self.policy = policy if policy is not None else Policy()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, override: bool = False) -> None:
        """Register a tool instance explicitly.

        Args:
            tool: Tool instance to register.
            override: If True, allows replacing an existing registration.
        """
        if not isinstance(tool, Tool):
            raise TypeError("Tool must be an instance of Tool.")
        if not override and tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve tool by name or return None if not registered."""
        if not isinstance(name, str):
            return None
        return self._tools.get(name)

    def get_tool(self, name: str) -> Tool:
        """Retrieve tool by name or raise UnknownToolError."""
        if not isinstance(name, str) or not name.strip() or name not in self._tools:
            raise UnknownToolError(
                f"Unknown tool: '{name}'. Registered tools: {sorted(list(self._tools.keys()))}"
            )
        return self._tools[name]

    def __getitem__(self, name: str) -> Tool:
        return self.get_tool(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool name is registered."""
        if not isinstance(name, str):
            return False
        return name in self._tools

    def list_tools(self) -> list[dict[str, Any]]:
        """List metadata for all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    p_name: {
                        "type": p.type_name,
                        "required": p.required,
                        "default": p.default,
                        "description": p.description,
                    }
                    for p_name, p in tool.parameters.items()
                },
            }
            for tool in self._tools.values()
        ]

    def invoke(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Invoke a registered tool safely with argument validation.

        Args:
            name: Explicit tool name.
            arguments: Dictionary of arguments for the tool.

        Returns:
            ToolResult containing execution status, output, and source citations.

        Raises:
            UnknownToolError: If tool name is unregistered or invalid.
            ToolArgumentError: If arguments fail validation.
            PermissionError: If tool execution violates security.Policy.
        """
        tool = self.get_tool(name)
        args = arguments if arguments is not None else {}
        validated_args = tool.validate_arguments(args)

        # Execute underlying function (gated by Policy)
        raw_output = tool.func(**validated_args)

        # Normalize output and extract source citations
        sources: list[dict[str, Any]] = []
        output = raw_output

        if isinstance(raw_output, ToolResult):
            return raw_output
        elif isinstance(raw_output, dict) and "answer" in raw_output and "sources" in raw_output:
            # Knowledge base question answering output
            sources = raw_output.get("sources", [])
            output = raw_output.get("answer", "")
        elif name == "read_file" and "path" in validated_args:
            resolved_p = str(Path(validated_args["path"]).expanduser().resolve())
            sources = [{"source": resolved_p, "page": None}]
        elif name == "search_files" and isinstance(raw_output, list):
            sources = [
                {"source": item.get("path"), "name": item.get("name")}
                for item in raw_output
                if isinstance(item, dict)
            ]

        return ToolResult(
            tool_name=name,
            output=output,
            sources=sources,
            success=True,
            error=None,
        )

    def execute(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Alias for invoke()."""
        return self.invoke(name, arguments)


# ─── Default Factory ──────────────────────────────────────────────────────────

def create_default_registry(
    policy: Optional[Policy] = None,
    knowledge_base: Optional[Any] = None,
) -> ToolRegistry:
    """Creates a pre-configured ToolRegistry with standard SentinelAI tools.

    Registers:
    - list_directory
    - search_files
    - read_file
    - ask_knowledge_base (if knowledge_base is provided)
    """
    pol = policy if policy is not None else Policy()
    registry = ToolRegistry(policy=pol)

    # 1. list_directory
    registry.register(
        Tool(
            name="list_directory",
            description="List directory contents strictly within permitted Policy locations.",
            func=lambda path: filesystem.list_directory(path, policy=pol),
            parameters=[
                ToolParameter(
                    name="path",
                    type_=(str, Path),
                    required=True,
                    description="Directory path to list.",
                )
            ],
        )
    )

    # 2. search_files
    registry.register(
        Tool(
            name="search_files",
            description="Recursively search for files matching query within permitted directory.",
            func=lambda directory, query: filesystem.search_files(directory, query, policy=pol),
            parameters=[
                ToolParameter(
                    name="directory",
                    type_=(str, Path),
                    required=True,
                    description="Root directory to search within.",
                ),
                ToolParameter(
                    name="query",
                    type_=str,
                    required=True,
                    description="Filename search query string.",
                ),
            ],
        )
    )

    # 3. read_file
    registry.register(
        Tool(
            name="read_file",
            description="Read text file content safely within permitted locations.",
            func=lambda path, max_bytes=filesystem.DEFAULT_MAX_FILE_SIZE: filesystem.read_file(
                path, policy=pol, max_bytes=max_bytes
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type_=(str, Path),
                    required=True,
                    description="Path to the file to read.",
                ),
                ToolParameter(
                    name="max_bytes",
                    type_=int,
                    required=False,
                    default=filesystem.DEFAULT_MAX_FILE_SIZE,
                    description="Maximum allowed file size in bytes.",
                ),
            ],
        )
    )

    # 4. ask_knowledge_base (if knowledge_base is supplied)
    if knowledge_base is not None:
        registry.register(
            Tool(
                name="ask_knowledge_base",
                description="Query the indexed knowledge base for answers with evidence and source citations.",
                func=lambda question, k=5, threshold=0.35: knowledge_base.ask(
                    question=question, k=k, threshold=threshold
                ),
                parameters=[
                    ToolParameter(
                        name="question",
                        type_=str,
                        required=True,
                        description="User question to answer using indexed documents.",
                    ),
                    ToolParameter(
                        name="k",
                        type_=int,
                        required=False,
                        default=5,
                        description="Maximum number of evidence chunks to retrieve.",
                    ),
                    ToolParameter(
                        name="threshold",
                        type_=(int, float),
                        required=False,
                        default=0.35,
                        description="Minimum similarity score threshold.",
                    ),
                ],
            )
        )

    return registry
