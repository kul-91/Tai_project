"""Secure Agent Orchestrator for SentinelAI.

Coordinates user requests and tool execution through strict Policy authorization,
explicit tool registration, parameter validation, and call limits.
No arbitrary Python execution, subprocess spawning, dynamic LLM tool registration,
or path traversal escapes are permitted. All outputs from retrieved documents are
treated as untrusted data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

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
from security.policy import Policy

logger = logging.getLogger("sentinelai.agent.orchestrator")


class AgentOrchestrator:
    """Secure orchestrator for invoking explicitly registered SentinelAI tools.

    Ensures:
    - Only registered tools can ever be executed.
    - All filesystem paths pass strictly through security.Policy authorization.
    - Deterministic routing or local LLM tool selection (Step 6.2).
    - Maximum tool calls per request are enforced.
    - Returns structured responses with answers, tool calls, results, and sources.
    - Fails closed on unauthorized access, unknown tools, or invalid arguments.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        policy: Optional[Policy] = None,
        knowledge_base: Optional[Any] = None,
        max_tool_calls: int = 5,
        raise_on_error: bool = False,
        model_name: str = DEFAULT_MODEL_NAME,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_client: Optional[Any] = None,
        use_llm: Optional[bool] = None,
        max_model_tool_calls: int = 1,
    ) -> None:
        """Initialize AgentOrchestrator.

        Args:
            registry: Optional pre-configured ToolRegistry.
            policy: Security Policy instance. Defaults to Policy().
            knowledge_base: Optional KnowledgeBase instance.
            max_tool_calls: Maximum tool invocations allowed per request.
            raise_on_error: If True, raises exceptions directly instead of returning
                            a structured error response. Default is False (fail-closed structured dict).
            model_name: Local Ollama model name (defaults to qwen3:4b).
            ollama_base_url: Base URL for local Ollama service.
            ollama_client: Optional pre-configured or mock ollama.Client.
            use_llm: If True, uses local LLM tool selection for natural language queries.
                     Defaults to True if ollama_client is provided, else False.
            max_model_tool_calls: Maximum model-selected tool executions per request (default 1).
        """
        self.policy = policy if policy is not None else Policy()
        self.knowledge_base = knowledge_base
        self.max_tool_calls = max_tool_calls
        self.raise_on_error = raise_on_error
        self.total_tool_calls: int = 0
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.ollama_client = ollama_client
        self.use_llm = use_llm if use_llm is not None else (ollama_client is not None)
        self.max_model_tool_calls = max_model_tool_calls

        if registry is not None:
            self.registry = registry
            # If knowledge_base was supplied but not registered yet, register it
            if self.knowledge_base is not None and not self.registry.has_tool("ask_knowledge_base"):
                self._register_kb_tool(self.knowledge_base)
        else:
            self.registry = create_default_registry(
                policy=self.policy,
                knowledge_base=self.knowledge_base,
            )

        self.selector = LLMToolSelector(
            registry=self.registry,
            model_name=self.model_name,
            ollama_base_url=self.ollama_base_url,
            client=self.ollama_client,
        )

    def select_tool(self, request_text: str) -> dict[str, Any]:
        """Selects an appropriate tool using the configured local model.

        Args:
            request_text: Natural language user query.

        Returns:
            Dictionary containing {"tool_name": str, "arguments": dict}.
        """
        return self.selector.select_tool(request_text)

    def _register_kb_tool(self, kb: Any) -> None:
        """Register knowledge base question answering tool."""
        self.registry.register(
            Tool(
                name="ask_knowledge_base",
                description="Query the indexed knowledge base for answers with evidence and source citations.",
                func=lambda question, k=5, threshold=0.35: kb.ask(
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

    # ── Deterministic Request Routing ─────────────────────────────────────────

    def _route_request(self, request_text: str) -> tuple[Optional[str], Union[dict[str, Any], str]]:
        """Deterministically routes a natural language request to an appropriate tool.

        Returns:
            Tuple of (tool_name, arguments_dict) on successful routing.
            Tuple of (None, clarification_or_error_message) if command is ambiguous or incomplete.
        """
        text = request_text.strip()
        lower_text = text.lower()

        # 1. Catch bare/incomplete commands before any guessing
        if lower_text in {"read", "read file", "read_file", "cat"}:
            return (None, "Incomplete command: 'read file' requires a file path (e.g. 'read file /path/to/file.txt').")
        if lower_text in {"list", "list directory", "list_directory", "list dir", "list files", "ls", "dir"}:
            return (None, "Incomplete command: 'list directory' requires a directory path (e.g. 'list directory /path/to/dir').")
        if lower_text in {"search", "search files", "search_files", "find files"}:
            return (None, "Incomplete command: 'search files' requires both a directory and a search query (e.g. 'search files in <dir> for <query>').")

        # 2. Directory listing commands:
        # e.g., "list directory /path", "list_directory /path", "ls /path", "dir /path", "list files in /path"
        m_list = re.match(
            r"^(?:list\s+(?:the\s+)?(?:directory|dir|files\s+in)|list_directory|ls|dir)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_list:
            path_val = m_list.group(1).strip().strip("'\"")
            if not path_val:
                return (None, "Incomplete command: 'list directory' requires a directory path.")
            return ("list_directory", {"path": path_val})

        # 3. Read file commands (must explicitly indicate file reading, NOT bare 'read'):
        # e.g., "read file /path", "read_file /path", "cat /path"
        m_read = re.match(
            r"^(?:read\s+(?:the\s+)?file|read_file|cat)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_read:
            path_val = m_read.group(1).strip().strip("'\"")
            if not path_val:
                return (None, "Incomplete command: 'read file' requires a file path.")
            # Check for conversational ambiguity: e.g. "read file to answer questions about RAG"
            has_path_marker = "/" in path_val or "\\" in path_val or path_val.startswith((".", "~"))
            has_file_ext = bool(re.search(r"\.[a-zA-Z0-9]{1,5}$", path_val))
            if " " in path_val and not (has_path_marker or has_file_ext):
                return (
                    None,
                    f"Ambiguous command: '{text}'. If you intended to read a file, provide a valid path. "
                    "If asking a question, ask without 'read file'.",
                )
            return ("read_file", {"path": path_val})

        # 4. Search files commands:
        # e.g., "search files in /dir for query", "search files /dir for query", "find files in /dir matching query"
        m_search = re.match(
            r"^(?:search\s+(?:for\s+files\s+in|files?\s+in|files?)\s+|find\s+files?\s+(?:in\s+)?)(.+?)\s+(?:for|matching)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_search:
            dir_val = m_search.group(1).strip().strip("'\"")
            query_val = m_search.group(2).strip().strip("'\"")
            if not dir_val or not query_val:
                return (None, "Incomplete command: 'search files' requires both a directory and a search query.")
            return ("search_files", {"directory": dir_val, "query": query_val})

        # Incomplete search files attempts: e.g. "search files in /path" (no query)
        m_search_incomplete = re.match(
            r"^(?:search\s+(?:for\s+files\s+in|files?\s+in|files?)|find\s+files?\s+(?:in\s+)?)\s*(.*)$",
            text,
            re.IGNORECASE,
        )
        if m_search_incomplete:
            return (
                None,
                "Incomplete command: 'search files' requires both a directory and a search query "
                "(e.g. 'search files in <dir> for <query>').",
            )

        # 5. Explicit ask / query knowledge base command:
        # e.g., "ask question", "ask the knowledge base question", "query question"
        m_ask = re.match(
            r"^(?:ask\s+(?:the\s+)?(?:knowledge\s+base|kb)\s+|ask_knowledge_base\s+|ask\s+)(.+)$",
            text,
            re.IGNORECASE,
        )
        if m_ask:
            q = m_ask.group(1).strip()
            if not q:
                return (None, "Incomplete command: 'ask' requires a question string.")
            return ("ask_knowledge_base", {"question": q})

        # 6. Ordinary document questions:
        # If knowledge base is registered, route natural language questions / queries to ask_knowledge_base
        if self.registry.has_tool("ask_knowledge_base"):
            is_question = (
                text.endswith("?")
                or any(
                    lower_text.startswith(kw)
                    for kw in (
                        "what", "where", "when", "who", "why", "how", "which",
                        "is", "are", "can", "could", "does", "do", "will", "would",
                        "tell", "explain", "describe", "summarize", "show",
                        "find out", "lookup", "check", "read through", "read about",
                        "read me", "read the",
                    )
                )
                or len(text.split()) >= 3  # multi-word descriptive queries
            )
            if is_question:
                return ("ask_knowledge_base", {"question": text})

        # 7. Ambiguous or unknown single words / unrecognized commands
        return (
            None,
            f"Ambiguous or unrecognized request: '{text}'. Please provide a valid command "
            "(e.g. 'list directory /path', 'read file /path') or ask a question.",
        )

    def _format_tool_answer(self, tool_name: str, output: Any) -> str:
        """Formats raw tool output into a user-facing answer string."""
        if isinstance(output, str):
            return output
        elif isinstance(output, list):
            if tool_name == "list_directory":
                items_desc = [
                    f"- {item.get('name')} ({'dir' if item.get('is_dir') else 'file'})"
                    for item in output
                ]
                return f"Directory contents ({len(output)} items):\n" + "\n".join(items_desc)
            elif tool_name == "search_files":
                matches_desc = [
                    f"- {item.get('name')}: {item.get('path')}"
                    for item in output
                ]
                return f"Found {len(output)} matching files:\n" + "\n".join(matches_desc)
            return str(output)
        elif isinstance(output, dict):
            return output.get("answer", str(output))
        return str(output)

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_tool(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        max_tool_calls: Optional[int] = None,
    ) -> dict[str, Any]:
        """Directly executes an explicitly specified registered tool.

        Args:
            tool_name: Registered tool name.
            arguments: Dictionary of arguments.
            max_tool_calls: Optional per-request call limit.

        Returns:
            Structured result dictionary.
        """
        return self.run(
            request="",
            tool_name=tool_name,
            arguments=arguments,
            max_tool_calls=max_tool_calls,
        )

    @staticmethod
    def _extract_call_args(item: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
        """Safely extracts tool name and arguments from a call dict, properly handling empty dicts."""
        t = item.get("tool") or item.get("tool_name") or item.get("action")
        if "arguments" in item and isinstance(item["arguments"], dict):
            a = item["arguments"]
        elif "args" in item and isinstance(item["args"], dict):
            a = item["args"]
        elif "params" in item and isinstance(item["params"], dict):
            a = item["params"]
        else:
            a = {k: v for k, v in item.items() if k not in ("tool", "tool_name", "action")}
        return (t, a)

    def run(
        self,
        request: Union[str, dict[str, Any], list[dict[str, Any]]],
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
        max_tool_calls: Optional[int] = None,
        use_llm: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Processes a request and executes the selected registered tool(s).

        Args:
            request: Natural language query string, structured request dict,
                     or list of tool call dicts for multi-step requests.
            tool_name: Optional explicit tool name (bypasses routing).
            arguments: Optional dictionary of arguments for explicit tool calls.
            max_tool_calls: Optional override for maximum tool calls allowed.
            use_llm: Optional override to enable/disable LLM tool selection.

        Returns:
            Structured response dictionary::

                {
                    "success": bool,
                    "answer": str,
                    "tool_name": Optional[str | list[str]],
                    "tool_calls": [
                        {
                            "tool": str,
                            "arguments": dict,
                            "success": bool,
                            "error": Optional[str],
                        }
                    ],
                    "result": Any,
                    "sources": list[dict],
                    "error": Optional[str],
                }
        """
        effective_limit = max_tool_calls if max_tool_calls is not None else self.max_tool_calls

        response: dict[str, Any] = {
            "success": False,
            "answer": "",
            "tool_name": None,
            "tool_calls": [],
            "result": None,
            "sources": [],
            "error": None,
        }

        # 1. Resolve raw tool calls to execute in this request
        raw_calls: list[tuple[Optional[str], dict[str, Any]]] = []

        if tool_name:
            raw_calls.append((tool_name, arguments if arguments is not None else {}))
        elif isinstance(request, list):
            if not request:
                return self._fail("Request list cannot be empty.", response)
            for idx, item in enumerate(request):
                if isinstance(item, dict):
                    raw_calls.append(self._extract_call_args(item))
                else:
                    return self._fail(
                        f"Invalid item at index {idx} in tool call list: expected dict, got {type(item).__name__}.",
                        response,
                    )
        elif isinstance(request, dict):
            if "tool_calls" in request and isinstance(request["tool_calls"], list):
                if not request["tool_calls"]:
                    return self._fail("tool_calls list cannot be empty.", response)
                for idx, item in enumerate(request["tool_calls"]):
                    if isinstance(item, dict):
                        raw_calls.append(self._extract_call_args(item))
                    else:
                        return self._fail(
                            f"Invalid item at index {idx} in tool_calls: expected dict, got {type(item).__name__}.",
                            response,
                        )
            else:
                raw_calls.append(self._extract_call_args(request))
        elif isinstance(request, str):
            req_text = request.strip()
            if not req_text:
                return self._fail("Request cannot be empty.", response)

            # Check if string is directly a registered tool name (e.g. run("list_directory", arguments={...}))
            if self.registry.has_tool(req_text) and arguments is not None:
                raw_calls.append((req_text, arguments))
            else:
                active_use_llm = use_llm if use_llm is not None else self.use_llm
                if active_use_llm:
                    # Enforce per-request model call limit (controlled execution)
                    effective_limit = min(effective_limit, self.max_model_tool_calls)
                    if effective_limit <= 0:
                        err_msg = f"Tool call limit exceeded: maximum allowed is {effective_limit} per request."
                        if self.raise_on_error:
                            raise ToolExecutionLimitError(err_msg)
                        return self._fail(err_msg, response)

                    try:
                        selection = self.select_tool(req_text)
                        routed_tool = selection["tool_name"]
                        target_args = dict(selection["arguments"])
                        if arguments:
                            target_args.update(arguments)
                        raw_calls.append((routed_tool, target_args))
                    except Exception as exc:
                        if self.raise_on_error:
                            raise
                        attempted_tool = getattr(exc, "tool_name", None)
                        return self._fail(str(exc), response, tool_name=attempted_tool)
                else:
                    routed_tool, routed_data = self._route_request(req_text)
                    if routed_tool is None:
                        return self._fail(str(routed_data), response)
                    target_args = dict(routed_data) if isinstance(routed_data, dict) else {}
                    if arguments:
                        target_args.update(arguments)
                    raw_calls.append((routed_tool, target_args))
        else:
            return self._fail(
                f"Unsupported request type '{type(request).__name__}'. Must be str, dict, or list.",
                response,
            )

        # 2. Sequential execution with strict per-call limit enforcement
        calls_in_request = 0
        all_results = []
        all_sources = []
        all_answers = []

        for target_tool, target_args in raw_calls:
            # Validate tool name
            if not target_tool or not isinstance(target_tool, str) or not target_tool.strip():
                return self._fail("Tool name must be a non-empty string.", response)

            target_tool = target_tool.strip()

            if not self.registry.has_tool(target_tool):
                err_msg = f"Unknown tool: '{target_tool}'."
                if self.raise_on_error:
                    raise UnknownToolError(err_msg)
                return self._fail(err_msg, response, tool_name=target_tool, arguments=target_args)

            # Enforce call limit BEFORE every actual tool execution
            if calls_in_request >= effective_limit or effective_limit <= 0:
                err_msg = (
                    f"Tool call limit exceeded: maximum allowed is {effective_limit} per request."
                )
                response["tool_calls"].append({
                    "tool": target_tool,
                    "arguments": target_args,
                    "success": False,
                    "error": err_msg,
                })
                if self.raise_on_error:
                    raise ToolExecutionLimitError(err_msg)
                return self._fail(err_msg, response, tool_name=target_tool, arguments=target_args)

            # Prepare call record
            call_record: dict[str, Any] = {
                "tool": target_tool,
                "arguments": target_args,
                "success": False,
                "error": None,
            }
            response["tool_calls"].append(call_record)

            try:
                calls_in_request += 1
                self.total_tool_calls += 1

                result: ToolResult = self.registry.invoke(target_tool, target_args)

                call_record["success"] = True
                all_results.append(result.output)
                if result.sources:
                    all_sources.extend(result.sources)

                ans = self._format_tool_answer(target_tool, result.output)
                all_answers.append(ans)

            except Exception as exc:
                call_record["error"] = str(exc)
                if self.raise_on_error:
                    raise
                return self._fail(str(exc), response, tool_name=target_tool, arguments=target_args)

        # All requested calls executed successfully
        response["success"] = True
        response["sources"] = all_sources
        if len(all_results) == 1:
            response["result"] = all_results[0]
            response["answer"] = all_answers[0]
            response["tool_name"] = response["tool_calls"][0]["tool"]
        else:
            response["result"] = all_results
            response["answer"] = "\n\n".join(all_answers)
            response["tool_name"] = [c["tool"] for c in response["tool_calls"]]

        return response

    def _fail(
        self,
        error_msg: str,
        response: dict[str, Any],
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Helper to safely fail closed and populate error details."""
        response["success"] = False
        response["error"] = error_msg
        response["answer"] = ""
        if tool_name:
            if response.get("tool_name") is None:
                response["tool_name"] = tool_name
            if not response["tool_calls"]:
                response["tool_calls"].append({
                    "tool": tool_name,
                    "arguments": arguments or {},
                    "success": False,
                    "error": error_msg,
                })
        return response
