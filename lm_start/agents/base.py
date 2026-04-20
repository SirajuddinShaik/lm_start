"""
Base Agent Module

Provides common functionality for all agents including:
- LLM client integration
- Error context tracking
- Logging
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from lm_start.utils.prompt_manager import get_prompt, get_mock_response


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    message: str
    actions_taken: List[str] = field(default_factory=list)
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """
    Base class for all agents.

    Provides common functionality for error handling, LLM integration,
    and result tracking.
    """

    def __init__(
        self,
        name: str,
        model_dir: Optional[str] = None,
        max_retries: int = 5,
        verbose: bool = False,
    ):
        self.name = name
        self.model_dir = Path(model_dir) if model_dir else None
        self.max_retries = max_retries
        self.verbose = verbose
        self.status = AgentStatus.IDLE
        self.attempt_count = 0
        self.action_history: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        prefix = f"[{self.name}]"
        if level == "error":
            print(f"{prefix} ERROR: {message}")
        elif level == "warn":
            print(f"{prefix} WARN: {message}")
        elif level == "success":
            print(f"{prefix} ✓ {message}")
        else:
            print(f"{prefix} {message}")

    def record_action(
        self, action: str, result: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an action taken by the agent."""
        self.action_history.append(
            {
                "action": action,
                "result": result,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent with given context.

        Args:
            context: Dictionary with error details, model info, etc.

        Returns:
            AgentResult with success status and actions taken
        """
        pass

    def consult_llm(self, prompt: str) -> str:
        """
        Consult LLM for a solution.

        Args:
            prompt: The prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            from core.config import get_config

            config = get_config()
            llm_config = config.llm

            self.log("Consulting LLM for strategy...", "info")

            # Try actual LLM call if available
            response = self._call_llm_api(prompt, llm_config)
            if response:
                return response

            # Fallback to smart mock based on prompt analysis
            return self._smart_mock_llm_call(prompt)

        except Exception as e:
            self.log(f"LLM consultation failed: {e}", "error")
            return self._smart_mock_llm_call(prompt)

    def _call_llm_api(self, prompt: str, llm_config) -> Optional[str]:
        """Try to call actual LLM API."""
        try:
            import requests

            model = llm_config.model
            timeout = llm_config.timeout

            # Try OpenAI-compatible endpoint
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "")

            if not api_key:
                return None

            response = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": llm_config.temperature,
                    "max_tokens": llm_config.max_tokens,
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

        except Exception:
            pass

        return None

    def _smart_mock_llm_call(self, prompt: str) -> str:
        """Smart mock that analyzes the prompt and returns relevant response."""
        prompt_lower = prompt.lower()

        if "next tests" in prompt_lower or "what should we test" in prompt_lower:
            return get_mock_response("next_tests")

        if "fix" in prompt_lower or "error" in prompt_lower:
            return get_mock_response("error_fix")

        if "strategy" in prompt_lower:
            return get_mock_response("strategy")

        return get_mock_response("default")


class FixStrategy:
    """
    Represents a fix strategy that can be applied.
    """

    def __init__(self, name: str, description: str, applies_to: List[str], action: Any):
        self.name = name
        self.description = description
        self.applies_to = applies_to
        self.action = action

    def can_apply(self, error_type: str) -> bool:
        """Check if this strategy can be applied to the error."""
        return error_type in self.applies_to

    def apply(self, context: Dict[str, Any]) -> AgentResult:
        """Apply the fix strategy."""
        try:
            return self.action(context)
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Fix failed: {e}",
                error_type="FIX_FAILED",
                error_details=str(e),
            )

    def spawn_opencode_session(
        self,
        task: str,
        context: Dict[str, Any],
        timeout_minutes: int = 30,
    ) -> Dict[str, Any]:
        """Spawn a new OpenCode session to handle complex tasks.

        This creates an isolated session with full context for manual intervention
        when automatic fixes fail.

        Args:
            task: Description of what needs to be fixed
            context: Dictionary with all relevant context (paths, errors, logs, etc.)
            timeout_minutes: How long to wait for the session

        Returns:
            Dictionary with 'success', 'message', and 'actions_taken'
        """
        import subprocess
        import tempfile
        import json
        from pathlib import Path

        print(f"[FixStrategy] Spawning OpenCode session for: {task}")

        context_file = Path(tempfile.mktemp(suffix="_opencode_context.json"))
        context_data = {
            "task": task,
            "agent_name": self.name,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }

        with open(context_file, "w") as f:
            json.dump(context_data, f, indent=2)

        print(f"[FixStrategy] Context saved to: {context_file}")

        # In a real implementation, this would:
        # 1. Call the OpenCode API to spawn a new session
        # 2. Pass the context file path
        # 3. Wait for the session to complete
        # 4. Collect results

        # For now, return a placeholder indicating what would happen
        return {
            "success": False,
            "message": "OpenCode spawning not yet implemented. Context prepared.",
            "actions_taken": ["prepared_context", "context_file_created"],
            "context_file": str(context_file),
            "task": task,
        }
