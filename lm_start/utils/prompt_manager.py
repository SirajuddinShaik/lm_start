"""
Prompt Manager

Centralized prompt management for all agents.
Prompts are loaded from YAML files in the prompts/ directory.
Supports Jinja2-style templating for dynamic content.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class PromptManager:
    """
    Manages prompts for agentic operations.

    Loads prompts from YAML files and provides templating capabilities.
    Supports fallback to default prompts if specific ones are not found.
    """

    _instance: Optional["PromptManager"] = None
    _prompts: Dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls, prompts_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, prompts_dir: Optional[str] = None):
        if not self._loaded:
            self._load_prompts(prompts_dir)

    def _get_prompts_directory(self) -> Path:
        """Get the prompts directory path."""
        # Look for prompts relative to this file
        current_file = Path(__file__)
        prompts_dir = current_file.parent.parent / "prompts"

        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        return prompts_dir

    def _load_prompts(self, prompts_dir: Optional[str] = None) -> None:
        """Load all prompt files from the prompts directory."""
        if prompts_dir:
            base_path = Path(prompts_dir)
        else:
            base_path = self._get_prompts_directory()

        self._prompts = {}

        # Load all YAML files in the prompts directory
        for yaml_file in base_path.glob("*.yaml"):
            category = yaml_file.stem  # filename without extension
            try:
                with open(yaml_file, "r") as f:
                    content = yaml.safe_load(f)
                    if content:
                        self._prompts[category] = content
            except Exception as e:
                print(f"[PromptManager] Warning: Failed to load {yaml_file}: {e}")

        self._loaded = True

        if not self._prompts:
            print("[PromptManager] Warning: No prompts loaded!")

    def reload(self) -> None:
        """Reload all prompts from disk."""
        self._loaded = False
        self._load_prompts()

    def get(
        self,
        key: str,
        context: Optional[Dict[str, Any]] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Get a prompt by key and render it with context.

        Args:
            key: Dot-notation key (e.g., "defaults.strategy_consultation")
            context: Dictionary of variables to substitute
            default: Default value if prompt not found

        Returns:
            Rendered prompt string
        """
        keys = key.split(".")
        value = self._prompts

        # Navigate to the prompt
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                if default is not None:
                    value = default
                    break
                raise KeyError(f"Prompt not found: {key}")

        # Convert to string if needed
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        elif not isinstance(value, str):
            value = str(value)

        # Apply template substitution
        if context:
            value = self._render_template(value, context)

        return value

    def get_mock_response(self, prompt_type: str) -> str:
        """
        Get a mock LLM response for testing/development.

        Args:
            prompt_type: Type of mock response (next_tests, error_fix, strategy, default)

        Returns:
            JSON string with mock response
        """
        try:
            return self.get(f"defaults.mock_responses.{prompt_type}")
        except KeyError:
            # Fallback to default mock
            return json.dumps(
                {
                    "analysis": "Continue optimization process",
                    "continue": True,
                    "fallback": True,
                }
            )

    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Simple template renderer (Jinja2-style).

        Supports:
        - {{variable}} - variable substitution
        - {{variable|json}} - JSON encode
        - {{variable|indent}} - indent multiline strings
        """
        result = template

        # Find all template variables
        pattern = r"\{\{\s*([\w.]+)(?:\|(\w+))?\s*\}\}"

        for match in re.finditer(pattern, template):
            full_match = match.group(0)
            var_path = match.group(1)
            filter_name = match.group(2)

            # Get value from context
            value = self._get_nested_value(context, var_path)

            # Apply filters
            if filter_name == "json":
                value = json.dumps(value, indent=2, default=str)
            elif filter_name == "indent":
                if isinstance(value, str):
                    lines = value.split("\n")
                    value = "\n  ".join(lines)
            elif filter_name == "upper":
                value = str(value).upper()
            elif filter_name == "lower":
                value = str(value).lower()

            # Convert to string
            if value is None:
                value = ""
            elif not isinstance(value, str):
                value = str(value)

            result = result.replace(full_match, value)

        return result

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get a nested value from a dictionary using dot notation."""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def list_prompts(self) -> List[str]:
        """List all available prompt keys."""
        keys = []

        def collect_keys(data: Any, prefix: str = ""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, str):
                        keys.append(new_prefix)
                    else:
                        collect_keys(value, new_prefix)

        collect_keys(self._prompts)
        return sorted(keys)

    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all prompts in a category."""
        return self._prompts.get(category, {})


# Global instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(
    key: str,
    context: Optional[Dict[str, Any]] = None,
    default: Optional[str] = None,
) -> str:
    """
    Convenience function to get a prompt.

    Args:
        key: Dot-notation key (e.g., "defaults.strategy_consultation")
        context: Dictionary of variables to substitute
        default: Default value if prompt not found

    Returns:
        Rendered prompt string
    """
    return get_prompt_manager().get(key, context, default)


def get_mock_response(prompt_type: str) -> str:
    """Convenience function to get a mock response."""
    return get_prompt_manager().get_mock_response(prompt_type)
