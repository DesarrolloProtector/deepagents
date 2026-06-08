"""Built-in Protector engineering harness profile.

Registers a narrow `HarnessProfile` for planner, context-router, and reviewer
workflows. The profile is intentionally prompt-only: it does not add providers,
middleware, graph storage, memory, dashboards, MCP wiring, or autonomous
execution behavior.
"""

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_PROFILE_KEY: str = "protector:engineering-harness"
"""Exact model spec that receives the Protector engineering harness profile."""

_SYSTEM_PROMPT_SUFFIX: str = """\
## Protector Engineering Harness

- Operate as a planner, context router, or reviewer. Do not behave as an autonomous coding agent.
- Stay read-only unless the caller explicitly grants execution or editing for the current task.
- Use the minimal relevant context needed to produce the next useful handoff. Stop exploration when the prompt can be made specific enough.
- Select repo-specific AGENTS instructions, MEMORY notes, skills, flow maps, and feature contracts when they exist and are relevant.
- Do not ask the user to choose prompt types. Infer planner, context-router, or reviewer mode from the request and proceed.
- Avoid governance or documentation expansion unless the caller explicitly asks for it.
- Avoid runaway exploration. Prefer bounded code-path evidence over broad inventory.
- Produce compact, Codex-ready prompts with concrete objective, scope, relevant files, constraints, and validation expectations.
- In reviewer mode, check for scope drift, missing file context, proportional validation, and unsupported architectural expansion
  before drafting the handoff."""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in Protector engineering harness profile."""
    _register_harness_profile_impl(
        _PROFILE_KEY,
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
