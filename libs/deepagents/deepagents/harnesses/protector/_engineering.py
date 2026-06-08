"""Reusable Protector engineering harness helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Literal

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

HARNESS_PROFILE = "protector:engineering-harness"
HARNESS_PROFILE_ENV_VAR = "DEEPAGENTS_ENGINEERING_HARNESS_PROFILE"
FEATURE_HINTS = frozenset(
    {
        "api",
        "behavior",
        "contract",
        "feature",
        "flow",
        "public",
        "workflow",
    }
)
TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "contract": ("firma", "signature"),
    "firma": ("contract", "signature"),
    "payment": ("pay", "receipt", "receipts"),
    "ui": ("operator", "surface"),
    "workflow": ("flow",),
}
REQUIRED_REVIEW_SECTIONS = (
    "Files read",
    "Files changed",
    "Summary",
    "Validation",
)
DRIFT_TERMS = (
    "dashboard",
    "memory",
    "graph",
    "mcp",
    "autonomous",
    "proposal",
    "apply",
    "archive",
    "governance",
    "documentation",
)
VALIDATION_EVIDENCE_TERMS = (
    "build",
    "check",
    "compiled",
    "passed",
    "pytest",
    "ruff",
    "smoke",
    "test",
    "verified",
)
VALIDATION_NEGATION_TERMS = (
    "not run",
    "not validated",
    "unable to run",
    "wasn't run",
    "were not run",
)


class HarnessUsageError(ValueError):
    """Raised when harness CLI-style inputs are invalid."""


@dataclass(frozen=True)
class RenderedOutput:
    """Console payload plus the exact prompt section available for file output."""

    payload: str
    codex_prompt: str
    selected_context_count: int


@dataclass(frozen=True)
class ReviewFindings:
    """Rendered reviewer findings plus machine-readable status metadata."""

    text: str
    status: str
    follow_up: str | None


@dataclass(frozen=True)
class _RepoContext:
    """Bounded read-only context routing result for a target repository."""

    root: Path
    mandatory: tuple[Path, ...]
    skills: tuple[tuple[str, Path], ...]
    flows: tuple[Path, ...]
    feature_contract: Path | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ContextSelection:
    """Selected and rejected context display rows."""

    selected: tuple[str, ...]
    not_selected: tuple[str, ...]


@dataclass(frozen=True)
class _ReviewResult:
    """Deterministic reviewer findings for a pasted Codex output."""

    status: str
    missing_sections: tuple[str, ...]
    drift_warnings: tuple[str, ...]
    validation_warnings: tuple[str, ...]
    follow_up: str | None


class _DryRunHarnessModel(BaseChatModel):
    """Local chat model used only to resolve a harness profile without a provider."""

    model_name: str
    provider: str
    bound_tools: tuple[object, ...] = Field(default=(), exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, object] | type | Callable[..., object] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: object,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Record tool binding without contacting a model provider."""
        _ = (tool_choice, kwargs)
        self.bound_tools = tuple(tools)
        return self

    def _get_ls_params(self) -> dict[str, str]:
        """Expose provider metadata so harness lookup resolves the profile."""
        return {"ls_provider": self.provider}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        """Return a deterministic dry-run response if accidentally invoked."""
        _ = (messages, stop, run_manager, kwargs)
        message = AIMessage(content="Dry-run model: no provider call was made.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        """Identify this as a local dry-run model."""
        return "protector-engineering-harness-dry-run"


def split_profile_key(spec: str) -> tuple[str, str]:
    """Split a `provider:model` profile key after validating the expected shape."""
    provider, separator, model = spec.partition(":")
    if not separator or not provider or not model or ":" in model:
        msg = f"expected a single `provider:model` spec, got {spec!r}"
        raise HarnessUsageError(msg)
    return provider, model


def resolve_harness_profile(profile: str) -> str:
    """Return the selected harness profile after validating this example's scope."""
    if profile != HARNESS_PROFILE:
        msg = f"this smoke path only supports --harness-profile {HARNESS_PROFILE}"
        raise HarnessUsageError(msg)
    return profile


def validate_real_model(model: str | None, harness_profile: str) -> None:
    """Reject confusing real-model values before constructing anything."""
    if model == harness_profile:
        msg = (
            f"--model {model!r} is invalid here: it names a harness profile, not a real provider model. "
            f"Use --harness-profile {harness_profile} and omit --model for the API-key-free dry run."
        )
        raise HarnessUsageError(msg)


def build_read_only_agent(harness_profile: str) -> object:
    """Construct a DeepAgent that resolves the Protector harness profile."""
    provider, model_name = split_profile_key(harness_profile)
    model = _DryRunHarnessModel(provider=provider, model_name=model_name)
    return create_deep_agent(
        model=model,
        backend=StateBackend(),
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/**"],
                mode="deny",
            )
        ],
        name="protector-engineering-harness-smoke",
    )


def render_output(
    *,
    task: str,
    repo: Path | None,
    mode: Literal["auto", "planner", "reviewer"],
    harness_profile: str,
    real_model: str | None,
    agent_type: str,
    output: Path | None,
) -> RenderedOutput:
    """Render a compact read-only handoff payload for Codex."""
    repo_text = str(repo.resolve()) if repo is not None else "(not provided)"
    selection = _select_context(repo, task)
    context_route = _render_context_route(selection)
    codex_prompt = _render_codex_prompt(task, selection)
    selected_context_count = _selected_context_count(selection)
    model_text = "not used in dry-run; no provider call was made"
    if real_model is not None:
        model_text = f"{real_model} (recorded only; not used in dry-run)"
    output_text = str(output.resolve()) if output is not None else "(not requested)"
    payload = f"""Codex-ready engineering harness prompt

Task: {task}
Repo: {repo_text}
Mode: {_mode_label(mode)}
Harness profile: {harness_profile}
Real model: {model_text}
Constructed DeepAgent: {agent_type}
Output file: {output_text}

{context_route}

{codex_prompt}

Instructions:
- Work read-only unless the caller explicitly grants execution or editing.
- Use minimal relevant context: AGENTS, MEMORY, skills, flow maps, and feature contracts when present.
- Do not ask the user to choose prompt types; infer planner/context-router/reviewer needs from the task.
- Avoid governance or documentation expansion and stop exploration once the handoff is specific enough.
- Produce a compact implementation or review prompt for Codex; do not run Codex here.
- Reviewer checks: scope drift, missing file context, proportional validation, and unsupported architectural expansion.

API note:
- The current string model path couples provider initialization and harness lookup, so harness profile selection is separate here.
- `create_deep_agent(model="{harness_profile}")` would try to initialize provider `protector`, so this smoke uses a prebuilt dry-run
  `BaseChatModel` whose provider/model metadata resolves the built-in harness profile without credentials."""
    return RenderedOutput(payload=payload, codex_prompt=codex_prompt, selected_context_count=selected_context_count)


def write_prompt_output(path: Path, prompt: str, *, overwrite: bool) -> None:
    """Write the prompt section to `path` with explicit overwrite protection."""
    target = path.resolve()
    if target.exists() and not overwrite:
        msg = f"output file already exists: {target}; pass --overwrite to replace it"
        raise HarnessUsageError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{prompt}\n", encoding="utf-8")


def render_review_findings(path: Path, task: str, text: str) -> str:
    """Render compact reviewer findings for a Codex output file."""
    return review_codex_output(task=task, output=text, source=str(path.resolve())).text


def review_codex_output(*, task: str, output: str, source: str) -> ReviewFindings:
    """Review Codex output from a file or in-memory paste using the same checks."""
    result = _review_codex_output(task, output)
    follow_up = result.follow_up or "(none)"
    rendered = f"""Reviewer Findings:
Status: {result.status}
Codex output: {source}
Missing sections:
{_one_line_list(result.missing_sections)}
Drift warnings:
{_one_line_list(result.drift_warnings)}
Validation warnings:
{_one_line_list(result.validation_warnings)}
Suggested follow-up prompt:
- {follow_up}"""
    return ReviewFindings(text=rendered, status=result.status, follow_up=result.follow_up)


def _mode_label(mode: Literal["auto", "planner", "reviewer"]) -> str:
    """Return a compact mode label for the Codex handoff."""
    if mode == "auto":
        return "infer planner or reviewer from the task"
    return mode


def _relative_path(root: Path, path: Path) -> str:
    """Format a path relative to the target repo when possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _one_line_list(items: tuple[str, ...]) -> str:
    """Render a compact bullet list."""
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


def _selected_context_count(selection: _ContextSelection) -> int:
    """Return the number of concrete context paths selected for the prompt."""
    return len(tuple(item for item in selection.selected if not item.startswith("repo not provided")))


def _tokens(text: str) -> frozenset[str]:
    """Return deterministic lowercase task/path tokens."""
    base = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}
    expanded = set(base)
    for token in base:
        expanded.update(TOKEN_ALIASES.get(token, ()))
    return frozenset(expanded)


def _matches_task(text: str, task_tokens: frozenset[str]) -> bool:
    """Return whether a path/name contains any task token."""
    return bool(_tokens(text) & task_tokens)


def _inspect_repo_context(repo: Path) -> _RepoContext:
    """Inspect only fixed context paths and one-level candidate directories."""
    root = repo.resolve()
    mandatory_candidates = (root / "AGENTS.md", root / "MEMORY.md")
    mandatory = tuple(path for path in mandatory_candidates if path.is_file())

    skills_root = root / ".codex" / "skills"
    skills: list[tuple[str, Path]] = []
    if skills_root.is_dir():
        for skill_dir in sorted((path for path in skills_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                skills.append((skill_dir.name, skill_file))

    flows_root = root / "Docs" / "flows"
    flows: tuple[Path, ...] = ()
    if flows_root.is_dir():
        flows = tuple(sorted((path for path in flows_root.glob("*.md") if path.is_file()), key=lambda path: path.name.lower()))

    feature_contract_path = root / ".codex" / "agent-workflow" / "feature-contract-template.md"
    feature_contract = feature_contract_path if feature_contract_path.is_file() else None

    warnings: list[str] = []
    if not (root / "AGENTS.md").is_file():
        warnings.append("missing AGENTS.md")
    if not (root / "MEMORY.md").is_file():
        warnings.append("missing MEMORY.md")
    if not skills:
        warnings.append("missing .codex/skills/*/SKILL.md candidates")
    if not flows:
        warnings.append("missing Docs/flows/*.md flow maps")
    if feature_contract is None:
        warnings.append("missing .codex/agent-workflow/feature-contract-template.md")

    return _RepoContext(
        root=root,
        mandatory=mandatory,
        skills=tuple(skills),
        flows=flows,
        feature_contract=feature_contract,
        warnings=tuple(warnings),
    )


def _select_skills(context: _RepoContext, task_tokens: frozenset[str]) -> tuple[tuple[str, Path], ...]:
    """Select up to three task-relevant skills by name/path tokens."""
    matches = [(name, path) for name, path in context.skills if _matches_task(f"{name} {_relative_path(context.root, path)}", task_tokens)]
    return tuple(matches[:3])


def _select_flows(context: _RepoContext, task_tokens: frozenset[str]) -> tuple[Path, ...]:
    """Select up to two task-relevant flow maps by filename/path tokens."""
    matches = [path for path in context.flows if _matches_task(_relative_path(context.root, path), task_tokens)]
    return tuple(matches[:2])


def _feature_contract_applies(task_tokens: frozenset[str]) -> bool:
    """Return whether the task suggests changed feature or contract behavior."""
    return bool(task_tokens & FEATURE_HINTS)


def _select_context(repo: Path | None, task: str) -> _ContextSelection:
    """Select bounded context rows for the handoff."""
    if repo is None:
        return _ContextSelection(
            selected=("repo not provided; no repo inspection performed",),
            not_selected=(),
        )

    context = _inspect_repo_context(repo)
    task_tokens = _tokens(task)
    selected_skills = _select_skills(context, task_tokens)
    selected_flows = _select_flows(context, task_tokens)
    selected_feature_contract = context.feature_contract if context.feature_contract is not None and _feature_contract_applies(task_tokens) else None

    selected: list[str] = []
    selected.extend(_relative_path(context.root, path) for path in context.mandatory)
    selected.extend(_relative_path(context.root, path) for _, path in selected_skills)
    selected.extend(_relative_path(context.root, path) for path in selected_flows)
    if selected_feature_contract is not None:
        selected.append(_relative_path(context.root, selected_feature_contract))

    not_selected: list[str] = []
    selected_skill_paths = {path for _, path in selected_skills}
    for name, path in context.skills:
        if path not in selected_skill_paths:
            not_selected.append(f"{name}: {_relative_path(context.root, path)} (no task keyword match)")
    selected_flow_paths = set(selected_flows)
    not_selected.extend(f"{_relative_path(context.root, path)} (no task keyword match)" for path in context.flows if path not in selected_flow_paths)
    if context.feature_contract is not None and selected_feature_contract is None:
        not_selected.append(f"{_relative_path(context.root, context.feature_contract)} (task does not suggest feature/contract behavior)")
    not_selected.extend(f"{warning} (expected artifact not found)" for warning in context.warnings)
    return _ContextSelection(selected=tuple(selected), not_selected=tuple(not_selected))


def _render_context_route(selection: _ContextSelection) -> str:
    """Render bounded context routing metadata for the handoff."""
    return f"""Selected Context:
{_one_line_list(selection.selected)}

Not Selected:
{_one_line_list(selection.not_selected)}"""


def _title_from_task(task: str) -> str:
    """Infer a compact prompt title from task text."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", task)
    title = " ".join(words[:8]).strip()
    return title or "Engineering Harness Task"


def _validation_expectations(task: str) -> tuple[str, ...]:
    """Return deterministic validation expectations scaled to task wording."""
    task_tokens = _tokens(task)
    expectations = ["start with read-only inspection of the selected context"]
    if task_tokens & {"review", "drift", "scope"}:
        expectations.append("verify scope drift and missing-file risks before recommending changes")
    if task_tokens & {"ui", "workflow", "flow", "payment", "firma", "contract", "api"}:
        expectations.append("identify the narrow runtime or route path that should be validated")
    if task_tokens & {"test", "bug", "fix", "change", "feature"}:
        expectations.append("propose the smallest relevant test or smoke check")
    else:
        expectations.append("state when no execution is needed beyond source inspection")
    return tuple(expectations)


def _render_codex_prompt(task: str, selection: _ContextSelection) -> str:
    """Render the compact prompt intended for Codex."""
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    return f"""Codex Prompt:
Title: {_title_from_task(task)}
Objective: {task}
Selected context paths:
{_one_line_list(selected_paths)}
Scope boundaries:
- Read selected context first; inspect additional files only when directly required by the task.
- Do not edit files unless the caller explicitly grants editing.
- Do not expand governance, documentation, memory, dashboards, MCP, or autonomous execution.
No-drift rules:
- Stay on the requested repo/workflow surface.
- Preserve existing public APIs and behavior unless the task explicitly asks to change them.
- Do not import assumptions from unrelated repos or broader architecture.
Validation expectations:
{_one_line_list(_validation_expectations(task))}
Mandatory output:
- Files read
- Files changed
- Summary
- Validation
- PASS/FAIL"""


def _contains_section(text: str, section: str) -> bool:
    """Return whether `text` contains a required output section label."""
    return re.search(rf"(^|\n)\s*-?\s*{re.escape(section)}\b", text, flags=re.IGNORECASE) is not None


def _has_pass_or_fail(text: str) -> bool:
    """Return whether `text` contains an explicit PASS or FAIL token."""
    return re.search(r"\b(PASS|FAIL)\b", text, flags=re.IGNORECASE) is not None


def _status_token(text: str) -> str | None:
    """Return the first explicit PASS/FAIL token found."""
    match = re.search(r"\b(PASS|FAIL)\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match is not None else None


def _validation_evidence_present(text: str) -> bool:
    """Return whether a PASS claim includes simple validation evidence."""
    lowered = text.lower()
    if any(term in lowered for term in VALIDATION_NEGATION_TERMS):
        return False
    return any(term in lowered for term in VALIDATION_EVIDENCE_TERMS)


def _review_codex_output(task: str, output: str) -> _ReviewResult:
    """Review pasted Codex output using deterministic text checks."""
    task_tokens = _tokens(task)
    missing = [section for section in REQUIRED_REVIEW_SECTIONS if not _contains_section(output, section)]
    if not _has_pass_or_fail(output):
        missing.append("PASS or FAIL")

    output_tokens = _tokens(output)
    drift_warnings = tuple(f"{term} mentioned without task scope" for term in DRIFT_TERMS if term in output_tokens and term not in task_tokens)

    validation_warnings: list[str] = []
    verdict = _status_token(output)
    if verdict == "PASS" and not _validation_evidence_present(output):
        validation_warnings.append("PASS claimed without validation evidence")

    if verdict == "FAIL":
        status = "FAIL"
    elif missing or drift_warnings or validation_warnings:
        status = "REVIEW_NEEDED"
    else:
        status = "PASS"

    follow_up = None
    if status != "PASS":
        follow_up = (
            "Revise the Codex output to include Files read, Files changed, Summary, Validation, "
            "and PASS/FAIL; remove unsupported scope drift; include concrete validation evidence before PASS."
        )
    return _ReviewResult(
        status=status,
        missing_sections=tuple(missing),
        drift_warnings=drift_warnings,
        validation_warnings=tuple(validation_warnings),
        follow_up=follow_up,
    )
