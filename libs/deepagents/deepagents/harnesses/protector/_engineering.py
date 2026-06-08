"""Reusable Protector engineering harness helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

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
IMPLEMENTATION_INTENT_TERMS = frozenset(
    {
        "arreglar",
        "bug",
        "corregir",
        "error",
        "falla",
        "fix",
        "regression",
        "reparar",
        "restore",
    }
)
REVIEW_ONLY_TERMS = frozenset(
    {
        "assess",
        "audit",
        "classify",
        "inspect",
        "review",
    }
)
PLANNING_ONLY_TERMS = frozenset(
    {
        "arquitectura",
        "design",
        "plan",
        "propuesta",
        "roadmap",
    }
)
DIAGNOSTIC_BOOTSTRAP_TERMS = frozenset(
    {
        "bootstrap",
        "configure",
        "diagnostic",
        "probe",
    }
)
CONTINUATION_FOLLOWUP_TERMS = frozenset(
    {
        "continuation",
        "follow",
        "follow-up",
    }
)
UI_RUNTIME_BUG_TERMS = frozenset(
    {
        "button",
        "dom",
        "js",
        "modal",
        "razor",
        "spinner",
        "ui",
        "view",
    }
)
PROVIDER_API_BUG_TERMS = frozenset(
    {
        "api",
        "config",
        "config_id",
        "dispatch",
        "payload",
        "provider",
        "set_config",
        "start_signature",
    }
)
STRUCTURED_TASK_HEADINGS = (
    "Title",
    "Objective",
    "State",
    "Current regression",
    "Expected behavior",
    "Scope",
    "Restrictions",
    "Validation",
    "PASS",
)
TaskMode = Literal[
    "implementation_fix",
    "review_only",
    "planning_only",
    "diagnostic_bootstrap",
    "continuation_followup",
    "ui_runtime_bug",
    "provider_api_bug",
]
SCOPE_BOUNDARY_BY_MODE: dict[TaskMode, tuple[str, ...]] = {
    "implementation_fix": (
        "Inspect only enough code to locate the faulty condition, then fix surgically.",
        "Scoped edits are allowed when needed to fix the requested issue.",
        "Preserve unrelated behavior.",
    ),
    "planning_only": (
        "Read selected context first; inspect additional files only when directly required by the task.",
        "Do not implement changes; produce planning/design output only.",
        "Do not turn the plan into code changes.",
    ),
    "diagnostic_bootstrap": (
        "Read selected context first; inspect additional files only when directly required by the task.",
        "Bounded diagnostic execution or configuration changes are allowed only when the task explicitly asks for them.",
        "Do not modify provider/runtime state beyond the requested diagnostic or bootstrap scope.",
    ),
    "continuation_followup": (
        "Inspect only enough code to locate the faulty condition, then fix surgically.",
        "Scoped edits are allowed when needed to fix the requested continuation issue.",
        "Preserve validated previous fixes and honor any Restrictions/PASS boundaries as must-not-touch items.",
    ),
    "ui_runtime_bug": (
        "Inspect only enough code to locate the faulty condition, then fix surgically.",
        "Scoped edits are allowed when needed to fix the requested UI/runtime issue.",
        "Preserve observed UI state and expected UI state; prove the exact hide/render condition.",
    ),
    "provider_api_bug": (
        "Inspect only enough code to locate the faulty condition, then fix surgically.",
        "Scoped edits are allowed only for the targeted provider/API issue.",
        "Preserve working provider, config, payload, and dispatch behavior unless the task specifically targets it.",
    ),
    "review_only": (
        "Read selected context first; inspect additional files only when directly required by the task.",
        "Do not edit files unless explicitly requested by this task.",
        "Return findings and risks without implementing changes.",
    ),
}
MODE_REQUIREMENTS_BY_MODE: dict[TaskMode, tuple[str, ...]] = {
    "ui_runtime_bug": (
        "Preserve observed UI state and expected UI state from the task details.",
        "Prove the exact condition that hides or renders the button/modal/spinner/view before changing it.",
    ),
    "provider_api_bug": (
        "Preserve unrelated working provider/config/payload/dispatch behavior.",
        "Do not modify ConfigId, SET_CONFIG, START_SIGNATURE, payload, or dispatch behavior unless the task specifically targets it.",
    ),
    "continuation_followup": (
        "Preserve validated previous fixes.",
        "When task text names restrictions or must-not-touch items, treat them as hard boundaries.",
    ),
    "diagnostic_bootstrap": (
        "Keep probes/configuration work bounded to the explicit diagnostic/bootstrap request.",
        "Report exact evidence from diagnostics instead of broad source-only guesses.",
    ),
}


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
- Follow the Codex Prompt scope boundaries for read-only versus scoped implementation work.
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
    sections = _parse_task_sections(task)
    explicit_title = sections.get("Title")
    if explicit_title:
        return _compact_title(explicit_title)
    objective = _task_objective(task)
    return _compact_title(objective)


def _compact_title(text: str) -> str:
    """Return a compact title from text."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
    title = " ".join(words[:8]).strip()
    return title or "Engineering Harness Task"


def _task_objective(task: str) -> str:
    """Return a concise objective without leading section-label noise."""
    sections = _parse_task_sections(task)
    objective = sections.get("Objective")
    if objective:
        return _compact_multiline(objective)

    stripped = _strip_heading_prefix(task.strip(), "Objective")
    if stripped != task.strip():
        return _compact_multiline(stripped)

    if sections:
        preamble = _task_preamble(task)
        if preamble:
            return _compact_multiline(preamble)
        for heading in ("Current regression", "Expected behavior"):
            section = sections.get(heading)
            if section:
                return _compact_multiline(section)
    return _compact_multiline(task)


def _compact_multiline(text: str) -> str:
    """Collapse task text to one readable line for title/objective fields."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _strip_heading_prefix(text: str, heading: str) -> str:
    """Strip a leading `<heading>:` prefix from one text block."""
    pattern = rf"^\s*{re.escape(heading)}\s*:?\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)


def _task_preamble(task: str) -> str:
    """Return text before the first structured heading."""
    heading_pattern = "|".join(re.escape(heading) for heading in STRUCTURED_TASK_HEADINGS)
    match = re.search(rf"(?im)^\s*(?:{heading_pattern})\s*:?", task)
    if match is None:
        return ""
    return task[: match.start()].strip()


def _parse_task_sections(task: str) -> dict[str, str]:
    """Parse explicit task sections that should survive prompt rendering."""
    headings = {heading.lower(): heading for heading in STRUCTURED_TASK_HEADINGS}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in task.splitlines():
        heading, inline_body = _section_heading(line, headings)
        if heading is not None:
            current = heading
            sections.setdefault(current, [])
            if inline_body:
                sections[current].append(inline_body)
            continue
        if current is not None:
            sections[current].append(line)
    return {heading: "\n".join(lines).strip() for heading, lines in sections.items() if "\n".join(lines).strip()}


def _section_heading(line: str, headings: dict[str, str]) -> tuple[str | None, str]:
    """Return the canonical structured heading for a line."""
    heading_pattern = "|".join(re.escape(heading) for heading in STRUCTURED_TASK_HEADINGS)
    match = re.match(rf"^\s*({heading_pattern})\s*:?\s*(.*)$", line, flags=re.IGNORECASE)
    if match is None:
        return None, ""
    heading = headings.get(match.group(1).strip().lower())
    if heading is None:
        return None, ""
    return heading, match.group(2).strip()


def _structured_task_details(task: str) -> str | None:
    """Render preserved task sections, excluding `Objective`."""
    sections = _parse_task_sections(task)
    rows: list[str] = []
    for heading in ("State", "Current regression", "Expected behavior", "Scope", "Restrictions", "Validation", "PASS"):
        body = sections.get(heading)
        if body:
            rows.append(f"{heading}:\n{body}")
    if not rows:
        return None
    return "\n\n".join(rows)


def _validation_expectations(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return deterministic validation expectations scaled to task wording."""
    task_tokens = _tokens(task)
    expectations = [_mode_validation_expectation(task_mode)]
    if task_tokens & {"review", "drift", "scope"}:
        expectations.append("verify scope drift and missing-file risks before recommending changes")
    if task_tokens & {"ui", "workflow", "flow", "payment", "firma", "contract", "api"}:
        expectations.append("identify the narrow runtime or route path that should be validated")
    if task_tokens & {"test", "bug", "fix", "change", "feature"}:
        expectations.append("propose the smallest relevant test or smoke check")
    else:
        expectations.append("state when no execution is needed beyond source inspection")
    return tuple(expectations)


def _mode_validation_expectation(task_mode: TaskMode) -> str:
    """Return the first validation expectation for a task mode."""
    if task_mode in {"implementation_fix", "continuation_followup", "ui_runtime_bug", "provider_api_bug"}:
        return "inspect only enough code to locate the faulty condition, then fix surgically"
    if task_mode == "planning_only":
        return "produce a bounded plan only; do not implement changes"
    if task_mode == "diagnostic_bootstrap":
        return "run only bounded diagnostics or configuration checks explicitly requested by the task"
    return "start with read-only inspection of the selected context"


def _has_implementation_intent(task: str) -> bool:
    """Return whether task wording asks Codex to implement a scoped fix."""
    return bool(_tokens(task) & IMPLEMENTATION_INTENT_TERMS)


def _classify_task_mode(task: str) -> TaskMode:
    """Classify task text into a deterministic prompt mode."""
    task_tokens = _tokens(task)
    lowered = task.lower()
    implementation_intent = _has_implementation_intent(task)
    signals = {
        "review_only": bool(task_tokens & REVIEW_ONLY_TERMS) or "analizar sin implementar" in lowered,
        "planning_only": bool(task_tokens & PLANNING_ONLY_TERMS),
        "diagnostic_bootstrap": bool(task_tokens & DIAGNOSTIC_BOOTSTRAP_TERMS) or "set_config" in lowered or "smoke real provider" in lowered,
        "continuation_followup": _has_continuation_followup_intent(task, task_tokens),
        "ui_runtime_bug": bool(task_tokens & UI_RUNTIME_BUG_TERMS),
        "provider_api_bug": bool(task_tokens & PROVIDER_API_BUG_TERMS) or "start_signature" in lowered or "set_config" in lowered,
    }
    ordered_rules: tuple[tuple[bool, TaskMode], ...] = (
        (signals["review_only"] and not implementation_intent, "review_only"),
        (signals["planning_only"] and not implementation_intent, "planning_only"),
        (signals["continuation_followup"], "continuation_followup"),
        (signals["diagnostic_bootstrap"], "diagnostic_bootstrap"),
        (signals["provider_api_bug"] and implementation_intent, "provider_api_bug"),
        (signals["ui_runtime_bug"] and implementation_intent, "ui_runtime_bug"),
        (implementation_intent, "implementation_fix"),
        (signals["provider_api_bug"], "provider_api_bug"),
        (signals["ui_runtime_bug"], "ui_runtime_bug"),
        (signals["planning_only"], "planning_only"),
    )
    for matches, task_mode in ordered_rules:
        if matches:
            return task_mode
    return "review_only"


def _has_continuation_followup_intent(task: str, task_tokens: frozenset[str]) -> bool:
    """Return whether task text asks to continue from previous work."""
    lowered = task.lower()
    sections = _parse_task_sections(task)
    current_regression = sections.get("Current regression", "").lower()
    return (
        bool(task_tokens & CONTINUATION_FOLLOWUP_TERMS)
        or "preserve previous fix" in lowered
        or "regression after fix" in lowered
        or ("after" in _tokens(current_regression) and "fix" in _tokens(current_regression))
    )


def _scope_boundaries(task_mode: TaskMode) -> tuple[str, ...]:
    """Return scope boundaries for the selected task mode."""
    return SCOPE_BOUNDARY_BY_MODE[task_mode]


def _mode_requirements(task_mode: TaskMode) -> tuple[str, ...]:
    """Return extra task-mode-specific requirements."""
    return MODE_REQUIREMENTS_BY_MODE.get(task_mode, ())


def _render_bullets(items: tuple[str, ...]) -> str:
    """Render a bullet list from plain text items."""
    return "\n".join(f"- {item}" for item in items)


def _render_codex_prompt(task: str, selection: _ContextSelection) -> str:
    """Render the compact prompt intended for Codex."""
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    objective = _task_objective(task)
    task_details = _structured_task_details(task)
    task_details_text = "(none)" if task_details is None else task_details
    task_mode = _classify_task_mode(task)
    mode_requirements = _mode_requirements(task_mode)
    mode_requirements_text = _render_bullets(mode_requirements) if mode_requirements else "- (none)"
    return f"""Codex Prompt:
Title: {_title_from_task(task)}
Task mode: {task_mode}
Objective: {objective}
Task details:
{task_details_text}
Selected context paths:
{_one_line_list(selected_paths)}
Scope boundaries:
{_render_bullets(_scope_boundaries(task_mode))}
No-drift rules:
- Stay on the requested repo/workflow surface.
- Do not broaden repo exploration beyond selected context unless directly needed.
- Preserve existing public APIs and behavior unless the task explicitly asks to change them.
- Do not import assumptions from unrelated repos or broader architecture.
- Do not expand governance, documentation, memory, dashboards, MCP, or autonomous execution.
- Do not redesign unrelated workflows.
Mode-specific requirements:
{mode_requirements_text}
Validation expectations:
{_one_line_list(_validation_expectations(task, task_mode))}
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
