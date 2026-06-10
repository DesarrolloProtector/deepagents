"""Reusable Protector engineering harness helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend
from deepagents.harnesses.protector._agentic import build_execution_plan, render_execution_plan as render_agentic_execution_plan
from deepagents.harnesses.protector._ecc import discover_pack_prompt_skill_benchmark_coverage, discover_protector_pack
from deepagents.harnesses.protector._prompt_skills import PromptSkill, available_prompt_skills, select_prompt_skills

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

HARNESS_PROFILE = "protector:engineering-harness"
HARNESS_PROFILE_ENV_VAR = "DEEPAGENTS_ENGINEERING_HARNESS_PROFILE"
OUTCOME_HISTORY_RELATIVE_PATH = Path(".protector-harness") / "outcome-history.jsonl"
OUTCOME_HISTORY_SIGNAL_LIMIT = 5
OUTCOME_LEARNING_SIGNAL_MIN_COUNT = 2
OUTCOME_LEARNING_PATTERN_LIMIT = 140
AUTOMATION_CANDIDATE_SCHEMA_VERSION = "ecc-automation-candidate-v1"
AUTOMATION_CANDIDATE_DECISIONS = frozenset({"automation_ready", "supervised_only", "blocked"})
AUTOMATION_CANDIDATE_REQUIRED_FIELDS = (
    "schema_version",
    "selected_pack",
    "task",
    "selected_context_paths",
    "selected_knowledge",
    "selected_skills",
    "benchmark_coverage",
    "review_contract",
    "learning_signals",
    "adaptive_adjustments",
    "automation_readiness",
    "proposed_codex_prompt",
    "execution_boundaries",
)
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
    "english": ("en", "language", "languages"),
    "en": ("english", "language", "languages"),
    "es": ("spanish", "language", "languages"),
    "firma": ("contract", "signature"),
    "i18n": ("localization", "language", "languages"),
    "idiomas": ("localization", "language", "languages"),
    "language": ("localization", "languages"),
    "languages": ("localization", "language"),
    "localizacion": ("localization", "language", "languages"),
    "localización": ("localization", "language", "languages"),
    "multidioma": ("localization", "language", "languages", "multilingual"),
    "multilingual": ("localization", "language", "languages"),
    "payment": ("pay", "receipt", "receipts"),
    "spanish": ("es", "language", "languages"),
    "traducir": ("localization", "translate", "language", "languages"),
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
OUTCOME_IGNORED_PATH_MARKERS = (
    "/.protector-harness/",
    "/bin/",
    "/connected services/",
    "/docs/",
    "/migrations/",
    "/obj/",
    "candidate.codex-output.txt",
    "candidate.json",
)
OUTCOME_IGNORED_SOURCE_MAP_SUFFIXES = (
    ".css.map",
    ".js.map",
)
OUTCOME_LARGE_GENERATED_LINE_LENGTH = 4000
OUTCOME_LARGE_GENERATED_LINE_MARKERS = (
    '"mappings"',
    "base64,",
    "sourcemappingurl",
    "sourcescontent",
)
_KNOWN_OUTPUT_HEADERS = frozenset(
    {
        "files read",
        "files changed",
        "summary",
        "validation",
        "pass",
        "fail",
        "pass/fail",
    }
)
GLOBAL_VALIDATION_LAW = "Choose the cheapest credible falsifier first; escalate validation only when uncertainty remains and record why."
VDR_REQUIRED_FIELDS = ("uncertainty", "cheapest_falsifier", "escalation_reason")
VALIDATION_ESCALATION_TERMS = (
    "container",
    "deployment",
    "e2e",
    "end-to-end",
    "environment",
    "external service",
    "full suite",
    "integration",
    "network",
    "provider",
    "service",
)
VDR_FIELD_EVIDENCE_TERMS = {
    "uncertainty": ("uncertain", "uncertainty", "unknown", "assumption", "risk"),
    "cheapest_falsifier": ("cheapest falsifier", "smallest", "minimal", "narrowest", "focused", "falsifier", "disprove"),
    "escalation_reason": ("escalation reason", "escalate", "escalation", "because", "needed", "required", "why"),
}
IMPLEMENTATION_INTENT_TERMS = frozenset(
    {
        "arreglar",
        "bug",
        "cambiar",
        "corregir",
        "change",
        "desaparece",
        "desaparecer",
        "eliminar",
        "eliminemos",
        "error",
        "falla",
        "fix",
        "hide",
        "modify",
        "modificar",
        "ocultar",
        "quitar",
        "quitemos",
        "remove",
        "replace",
        "reemplazar",
        "regression",
        "reparar",
        "restore",
        "ajustar",
    }
)
REVIEW_ONLY_TERMS = frozenset(
    {
        "analizar",
        "assess",
        "audit",
        "auditar",
        "classify",
        "evaluar",
        "inspect",
        "inspeccionar",
        "review",
        "revisar",
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
LOCALIZATION_TASK_TERMS = frozenset(
    {
        "en",
        "english",
        "es",
        "i18n",
        "idiomas",
        "language",
        "languages",
        "localizacion",
        "localización",
        "localization",
        "multidioma",
        "multilingual",
        "spanish",
        "traducir",
        "translate",
    }
)
LOCALIZATION_STRONG_TASK_TERMS = LOCALIZATION_TASK_TERMS - frozenset({"en", "es"})
LOCALIZATION_LANGUAGE_PAIR_PHRASES = ("es/en", "en/es", "spanish/english", "english/spanish")
DIAGNOSTIC_BOOTSTRAP_TERMS = frozenset(
    {
        "bootstrap",
        "configure",
        "diagnostic",
        "probe",
    }
)
NEGATIVE_PROVIDER_CONFIG_BOUNDARY_PHRASES = (
    "do not modify config",
    "do not modify configid",
    "do not modify config_id",
    "do not modify set_config",
    "do not change config",
    "do not change configid",
    "do not change config_id",
    "do not change set_config",
    "without touching config",
    "without touching configid",
    "without touching config_id",
    "without touching set_config",
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
        "autofill",
        "autofilled",
        "autofills",
        "button",
        "boton",
        "botón",
        "dom",
        "form",
        "añadir",
        "inputs",
        "js",
        "modal",
        "opcion",
        "opción",
        "razor",
        "spinner",
        "tabla",
        "table",
        "textbox",
        "ui",
        "view",
        "views",
        "vista",
        "vistas",
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
SPANISH_SUMMARY_TERMS = frozenset(
    {
        "añadir",
        "autocompleta",
        "autocompletan",
        "botón",
        "contraseña",
        "crear",
        "cuenta",
        "cuentas",
        "eliminar",
        "eliminemos",
        "entidades",
        "más",
        "opción",
        "quitar",
        "quitemos",
        "tabla",
        "vistas",
        "idiomas",
        "localización",
        "multidioma",
        "traducir",
    }
)
SPANISH_LITERAL_REFERENCES = (
    "FormNewBankDataId",
    "botón Añadir",
    "tabla de cuentas",
    "vistas de creación",
    "crear entidades",
    "contraseña",
    "Email",
    "login",
    "Lleida",
    "Lleida.net",
    "ProviderStatus=Success",
    "ProviderCorrelationId",
    "Ver estado firma",
    "Enviar a firmar",
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
KNOWLEDGE_SECTION_HEADINGS = (
    "Current phase",
    "Current priorities",
    "Authoritative workflows",
    "Protected decisions",
    "Forbidden directions",
    "Known useful routes/views/tests",
)
TaskMode = Literal[
    "implementation_fix",
    "review_only",
    "planning_only",
    "diagnostic_bootstrap",
    "continuation_followup",
    "ui_runtime_bug",
    "ui_visual_microfix",
    "provider_api_bug",
]
TASK_MODE_VALUES: tuple[str, ...] = (
    "implementation_fix",
    "review_only",
    "planning_only",
    "diagnostic_bootstrap",
    "continuation_followup",
    "ui_runtime_bug",
    "ui_visual_microfix",
    "provider_api_bug",
)
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
    "ui_visual_microfix": (
        "Inspect only the named UI file or nearest matching UI surface.",
        "Change only the requested visual markup/classes/icon styling.",
        "Preserve handlers, forms, routes, data binding, authorization, and delete semantics.",
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
    "ui_visual_microfix": (
        "Keep the change local to the named visual element or surface.",
        "Do not inspect navigation, workflow, feature-contract, onboarding, provider, memory, or audit context unless explicitly named by the task.",
        "Do not search candidate.json, candidate.codex-output.txt, .protector-harness, bin, obj, generated Connected Services, migrations, or Docs.",
        "When searching, exclude prompt/log artifacts so the task text cannot match itself.",
    ),
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
class TaskIntakeRefinement:
    """Deterministic task intake normalization for rough operator text."""

    raw_task: str
    evidence_paths: tuple[str, ...]
    evidence_notes: tuple[str, ...]
    status: str
    normalized_task: str
    target_surface: str
    requested_change: str
    observed_state: str
    expected_state: str
    explicit_non_goals: tuple[str, ...]
    preserved_behavior: tuple[str, ...]
    suspected_risk_level: str
    missing_details: tuple[str, ...]


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
class OutcomeSkillSummary:
    """Selected prompt skill recorded in supervised outcome history."""

    name: str
    source: str


@dataclass(frozen=True)
class OutcomeSummary:
    """Structured supervised outcome summary safe for local history."""

    timestamp: str
    task: str
    selected_pack: str
    selected_skills: tuple[OutcomeSkillSummary, ...]
    status: str
    changed_files: tuple[str, ...]
    executed_validations: tuple[str, ...]
    deviations: tuple[str, ...]
    follow_up_prompt: str | None
    suggested_benchmark_additions: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeReport:
    """Supervised Codex outcome report plus machine-readable status."""

    text: str
    status: str
    follow_up: str | None
    summary: OutcomeSummary


@dataclass(frozen=True)
class ReadinessDecisionRecord:
    """Evidence and rationale behind an automation-readiness classification."""

    evidence: dict[str, object]
    remaining_uncertainty: tuple[str, ...]
    automation_rationale: tuple[str, ...]
    supervision_rationale: tuple[str, ...]
    blocking_rationale: tuple[str, ...]
    decision: str


@dataclass(frozen=True)
class AutomationReadinessDecision:
    """Deterministic readiness decision for future supervised automation."""

    classification: str
    reasons: tuple[str, ...]
    recommendation: str
    record: ReadinessDecisionRecord


@dataclass(frozen=True)
class RenderedReviewerPrompt:
    """Codex reviewer prompt plus deterministic findings metadata."""

    prompt: str
    findings: ReviewFindings
    selected_context_count: int


@dataclass(frozen=True)
class RenderedExecutionPlan:
    """Controlled execution-plan output for Operator UI and CLI."""

    text: str
    automation_candidate: dict[str, object]
    profile: str
    agents: tuple[str, ...]
    skills: tuple[str, ...]
    safety_gates: tuple[str, ...]
    selected_context_count: int


@dataclass(frozen=True)
class RenderedAutomationCandidateDryRun:
    """Dry-run import result for one exported ECC automation candidate."""

    text: str
    valid: bool
    decision: str
    validation_errors: tuple[str, ...]


@dataclass(frozen=True)
class PromptBenchmarkExpected:
    """Expected prompt characteristics for one benchmark case."""

    repo_alias: str | None
    task_mode: TaskMode | None
    required_skills: tuple[str, ...]
    forbidden_skills: tuple[str, ...]
    required_prompt_text: tuple[str, ...]
    forbidden_prompt_text: tuple[str, ...]
    english_labels: bool


@dataclass(frozen=True)
class PromptBenchmarkResult:
    """Result of one prompt-quality benchmark case."""

    name: str
    passed: bool
    failures: tuple[str, ...]


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
class _RepoKnowledge:
    """Compact repo knowledge derived from a Protector knowledge file."""

    alias: str
    path: Path
    summary: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ContextSelection:
    """Selected and rejected context display rows."""

    selected: tuple[str, ...]
    not_selected: tuple[str, ...]
    knowledge: tuple[str, ...] = ()


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
    repo_alias: str | None = None,
) -> RenderedOutput:
    """Render a compact read-only handoff payload for Codex."""
    repo_text = str(repo.resolve()) if repo is not None else "(not provided)"
    selection = _select_context(repo, task, repo_alias=repo_alias)
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


def run_prompt_benchmarks(
    *,
    benchmarks_dir: Path | None = None,
    repo: Path | None = None,
    repo_alias: str | None = None,
) -> tuple[PromptBenchmarkResult, ...]:
    """Run prompt-quality benchmarks from fixture directories."""
    root = benchmarks_dir or _default_prompt_benchmarks_dir()
    if not root.is_dir():
        msg = f"prompt benchmark directory does not exist: {root}"
        raise HarnessUsageError(msg)

    cases = tuple(path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir())
    if not cases:
        msg = f"prompt benchmark directory contains no cases: {root}"
        raise HarnessUsageError(msg)

    context_repo = repo if repo is not None else _default_prompt_benchmark_repo()
    return tuple(_run_prompt_benchmark_case(path, context_repo, repo_alias) for path in cases)


def render_prompt_benchmark_report(results: tuple[PromptBenchmarkResult, ...]) -> str:
    """Render prompt benchmark results for the CLI."""
    rows = ["Prompt Benchmark Results"]
    passed_count = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        rows.append(f"{status} {result.name}")
        if result.passed:
            passed_count += 1
            continue
        rows.extend(f"  - {failure}" for failure in result.failures)
    failed_count = len(results) - passed_count
    rows.append(f"Summary: {passed_count} passed, {failed_count} failed")
    return "\n".join(rows)


def refine_operator_task(
    raw_task: str,
    *,
    evidence_paths: tuple[str, ...] = (),
    evidence_notes: tuple[str, ...] = (),
) -> TaskIntakeRefinement:
    """Normalize rough operator task text into bounded deterministic intent."""
    task = " ".join(raw_task.split())
    notes = tuple(" ".join(note.split()) for note in evidence_notes if note.strip())
    evidence_text = " ".join(notes)
    intake_text = " ".join(item for item in (task, evidence_text) if item)
    tokens = _tokens(intake_text)
    target_surface = _task_intake_target_surface(intake_text, tokens)
    requested_change = _task_intake_requested_change(intake_text, tokens)
    observed_state = _task_intake_observed_state(notes)
    expected_state = _task_intake_expected_state(notes)
    missing = _task_intake_missing_details(target_surface, requested_change, tokens)
    status = "needs_clarification" if missing else "ready"
    non_goals = _task_intake_non_goals(tokens)
    preserved = _task_intake_preserved_behavior(tokens)
    risk = _task_intake_risk_level(tokens)
    normalized = _task_intake_normalized_task(
        target_surface=target_surface,
        requested_change=requested_change,
        non_goals=non_goals,
        preserved_behavior=preserved,
        evidence_paths=evidence_paths,
        evidence_notes=notes,
        observed_state=observed_state,
        expected_state=expected_state,
    )
    return TaskIntakeRefinement(
        raw_task=task,
        evidence_paths=evidence_paths,
        evidence_notes=notes,
        status=status,
        normalized_task=normalized,
        target_surface=target_surface,
        requested_change=requested_change,
        observed_state=observed_state,
        expected_state=expected_state,
        explicit_non_goals=non_goals,
        preserved_behavior=preserved,
        suspected_risk_level=risk,
        missing_details=missing,
    )


def _effective_task_for_plan(raw_task: str, refinement: TaskIntakeRefinement | None) -> str:
    """Return the task text used for prompt/candidate generation."""
    if refinement is None or refinement.status != "ready":
        return raw_task
    return refinement.normalized_task


def _task_intake_target_surface(task: str, tokens: frozenset[str]) -> str:
    """Infer a bounded target surface without inventing files."""
    surface = ""
    slash_match = re.search(r"\b([A-Z][A-Za-z0-9]+)/([A-Z][A-Za-z0-9]+)(?:\s+([A-Z][A-Za-z0-9]+))?", task)
    if slash_match:
        suffix = slash_match.group(3) or ""
        surface = f"{slash_match.group(1)}/{slash_match.group(2)}{suffix}"
        if tokens & {"receipt", "receipts"}:
            surface = f"{surface} receipts table"
    elif tokens & {"receipt", "receipts"} and tokens & {"table", "tabla"}:
        surface = "receipts table"
    elif tokens & {"client", "clients"} and tokens & {"table", "tabla"}:
        surface = "clients table"
    elif tokens & {"ui", "view", "views", "razor", "css", "button", "icon"}:
        surface = "targeted UI surface"
    elif " on " in task.lower() or " in " in task.lower():
        surface = "named operator surface from task text"
    return surface


def _task_intake_requested_change(task: str, tokens: frozenset[str]) -> str:
    """Infer the requested change without broadening scope."""
    lowered = task.lower()
    if tokens & {"delete"} and tokens & {"icon", "button", "action"}:
        change = "Update only the delete action icon/button styling."
        if ("taildwind" in lowered or "tailwind" in lowered) and "trash" in lowered:
            change = (
                "Update only the delete action icon/button styling to match the existing Tailwind trash/delete action used by other promoted views."
            )
        elif "taildwind" in lowered or "tailwind" in lowered:
            change = "Update only the delete action icon/button styling to match the existing Tailwind-style delete actions."
        return change
    if tokens & {"css", "style", "styles", "visual"}:
        return "Update only the requested visual styling."
    if tokens & {"icon"}:
        return "Update only the requested icon styling."
    if _has_implementation_intent(task):
        return _task_objective(task)
    return ""


def _task_intake_missing_details(target_surface: str, requested_change: str, tokens: frozenset[str]) -> tuple[str, ...]:
    """Return clarification questions for essential missing intent."""
    missing: list[str] = []
    if not target_surface:
        missing.append("Which target surface, route, view, table, or component should change?")
    if not requested_change:
        missing.append("What specific change should be made?")
    if tokens & {"every", "global"} and not tokens & {"only", "specific", "targeted"}:
        missing.append("Should this apply globally or only to named surfaces?")
    return tuple(missing)


def _task_intake_non_goals(tokens: frozenset[str]) -> tuple[str, ...]:
    """Return explicit non-goals for a normalized intake task."""
    non_goals = [
        "Do not change routes.",
        "Do not change handlers.",
        "Do not change forms.",
        "Do not change table data.",
    ]
    if tokens & {"delete"}:
        non_goals.append("Do not change delete behavior.")
    non_goals.append("Do not change non-target UI.")
    return tuple(non_goals)


def _task_intake_preserved_behavior(tokens: frozenset[str]) -> tuple[str, ...]:
    """Return behavior that must be preserved by the normalized task."""
    preserved = ["Existing navigation, data binding, and submitted actions must keep working."]
    if tokens & {"delete"}:
        preserved.append("Delete action semantics, confirmation, authorization, and handlers must remain unchanged.")
    if tokens & {"table", "tabla", "receipt", "receipts"}:
        preserved.append("Existing table rows, columns, and data values must remain unchanged.")
    return tuple(preserved)


def _task_intake_risk_level(tokens: frozenset[str]) -> str:
    """Return a deterministic suspected risk level."""
    if tokens & {"provider", "api", "database", "migration", "auth", "security"}:
        return "high"
    if tokens & {"route", "workflow", "delete", "form"}:
        return "medium"
    return "low"


def _evidence_note_clauses(evidence_notes: tuple[str, ...]) -> tuple[str, ...]:
    """Split operator evidence notes into compact clauses without analyzing files."""
    clauses: list[str] = []
    for note in evidence_notes:
        clauses.extend(clause.strip(" -") for clause in re.split(r"[.;]\s*", note) if clause.strip(" -"))
    return tuple(clauses)


def _task_intake_observed_state(evidence_notes: tuple[str, ...]) -> str:
    """Return operator-provided observed state from evidence notes."""
    observed = tuple(
        clause
        for clause in _evidence_note_clauses(evidence_notes)
        if any(marker in clause.lower() for marker in ("screenshot shows", "screen shows", "observed", "currently", "is still", "shows"))
    )
    if observed:
        return f"Operator evidence note observation: {'; '.join(observed)}."
    return ""


def _task_intake_expected_state(evidence_notes: tuple[str, ...]) -> str:
    """Return operator-provided expected state from evidence notes."""
    expected = tuple(
        clause
        for clause in _evidence_note_clauses(evidence_notes)
        if any(marker in clause.lower() for marker in ("expected", "should", "match", "used elsewhere"))
    )
    if expected:
        return f"Operator evidence note expectation: {'; '.join(expected)}."
    return ""


def _task_intake_normalized_task(
    *,
    target_surface: str,
    requested_change: str,
    non_goals: tuple[str, ...],
    preserved_behavior: tuple[str, ...],
    evidence_paths: tuple[str, ...],
    evidence_notes: tuple[str, ...],
    observed_state: str,
    expected_state: str,
) -> str:
    """Render the normalized bounded task text."""
    target = target_surface or "the target surface that the operator must clarify"
    change = requested_change or "Clarify the requested change before implementation."
    non_goal_text = " ".join(non_goals)
    preserved_text = " ".join(preserved_behavior)
    note_text = ""
    if observed_state or expected_state:
        note_text = f" {observed_state} {expected_state}".rstrip()
    elif evidence_notes:
        note_text = f" Operator evidence notes are available: {_inline_or_none(evidence_notes)}."
    evidence_text = note_text
    if evidence_paths:
        evidence_text = (f"{evidence_text} Evidence references are available but not analyzed: {_inline_or_none(evidence_paths)}.").strip()
        evidence_text = f" {evidence_text}"
    return f"{change} Target surface: {target}. {non_goal_text} Preserve: {preserved_text}{evidence_text}"


def _task_intake_refinement_payload(refinement: TaskIntakeRefinement | None) -> dict[str, object]:
    """Return a JSON-ready task intake refinement payload."""
    if refinement is None:
        return {
            "enabled": False,
            "status": "not_requested",
        }
    return {
        "enabled": True,
        "status": refinement.status,
        "evidence_paths": refinement.evidence_paths,
        "evidence_notes": refinement.evidence_notes,
        "normalized_task": refinement.normalized_task,
        "target_surface": refinement.target_surface,
        "requested_change": refinement.requested_change,
        "observed_state": refinement.observed_state,
        "expected_state": refinement.expected_state,
        "explicit_non_goals": refinement.explicit_non_goals,
        "preserved_behavior": refinement.preserved_behavior,
        "suspected_risk_level": refinement.suspected_risk_level,
        "missing_details": refinement.missing_details,
    }


def _evidence_handling_rows(evidence_paths: tuple[str, ...], evidence_notes: tuple[str, ...]) -> tuple[str, ...]:
    """Return deterministic evidence-reference handling requirements."""
    if not evidence_paths and not evidence_notes:
        return ("No evidence references were attached.",)
    rows: list[str] = []
    if evidence_paths:
        rows.append("Treat evidence paths as operator-provided references only.")
    if evidence_notes:
        rows.append("Treat evidence notes as operator-provided observations, not inferred facts.")
    if evidence_paths and not evidence_notes:
        rows.append("Evidence paths are attached but not interpreted.")
    rows.append("Do not infer screenshot/image/file contents unless a human or future tool explicitly analyzes them.")
    rows.append("Codex/reviewer must consider these references when validating observed UI or file state.")
    return tuple(rows)


def _render_task_intake_refinement(refinement: TaskIntakeRefinement | None) -> str:
    """Render optional task-intake normalization for `ph plan`."""
    if refinement is None:
        return ""
    return f"""Task Intake Refinement:
Status: {refinement.status}
Raw operator task:
- {refinement.raw_task}
Evidence references:
{_one_line_list(refinement.evidence_paths)}
Operator evidence notes:
{_one_line_list(refinement.evidence_notes)}
Evidence handling:
- Evidence paths are references only; no OCR, image analysis, or file parsing was performed.
- Evidence notes are operator-provided observations, not inferred facts.
Normalized task intent:
- {refinement.normalized_task}
Target surface:
- {refinement.target_surface or "(missing)"}
Requested change:
- {refinement.requested_change or "(missing)"}
Observed state:
- {refinement.observed_state or "(not provided)"}
Expected state:
- {refinement.expected_state or "(not provided)"}
Explicit non-goals:
{_one_line_list(refinement.explicit_non_goals)}
Preserved behavior:
{_one_line_list(refinement.preserved_behavior)}
Suspected risk level:
- {refinement.suspected_risk_level}
Missing details/questions:
{_one_line_list(refinement.missing_details)}
"""


def render_controlled_execution_plan(
    *,
    task: str,
    repo: Path | None,
    repo_alias: str | None = None,
    include_history: bool = False,
    refine_task: bool = False,
    evidence_paths: tuple[str, ...] = (),
    evidence_notes: tuple[str, ...] = (),
) -> RenderedExecutionPlan:
    """Render a controlled multi-agent execution plan without invoking Codex."""
    task_refinement = (
        refine_operator_task(task, evidence_paths=evidence_paths, evidence_notes=evidence_notes)
        if refine_task or evidence_paths or evidence_notes
        else None
    )
    effective_task = _effective_task_for_plan(task, task_refinement)
    task_mode = _classify_task_mode(effective_task)
    selection = _select_context(repo, effective_task, repo_alias=repo_alias, task_mode=task_mode)
    prompt_skills = _selected_prompt_skills(effective_task, task_mode)
    plan = build_execution_plan(task_mode=task_mode, prompt_skills=prompt_skills)
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    codex_prompt = _render_codex_prompt(effective_task, selection)
    learning_signals = _outcome_learning_signals(repo, prompt_skills)
    readiness = _automation_readiness_decision(
        selection=selection,
        prompt_skills=prompt_skills,
        task_mode=task_mode,
        learning_signals=learning_signals,
        task_refinement=task_refinement,
    )
    history_signals = _recent_outcome_signals(repo, prompt_skills) if include_history else ()
    planning_adaptations = _adaptive_planning_adjustments(learning_signals)
    history_section = (
        f"""

Recent outcome signals:
{_one_line_list(history_signals)}

Outcome learning signals:
{_one_line_list(learning_signals)}

Adaptive planning adjustments:
{_one_line_list(planning_adaptations)}"""
        if include_history
        else ""
    )
    automation_candidate = _automation_candidate_payload(
        task=effective_task,
        raw_task=task,
        task_refinement=task_refinement,
        task_mode=task_mode,
        selection=selection,
        prompt_skills=prompt_skills,
        selected_paths=selected_paths,
        codex_prompt=codex_prompt,
        learning_signals=learning_signals,
        planning_adaptations=planning_adaptations,
        readiness=readiness,
    )
    text = f"""{render_agentic_execution_plan(plan)}

Task mode: {task_mode}
{_render_task_intake_refinement(task_refinement)}
Selected context paths:
{_one_line_list(selected_paths)}

Knowledge gates:
{_one_line_list(selection.knowledge)}

{
        _render_ecc_supervised_automation_pilot(
            task=effective_task,
            task_mode=task_mode,
            selection=selection,
            evidence_paths=evidence_paths,
            evidence_notes=evidence_notes,
            prompt_skills=prompt_skills,
            codex_prompt=codex_prompt,
        )
    }

{_render_automation_readiness(readiness)}
{history_section}"""
    return RenderedExecutionPlan(
        text=text,
        automation_candidate=automation_candidate,
        profile=plan.profile.name,
        agents=tuple(agent.name for agent in plan.agents),
        skills=tuple(skill.name for skill in plan.skills),
        safety_gates=plan.safety_gates,
        selected_context_count=_selected_context_count(selection),
    )


def _render_ecc_supervised_automation_pilot(
    *,
    task: str,
    task_mode: TaskMode,
    selection: _ContextSelection,
    evidence_paths: tuple[str, ...],
    evidence_notes: tuple[str, ...],
    prompt_skills: tuple[PromptSkill, ...],
    codex_prompt: str,
) -> str:
    """Render the read-only ECC-supervised Codex handoff pilot."""
    pack = discover_protector_pack(include_benchmarks=True)
    coverage = discover_pack_prompt_skill_benchmark_coverage()
    selected_pack_skills = tuple(skill for skill in prompt_skills if skill.source == "pack")
    selected_runtime_skills = tuple(skill for skill in prompt_skills if skill.source != "pack")
    missing_coverage = tuple(skill.name for skill in selected_pack_skills if not coverage.get(skill.name))
    blockers = _ecc_supervised_blockers(pack, missing_coverage)
    confidence = _benchmark_confidence(pack, missing_coverage)
    return f"""ECC Supervised Automation Pilot:
Concept owner: ECC
Protector role: verified pack inputs and prompt-quality behavior
Pilot mode: read-only planning
Codex execution: disabled
File edits: disabled
Autonomous loops: disabled
Background sessions: disabled

Selected ECC pack:
- Name: {pack.name}
- Path: {pack.path if pack.path is not None else "(not found)"}
- Pack validation: {pack.validation_status}
- Prompt skill validation: {pack.prompt_skills_validation_status}
- Capability validation: {pack.capabilities_validation_status}

Selected skills:
{_render_skill_source_rows(prompt_skills, coverage)}

Knowledge used:
{_one_line_list(_knowledge_used_rows(selection))}

Benchmark confidence:
- Confidence: {confidence}
- Benchmark validation: {pack.benchmark_validation_status}
- Benchmark cases: {pack.benchmark_cases_count}
- Covered selected pack skills: {_comma_or_none(tuple(skill.name for skill in selected_pack_skills if coverage.get(skill.name)))}
- Missing selected pack skill coverage: {_comma_or_none(missing_coverage)}

Proposed Codex prompt:
{codex_prompt}

Review criteria:
{_one_line_list(_ecc_supervised_review_criteria(task_mode, selected_pack_skills, selected_runtime_skills))}

Blockers or missing coverage:
{_one_line_list(blockers)}

{
        _render_ecc_review_contract(
            task=task,
            task_mode=task_mode,
            selection=selection,
            evidence_paths=evidence_paths,
            evidence_notes=evidence_notes,
            prompt_skills=prompt_skills,
            selected_pack_skills=selected_pack_skills,
            selected_runtime_skills=selected_runtime_skills,
            coverage=coverage,
        )
    }"""


def _render_skill_source_rows(skills: tuple[PromptSkill, ...], coverage: dict[str, tuple[str, ...]]) -> str:
    """Render selected skill source and benchmark coverage rows."""
    rows: list[str] = []
    for skill in skills:
        coverage_text = "generic fallback; no pack benchmark required"
        if skill.source == "pack":
            coverage_text = f"covered by {', '.join(coverage.get(skill.name, ())) or 'no benchmark case'}"
        rows.append(f"{skill.name} [{skill.source}] - {coverage_text}")
    return _one_line_list(tuple(rows))


def _render_ecc_review_contract(
    *,
    task: str,
    task_mode: TaskMode,
    selection: _ContextSelection,
    evidence_paths: tuple[str, ...],
    evidence_notes: tuple[str, ...],
    prompt_skills: tuple[PromptSkill, ...],
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
    coverage: dict[str, tuple[str, ...]],
) -> str:
    """Render the ECC-supervised review contract for a planned Codex handoff."""
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    return f"""ECC Review Contract:
Contract owner: ECC
Protector role: pack-backed review discipline and anti-drift criteria
Review execution: manual/supervised only
Git diff inspection: not automatic
Codex execution: disabled

Proposed Codex task:
- {_task_objective(task)}

Selected context paths:
{_one_line_list(selected_paths)}

Evidence references:
{_one_line_list(evidence_paths)}

Operator evidence notes:
{_one_line_list(evidence_notes)}

Evidence handling:
{_one_line_list(_evidence_handling_rows(evidence_paths, evidence_notes))}

Expected implementation areas:
{_one_line_list(_expected_implementation_areas(prompt_skills, task_mode))}

Expected files likely to change:
{_one_line_list(_expected_files_likely_to_change(task, selected_paths))}

Expected validation scope:
{_one_line_list(_expected_review_validation_scope(task, task_mode, prompt_skills))}

Validation Decision Record:
{_one_line_list(_validation_decision_record_rows())}

Benchmark relevance:
{_one_line_list(_benchmark_relevance_rows(selected_pack_skills, coverage))}

Review risks:
{_one_line_list(_review_risk_rows(task_mode, selected_pack_skills, selected_runtime_skills))}

Anti-drift checks:
{_one_line_list(_anti_drift_checks(selection, selected_pack_skills))}

PASS/FAIL criteria:
{_one_line_list(_review_pass_fail_criteria(task_mode, selected_pack_skills))}"""


def _render_automation_readiness(decision: AutomationReadinessDecision) -> str:
    """Render deterministic readiness for future supervised automation."""
    return f"""Automation Readiness:
Classification: {decision.classification}
Decision reasons:
{_one_line_list(decision.reasons)}
Readiness Decision Record:
{_render_readiness_decision_record(decision.record)}
Recommended next step:
- {decision.recommendation}
Automation boundaries:
- Codex execution remains disabled.
- Autonomous loops remain disabled.
- No model calls, workflow engine, or dashboard are introduced."""


def _automation_candidate_payload(
    *,
    task: str,
    raw_task: str,
    task_refinement: TaskIntakeRefinement | None,
    task_mode: TaskMode,
    selection: _ContextSelection,
    prompt_skills: tuple[PromptSkill, ...],
    selected_paths: tuple[str, ...],
    codex_prompt: str,
    learning_signals: tuple[str, ...],
    planning_adaptations: tuple[str, ...],
    readiness: AutomationReadinessDecision,
) -> dict[str, object]:
    """Build a deterministic JSON-ready automation candidate payload."""
    pack = discover_protector_pack(include_benchmarks=True)
    coverage = discover_pack_prompt_skill_benchmark_coverage()
    selected_pack_skills = tuple(skill for skill in prompt_skills if skill.source == "pack")
    selected_runtime_skills = tuple(skill for skill in prompt_skills if skill.source != "pack")
    return {
        "schema_version": "ecc-automation-candidate-v1",
        "selected_pack": {
            "name": pack.name,
            "path": str(pack.path) if pack.path is not None else None,
            "validation_status": pack.validation_status,
            "prompt_skills_validation_status": pack.prompt_skills_validation_status,
            "capabilities_validation_status": pack.capabilities_validation_status,
            "benchmark_validation_status": pack.benchmark_validation_status,
        },
        "task": {
            "summary": _task_objective(task),
            "raw_operator_task": raw_task,
            "mode": task_mode,
            "intake_refinement": _task_intake_refinement_payload(task_refinement),
            "evidence_paths": task_refinement.evidence_paths if task_refinement is not None else (),
            "evidence_notes": task_refinement.evidence_notes if task_refinement is not None else (),
        },
        "selected_context_paths": selected_paths,
        "selected_knowledge": {
            "paths": _selected_knowledge_paths(selection),
            "facts": selection.knowledge,
        },
        "selected_skills": tuple({"name": skill.name, "source": skill.source} for skill in prompt_skills),
        "benchmark_coverage": {skill.name: coverage.get(skill.name, ()) for skill in selected_pack_skills},
        "review_contract": {
            "expected_implementation_areas": _expected_implementation_areas(prompt_skills, task_mode),
            "expected_files_likely_to_change": _expected_files_likely_to_change(task, selected_paths),
            "evidence_references": task_refinement.evidence_paths if task_refinement is not None else (),
            "operator_evidence_notes": task_refinement.evidence_notes if task_refinement is not None else (),
            "evidence_handling": _evidence_handling_rows(
                task_refinement.evidence_paths if task_refinement is not None else (),
                task_refinement.evidence_notes if task_refinement is not None else (),
            ),
            "expected_validation_scope": _expected_review_validation_scope(task, task_mode, prompt_skills),
            "validation_decision_record": _validation_decision_record_payload(),
            "benchmark_relevance": _benchmark_relevance_rows(selected_pack_skills, coverage),
            "review_risks": _review_risk_rows(task_mode, selected_pack_skills, selected_runtime_skills),
            "anti_drift_checks": _anti_drift_checks(selection, selected_pack_skills),
            "pass_fail_criteria": _review_pass_fail_criteria(task_mode, selected_pack_skills),
        },
        "learning_signals": learning_signals,
        "adaptive_adjustments": planning_adaptations,
        "automation_readiness": {
            "classification": readiness.classification,
            "reasons": readiness.reasons,
            "recommended_next_step": readiness.recommendation,
            "readiness_decision_record": _readiness_decision_record_payload(readiness.record),
        },
        "proposed_codex_prompt": codex_prompt,
        "execution_boundaries": {
            "codex_execution": False,
            "model_calls": False,
            "autonomous_loops": False,
            "workflow_engine": False,
        },
    }


def render_automation_candidate_dry_run(candidate: object, *, source: str) -> RenderedAutomationCandidateDryRun:
    """Validate an exported automation candidate without executing Codex or models."""
    schema_errors = _automation_candidate_schema_errors(candidate)
    payload = candidate if isinstance(candidate, dict) else {}
    pack = discover_protector_pack(include_benchmarks=True)
    coverage = discover_pack_prompt_skill_benchmark_coverage()
    selected_skills, skill_errors = _candidate_selected_prompt_skills(payload)
    selected_pack_skills = tuple(skill for skill in selected_skills if skill.source == "pack")
    learning_signals = _candidate_string_sequence(_candidate_field(payload, "learning_signals"))
    selected_knowledge = _candidate_dict_field(payload, "selected_knowledge")
    has_knowledge = bool(_candidate_string_sequence(selected_knowledge.get("paths")) or _candidate_string_sequence(selected_knowledge.get("facts")))
    missing_coverage = tuple(skill.name for skill in selected_pack_skills if not coverage.get(skill.name))
    candidate_readiness = _candidate_dict_field(payload, "automation_readiness")
    candidate_classification = _candidate_string_field(candidate_readiness, "classification")
    recomputed_classification = _automation_readiness_classification(
        pack=pack,
        missing_coverage=missing_coverage,
        selected_pack_skills=selected_pack_skills,
        has_knowledge=has_knowledge,
        problem_signals=_problem_learning_signals(learning_signals),
        task_mode=_candidate_task_mode(payload),
        task_refinement=_candidate_task_refinement(payload),
    )
    validation_errors = tuple(
        _unique_preserve_order(
            (
                *schema_errors,
                *_candidate_pack_validation_errors(payload, pack),
                *skill_errors,
                *_candidate_benchmark_coverage_errors(payload, coverage, selected_pack_skills),
                *_candidate_readiness_validation_errors(
                    payload,
                    candidate_classification=candidate_classification,
                    recomputed_classification=recomputed_classification,
                ),
                *_candidate_execution_boundary_errors(payload),
            )
        )
    )
    decision = _candidate_dry_run_decision(candidate_classification, validation_errors)
    return RenderedAutomationCandidateDryRun(
        text=_render_candidate_dry_run_text(
            payload=payload,
            source=source,
            decision=decision,
            validation_errors=validation_errors,
            pack_valid=not _candidate_pack_validation_errors(payload, pack),
            skills_available=not skill_errors,
            coverage_valid=not _candidate_benchmark_coverage_errors(payload, coverage, selected_pack_skills),
            readiness_explainable=not _candidate_readiness_validation_errors(
                payload,
                candidate_classification=candidate_classification,
                recomputed_classification=recomputed_classification,
            ),
        ),
        valid=not validation_errors,
        decision=decision,
        validation_errors=validation_errors,
    )


def render_candidate_outcome_report(
    *,
    candidate: object,
    candidate_source: str,
    codex_output: str,
    codex_output_source: str,
    repo: Path | None = None,
) -> OutcomeReport:
    """Render a supervised outcome report using an exported candidate as the plan source."""
    dry_run = render_automation_candidate_dry_run(candidate, source=candidate_source)
    if not dry_run.valid:
        msg = f"candidate validation failed: {'; '.join(dry_run.validation_errors)}"
        raise HarnessUsageError(msg)
    payload = candidate if isinstance(candidate, dict) else {}
    task = _candidate_task_summary(payload)
    skills, skill_errors = _candidate_selected_prompt_skills(payload)
    if skill_errors:
        msg = f"candidate skill validation failed: {'; '.join(skill_errors)}"
        raise HarnessUsageError(msg)

    coverage = discover_pack_prompt_skill_benchmark_coverage()
    pack = discover_protector_pack(include_benchmarks=True)
    contract = _candidate_dict_field(payload, "review_contract")
    selected_pack_skills = tuple(skill for skill in skills if skill.source == "pack")
    selected_runtime_skills = tuple(skill for skill in skills if skill.source != "pack")
    expected_files = _candidate_string_sequence(contract.get("expected_files_likely_to_change"))
    evidence_paths = _candidate_evidence_paths(payload)
    evidence_notes = _candidate_evidence_notes(payload)
    expected_validation = _candidate_string_sequence(contract.get("expected_validation_scope"))
    repo_changed_files = _repo_diff_changed_files(repo)
    if _candidate_executor_failed(codex_output, repo_changed_files=repo_changed_files):
        return _render_candidate_executor_failed_outcome(
            payload=payload,
            task=task,
            candidate_source=candidate_source,
            codex_output_source=codex_output_source,
            skills=skills,
            coverage=coverage,
            pack=pack,
            evidence_paths=evidence_paths,
            evidence_notes=evidence_notes,
            expected_files=expected_files,
            expected_validation=expected_validation,
        )

    review_text = _pollution_filtered_codex_output(_codex_output_after_prompt_contract(codex_output))
    actual_changed_files = _first_nonempty_tuple(
        _changed_file_section_items(review_text),
        _changed_files_from_unified_diff(review_text),
        repo_changed_files,
    )
    actual_validation = _first_nonempty_tuple(
        _output_section_items(review_text, "Validation"),
        _validation_items_from_narrative(review_text),
    )
    review = _review_codex_output(task, review_text)
    selected_skill_deviations = _selected_skill_behavior_deviations(review_text, selected_pack_skills, selected_runtime_skills)
    blocker_deviations = _outcome_blocker_deviations(pack, selected_pack_skills, coverage)
    readiness_deviations = _candidate_outcome_readiness_deviations(payload)
    file_deviations = _changed_file_deviations(expected_files, actual_changed_files)
    validation_gaps = _outcome_validation_gaps(expected_validation, actual_validation, review.validation_warnings)
    preliminary_deviations = _unique_preserve_order(
        [*file_deviations, *selected_skill_deviations, *blocker_deviations, *readiness_deviations, *review.drift_warnings]
    )
    pass_fail_gaps = _pass_fail_consistency_gaps(review_text, review, deviations=tuple(preliminary_deviations), validation_gaps=validation_gaps)
    deviations = _unique_preserve_order([*preliminary_deviations, *pass_fail_gaps])
    status = _outcome_status(review, deviations=tuple(deviations), validation_gaps=validation_gaps)
    follow_up = _outcome_follow_up_prompt(task, deviations=tuple(deviations), validation_gaps=validation_gaps, status=status)
    benchmark_additions = _suggested_benchmark_additions(selected_pack_skills, coverage, review_text, deviations)
    summary = OutcomeSummary(
        timestamp=_utc_timestamp(),
        task=_task_objective(task),
        selected_pack=pack.name,
        selected_skills=tuple(OutcomeSkillSummary(name=skill.name, source=skill.source) for skill in skills),
        status=_history_status(status),
        changed_files=actual_changed_files,
        executed_validations=actual_validation,
        deviations=tuple(deviations),
        follow_up_prompt=follow_up,
        suggested_benchmark_additions=_meaningful_benchmark_additions(benchmark_additions),
    )
    text = f"""ECC Supervised Outcome Report
Status: {status}
Codex output: {codex_output_source}
Original planned task:
- {_task_objective(task)}

Selected pack:
- Name: {pack.name}
- Validation: {pack.validation_status}
- Benchmark validation: {pack.benchmark_validation_status}

Candidate source:
- {candidate_source}

Automation readiness:
- Classification: {_candidate_readiness_classification(payload)}
- Recommended next step: {_candidate_readiness_next_step(payload)}

Selected skills:
{_render_skill_source_rows(skills, coverage)}

Knowledge used:
{_one_line_list(_candidate_knowledge_used_rows(payload))}

Evidence references:
{_one_line_list(evidence_paths)}

Operator evidence notes:
{_one_line_list(evidence_notes)}

Expected files likely to change:
{_one_line_list(expected_files)}

Actual files changed:
{_one_line_list(actual_changed_files)}

Deviations from plan:
{_one_line_list(tuple(deviations))}

Expected validation scope:
{_one_line_list(expected_validation)}

Executed validation:
{_one_line_list(actual_validation)}

Validation gaps:
{_one_line_list(validation_gaps)}

PASS/FAIL consistency:
{_one_line_list(pass_fail_gaps)}

Suggested benchmark additions:
{_one_line_list(benchmark_additions)}

Follow-up prompt:
- {follow_up or "(none)"}

Supervision boundaries:
- Codex execution was not invoked.
- Git diffs were not inspected automatically.
- No memory/session persistence was written.
- No autonomous loop was started."""
    return OutcomeReport(text=text, status=status, follow_up=follow_up, summary=summary)


def _render_candidate_executor_failed_outcome(
    *,
    payload: dict[object, object],
    task: str,
    candidate_source: str,
    codex_output_source: str,
    skills: tuple[PromptSkill, ...],
    coverage: dict[str, tuple[str, ...]],
    pack: object,
    evidence_paths: tuple[str, ...],
    evidence_notes: tuple[str, ...],
    expected_files: tuple[str, ...],
    expected_validation: tuple[str, ...],
) -> OutcomeReport:
    """Render a candidate outcome for launcher/local-executor failure."""
    status = "executor_failed"
    validation_gaps = ("executor unavailable / no validation run",)
    follow_up = f"Repair the local candidate executor before rerunning candidate execution for: {_task_objective(task)}."
    summary = OutcomeSummary(
        timestamp=_utc_timestamp(),
        task=_task_objective(task),
        selected_pack=pack.name,
        selected_skills=tuple(OutcomeSkillSummary(name=skill.name, source=skill.source) for skill in skills),
        status=status,
        changed_files=(),
        executed_validations=(),
        deviations=(),
        follow_up_prompt=follow_up,
        suggested_benchmark_additions=(),
    )
    text = f"""ECC Supervised Outcome Report
Status: {status}
Codex output: {codex_output_source}
Original planned task:
- {_task_objective(task)}

Selected pack:
- Name: {pack.name}
- Validation: {pack.validation_status}
- Benchmark validation: {pack.benchmark_validation_status}

Candidate source:
- {candidate_source}

Automation readiness:
- Classification: {_candidate_readiness_classification(payload)}
- Recommended next step: {_candidate_readiness_next_step(payload)}

Selected skills:
{_render_skill_source_rows(skills, coverage)}

Knowledge used:
{_one_line_list(_candidate_knowledge_used_rows(payload))}

Evidence references:
{_one_line_list(evidence_paths)}

Operator evidence notes:
{_one_line_list(evidence_notes)}

Expected files likely to change:
{_one_line_list(expected_files)}

Actual files changed:
{_one_line_list(())}

Deviations from plan:
{_one_line_list(())}

Expected validation scope:
{_one_line_list(expected_validation)}

Executed validation:
{_one_line_list(())}

Validation gaps:
{_one_line_list(validation_gaps)}

PASS/FAIL consistency:
{_one_line_list(())}

Suggested benchmark additions:
{_one_line_list(("(none)",))}

Follow-up prompt:
- {follow_up}

Supervision boundaries:
- Codex execution failed before implementation review.
- Git diffs were not inspected automatically.
- No memory/session persistence was written.
- No autonomous loop was started."""
    return OutcomeReport(text=text, status=status, follow_up=follow_up, summary=summary)


def automation_candidate_approval_sha(candidate: object) -> str:
    """Return the canonical SHA-256 approval token for a candidate."""
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def automation_candidate_readiness_classification(candidate: object) -> str:
    """Return a candidate readiness classification for execution gating."""
    payload = candidate if isinstance(candidate, dict) else {}
    return _candidate_readiness_classification(payload)


def automation_candidate_codex_prompt(candidate: object) -> str:
    """Return the proposed Codex prompt from a candidate."""
    payload = candidate if isinstance(candidate, dict) else {}
    return _candidate_string_field(payload, "proposed_codex_prompt")


def _automation_candidate_schema_errors(candidate: object) -> tuple[str, ...]:
    """Return schema errors for an imported automation candidate."""
    if not isinstance(candidate, dict):
        return ("Candidate JSON must be an object.",)

    errors = [f"Candidate missing required field: {field}" for field in AUTOMATION_CANDIDATE_REQUIRED_FIELDS if field not in candidate]
    if candidate.get("schema_version") != AUTOMATION_CANDIDATE_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {_candidate_display_value(candidate.get('schema_version'))}; expected {AUTOMATION_CANDIDATE_SCHEMA_VERSION}"
        )
    object_fields = (
        "selected_pack",
        "task",
        "selected_knowledge",
        "benchmark_coverage",
        "review_contract",
        "automation_readiness",
        "execution_boundaries",
    )
    errors.extend(
        f"Candidate field must be an object: {field}" for field in object_fields if field in candidate and not isinstance(candidate.get(field), dict)
    )
    sequence_fields = ("selected_context_paths", "selected_skills", "learning_signals", "adaptive_adjustments")
    errors.extend(
        f"Candidate field must be a list: {field}"
        for field in sequence_fields
        if field in candidate and not _candidate_is_sequence(candidate.get(field))
    )
    if "selected_context_paths" in candidate and not _candidate_is_string_sequence(candidate.get("selected_context_paths")):
        errors.append("Candidate field must be a list of strings: selected_context_paths")
    if "proposed_codex_prompt" in candidate and not isinstance(candidate.get("proposed_codex_prompt"), str):
        errors.append("Candidate field must be a string: proposed_codex_prompt")
    errors.extend(_candidate_task_schema_errors(candidate))
    errors.extend(_candidate_selected_skill_schema_errors(candidate))
    errors.extend(_candidate_review_contract_schema_errors(candidate))
    errors.extend(_candidate_boundary_schema_errors(candidate))
    return tuple(errors)


def _candidate_task_schema_errors(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return schema errors for the task block."""
    task = _candidate_dict_field(candidate, "task")
    if not task:
        return ()
    errors: list[str] = []
    if not isinstance(task.get("summary"), str) or not task.get("summary"):
        errors.append("Candidate task.summary must be a non-empty string.")
    if not isinstance(task.get("mode"), str) or task.get("mode") not in TASK_MODE_VALUES:
        errors.append(f"Candidate task.mode is unsupported: {_candidate_display_value(task.get('mode'))}")
    if "evidence_paths" in task and not _candidate_is_string_sequence(task.get("evidence_paths")):
        errors.append("Candidate task.evidence_paths must be a list of strings.")
    if "evidence_notes" in task and not _candidate_is_string_sequence(task.get("evidence_notes")):
        errors.append("Candidate task.evidence_notes must be a list of strings.")
    errors.extend(_candidate_task_intake_schema_errors(task))
    return tuple(errors)


def _candidate_task_intake_schema_errors(task: dict[object, object]) -> tuple[str, ...]:
    """Return schema errors for optional task intake refinement metadata."""
    refinement = task.get("intake_refinement")
    if refinement is None:
        return ()
    if not isinstance(refinement, dict):
        return "Candidate task.intake_refinement must be an object."
    errors: list[str] = []
    if not isinstance(refinement.get("enabled"), bool):
        errors.append("Candidate task.intake_refinement.enabled must be a boolean.")
    if refinement.get("enabled") is not True:
        return tuple(errors)
    errors.extend(_candidate_enabled_task_intake_schema_errors(task, refinement))
    return tuple(errors)


def _candidate_enabled_task_intake_schema_errors(
    task: dict[object, object],
    refinement: dict[object, object],
) -> tuple[str, ...]:
    """Return schema errors for enabled task intake refinement metadata."""
    errors: list[str] = []
    if refinement.get("status") not in {"ready", "needs_clarification"}:
        errors.append("Candidate task.intake_refinement.status must be ready or needs_clarification.")
    errors.extend(
        f"Candidate task.intake_refinement.{field} must be a non-empty string."
        for field in ("normalized_task", "suspected_risk_level")
        if not isinstance(refinement.get(field), str) or not refinement.get(field)
    )
    errors.extend(
        f"Candidate task.intake_refinement.{field} must be a string."
        for field in ("target_surface", "requested_change", "observed_state", "expected_state")
        if not isinstance(refinement.get(field), str)
    )
    errors.extend(
        f"Candidate task.intake_refinement.{field} must be a list of strings."
        for field in ("evidence_paths", "evidence_notes", "explicit_non_goals", "preserved_behavior", "missing_details")
        if not _candidate_is_string_sequence(refinement.get(field))
    )
    if not isinstance(task.get("raw_operator_task"), str) or not task.get("raw_operator_task"):
        errors.append("Candidate task.raw_operator_task must be a non-empty string when intake refinement is enabled.")
    return tuple(errors)


def _candidate_selected_skill_schema_errors(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return schema errors for selected skill rows."""
    value = candidate.get("selected_skills")
    if not _candidate_is_sequence(value):
        return ()
    errors: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            errors.append(f"Candidate selected_skills[{index}] must be an object.")
            continue
        if not isinstance(item.get("name"), str) or not item.get("name"):
            errors.append(f"Candidate selected_skills[{index}].name must be a non-empty string.")
        if item.get("source") not in {"pack", "runtime_generic"}:
            errors.append(f"Candidate selected_skills[{index}].source must be pack or runtime_generic.")
    return tuple(errors)


def _candidate_review_contract_schema_errors(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return schema errors for the review contract block."""
    contract = _candidate_dict_field(candidate, "review_contract")
    if not contract:
        return ()
    fields = (
        "expected_implementation_areas",
        "expected_files_likely_to_change",
        "evidence_references",
        "operator_evidence_notes",
        "evidence_handling",
        "expected_validation_scope",
        "benchmark_relevance",
        "review_risks",
        "anti_drift_checks",
        "pass_fail_criteria",
    )
    errors = [
        f"Candidate review_contract.{field} must be a list of strings." for field in fields if not _candidate_is_string_sequence(contract.get(field))
    ]
    vdr = contract.get("validation_decision_record")
    if not isinstance(vdr, dict):
        errors.append("Candidate review_contract.validation_decision_record must be an object.")
    else:
        errors.extend(
            f"Candidate review_contract.validation_decision_record.{field} must be a non-empty string."
            for field in ("global_validation_law", *VDR_REQUIRED_FIELDS)
            if not isinstance(vdr.get(field), str) or not vdr.get(field)
        )
    return tuple(errors)


def _candidate_boundary_schema_errors(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return schema errors for execution boundaries."""
    boundaries = _candidate_dict_field(candidate, "execution_boundaries")
    if not boundaries:
        return ()
    return tuple(
        f"Candidate execution_boundaries.{field} must be a boolean."
        for field in ("codex_execution", "model_calls", "autonomous_loops", "workflow_engine")
        if not isinstance(boundaries.get(field), bool)
    )


def _candidate_pack_validation_errors(candidate: dict[object, object], pack: object) -> tuple[str, ...]:
    """Return errors when the selected pack no longer matches current evidence."""
    selected_pack = _candidate_dict_field(candidate, "selected_pack")
    if not selected_pack:
        return ()
    errors: list[str] = []
    candidate_name = _candidate_string_field(selected_pack, "name")
    if candidate_name != getattr(pack, "name", ""):
        errors.append(f"Selected pack changed: candidate={candidate_name or '(missing)'}, current={getattr(pack, 'name', '(missing)')}")
    for field in (
        "validation_status",
        "prompt_skills_validation_status",
        "capabilities_validation_status",
        "benchmark_validation_status",
    ):
        candidate_status = _candidate_string_field(selected_pack, field)
        current_status = str(getattr(pack, field, ""))
        if candidate_status and candidate_status != current_status:
            errors.append(f"Selected pack {field} changed: candidate={candidate_status}, current={current_status}")
    if not _pack_ready_for_automation(pack):
        errors.append("Selected pack is not currently valid with passing benchmarks.")
    return tuple(errors)


def _candidate_selected_prompt_skills(candidate: dict[object, object]) -> tuple[tuple[PromptSkill, ...], tuple[str, ...]]:
    """Return selected prompt skills if they are still available."""
    available = {(skill.name, skill.source): skill for skill in available_prompt_skills()}
    selected: list[PromptSkill] = []
    errors: list[str] = []
    value = candidate.get("selected_skills")
    if not _candidate_is_sequence(value):
        return (), ()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _candidate_string_field(item, "name")
        source = _candidate_string_field(item, "source")
        skill = available.get((name, source))
        if skill is None:
            errors.append(f"Selected skill unavailable: {name or '(missing)'} [{source or '(missing)'}]")
            continue
        selected.append(skill)
    return tuple(selected), tuple(errors)


def _candidate_benchmark_coverage_errors(
    candidate: dict[object, object],
    coverage: dict[str, tuple[str, ...]],
    selected_pack_skills: tuple[PromptSkill, ...],
) -> tuple[str, ...]:
    """Return errors when selected pack-skill coverage no longer matches."""
    candidate_coverage = _candidate_dict_field(candidate, "benchmark_coverage")
    errors: list[str] = []
    for skill in selected_pack_skills:
        current_cases = coverage.get(skill.name, ())
        candidate_cases = _candidate_string_sequence(candidate_coverage.get(skill.name))
        if not current_cases:
            errors.append(f"Selected pack skill lacks current benchmark coverage: {skill.name}")
            continue
        if not candidate_cases:
            errors.append(f"Candidate benchmark coverage missing selected pack skill: {skill.name}")
            continue
        if candidate_cases != current_cases:
            errors.append(
                f"Benchmark coverage changed for {skill.name}: candidate={_inline_or_none(candidate_cases)}, current={_inline_or_none(current_cases)}"
            )
    return tuple(errors)


def _candidate_readiness_validation_errors(
    candidate: dict[object, object],
    *,
    candidate_classification: str,
    recomputed_classification: str,
) -> tuple[str, ...]:
    """Return errors when readiness classification is stale or unexplained."""
    readiness = _candidate_dict_field(candidate, "automation_readiness")
    if not readiness:
        return ()
    errors: list[str] = []
    if candidate_classification not in AUTOMATION_CANDIDATE_DECISIONS:
        errors.append(f"Candidate readiness classification is unsupported: {_candidate_display_value(candidate_classification)}")
    elif candidate_classification != recomputed_classification:
        errors.append(f"Readiness classification changed: candidate={candidate_classification}, current={recomputed_classification}")
    if not _candidate_string_sequence(readiness.get("reasons")):
        errors.append("Readiness reasons are missing.")
    if not _candidate_string_field(readiness, "recommended_next_step"):
        errors.append("Readiness recommended next step is missing.")
    errors.extend(_candidate_readiness_decision_record_errors(readiness, candidate_classification))
    return tuple(errors)


def _candidate_readiness_decision_record_errors(readiness: dict[object, object], classification: str) -> tuple[str, ...]:
    """Return schema and explainability errors for an imported RDR."""
    record = _candidate_dict_field(readiness, "readiness_decision_record")
    if not record:
        return ("Readiness Decision Record is missing.",)
    errors: list[str] = []
    if not _candidate_dict_field(record, "evidence"):
        errors.append("Readiness Decision Record evidence is missing.")
    errors.extend(
        f"Readiness Decision Record {field} must be a list of strings."
        for field in ("remaining_uncertainty", "automation_rationale", "supervision_rationale", "blocking_rationale")
        if not _candidate_is_string_sequence(record.get(field))
    )
    decision = _candidate_string_field(record, "decision")
    if not decision:
        errors.append("Readiness Decision Record decision is missing.")
    elif classification and not decision.startswith(classification):
        errors.append("Readiness Decision Record decision does not explain the selected classification.")
    if classification == "automation_ready" and not _candidate_string_sequence(record.get("automation_rationale")):
        errors.append("automation_ready requires RDR automation rationale.")
    if classification == "supervised_only" and not _candidate_string_sequence(record.get("remaining_uncertainty")):
        errors.append("supervised_only requires concrete RDR remaining uncertainty.")
    if classification == "blocked" and not _candidate_string_sequence(record.get("blocking_rationale")):
        errors.append("blocked requires RDR missing evidence.")
    return tuple(errors)


def _candidate_execution_boundary_errors(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return errors if an artifact asks the dry-run importer to execute."""
    boundaries = _candidate_dict_field(candidate, "execution_boundaries")
    if not boundaries:
        return ()
    return tuple(
        f"Execution boundary must remain disabled: {field}"
        for field in ("codex_execution", "model_calls", "autonomous_loops", "workflow_engine")
        if boundaries.get(field) is True
    )


def _candidate_outcome_readiness_deviations(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return readiness blockers that should affect candidate outcome acceptance."""
    classification = _candidate_readiness_classification(candidate)
    if classification == "blocked":
        return ("Candidate readiness classification is blocked.",)
    return ()


def _candidate_dry_run_decision(candidate_classification: str, validation_errors: tuple[str, ...]) -> str:
    """Return the dry-run decision label."""
    if validation_errors:
        return "blocked"
    if candidate_classification == "automation_ready":
        return "would execute"
    if candidate_classification == "supervised_only":
        return "would require supervision"
    return "blocked"


def _candidate_task_summary(candidate: dict[object, object]) -> str:
    """Return the planned task summary from a candidate."""
    return _candidate_string_field(_candidate_dict_field(candidate, "task"), "summary")


def _candidate_readiness_classification(candidate: dict[object, object]) -> str:
    """Return the candidate readiness classification."""
    return _candidate_string_field(_candidate_dict_field(candidate, "automation_readiness"), "classification")


def _candidate_readiness_next_step(candidate: dict[object, object]) -> str:
    """Return the candidate readiness recommendation."""
    return _candidate_string_field(_candidate_dict_field(candidate, "automation_readiness"), "recommended_next_step") or "(missing)"


def _candidate_knowledge_used_rows(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return candidate knowledge rows without reselecting repo context."""
    selected_knowledge = _candidate_dict_field(candidate, "selected_knowledge")
    rows = [f"Knowledge path: {path}" for path in _candidate_string_sequence(selected_knowledge.get("paths"))]
    rows.extend(f"Knowledge fact: {fact}" for fact in _candidate_string_sequence(selected_knowledge.get("facts")))
    return tuple(rows)


def _candidate_evidence_paths(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return candidate evidence references without reading them."""
    task = _candidate_dict_field(candidate, "task")
    evidence = _candidate_string_sequence(task.get("evidence_paths"))
    if evidence:
        return evidence
    refinement = _candidate_dict_field(task, "intake_refinement")
    return _candidate_string_sequence(refinement.get("evidence_paths"))


def _candidate_evidence_notes(candidate: dict[object, object]) -> tuple[str, ...]:
    """Return candidate operator evidence notes without reading evidence files."""
    task = _candidate_dict_field(candidate, "task")
    notes = _candidate_string_sequence(task.get("evidence_notes"))
    if notes:
        return notes
    refinement = _candidate_dict_field(task, "intake_refinement")
    return _candidate_string_sequence(refinement.get("evidence_notes"))


def _render_candidate_dry_run_text(
    *,
    payload: dict[object, object],
    source: str,
    decision: str,
    validation_errors: tuple[str, ...],
    pack_valid: bool,
    skills_available: bool,
    coverage_valid: bool,
    readiness_explainable: bool,
) -> str:
    """Render the candidate import dry-run summary."""
    return f"""ECC Candidate Import Dry Run
Source: {source}
Schema: {_candidate_display_value(payload.get("schema_version"))}
Schema validation: {"PASS" if not validation_errors else "FAIL"}
Decision: {decision}
Dry-run only: no Codex execution, model calls, file edits, autonomous loops, or workflow engine are performed.

Current evidence:
- Selected pack still valid: {_yes_no(value=pack_valid)}
- Selected skills still available: {_yes_no(value=skills_available)}
- Benchmark coverage still valid: {_yes_no(value=coverage_valid)}
- Readiness classification explainable: {_yes_no(value=readiness_explainable)}

Validation findings:
{_one_line_list(validation_errors or ("(none)",))}

Codex prompt preview:
{_candidate_prompt_preview(_candidate_string_field(payload, "proposed_codex_prompt"))}

Review contract summary:
{_render_candidate_review_contract_summary(_candidate_dict_field(payload, "review_contract"))}

Evidence references:
{_one_line_list(_candidate_evidence_paths(payload))}

Operator evidence notes:
{_one_line_list(_candidate_evidence_notes(payload))}

Readiness Decision Record:
{_candidate_rdr_summary(_candidate_dict_field(payload, "automation_readiness"))}

Safety boundaries:
{_render_candidate_safety_boundaries(_candidate_dict_field(payload, "execution_boundaries"))}"""


def _render_candidate_review_contract_summary(contract: dict[object, object]) -> str:
    """Render compact review-contract rows from an imported candidate."""
    return _one_line_list(
        (
            f"Expected implementation areas: {_inline_or_none(_candidate_string_sequence(contract.get('expected_implementation_areas')))}",
            f"Evidence references: {_inline_or_none(_candidate_string_sequence(contract.get('evidence_references')))}",
            f"Operator evidence notes: {_inline_or_none(_candidate_string_sequence(contract.get('operator_evidence_notes')))}",
            f"Evidence handling: {_inline_or_none(_candidate_string_sequence(contract.get('evidence_handling')))}",
            f"Expected validation scope: {_inline_or_none(_candidate_string_sequence(contract.get('expected_validation_scope')))}",
            f"Validation Decision Record: {_candidate_vdr_summary(_candidate_dict_field(contract, 'validation_decision_record'))}",
            f"Benchmark relevance: {_inline_or_none(_candidate_string_sequence(contract.get('benchmark_relevance')))}",
            f"PASS/FAIL criteria: {_inline_or_none(_candidate_string_sequence(contract.get('pass_fail_criteria')))}",
        )
    )


def _candidate_vdr_summary(vdr: dict[object, object]) -> str:
    """Render compact VDR requirements from an imported candidate."""
    if not vdr:
        return "(missing)"
    fields = tuple(field for field in VDR_REQUIRED_FIELDS if _candidate_string_field(vdr, field))
    law = _candidate_string_field(vdr, "global_validation_law")
    if not law and not fields:
        return "(missing)"
    return f"law={law or '(missing)'}; fields={_inline_or_none(fields)}"


def _candidate_rdr_summary(readiness: dict[object, object]) -> str:
    """Render compact RDR decision rows from an imported candidate."""
    record = _candidate_dict_field(readiness, "readiness_decision_record")
    if not record:
        return "(missing)"
    return _candidate_string_field(record, "decision") or "(missing)"


def _render_candidate_safety_boundaries(boundaries: dict[object, object]) -> str:
    """Render enforced dry-run safety boundaries."""
    boundary_rows = (
        f"Codex execution: {_boundary_state(boundaries.get('codex_execution'))}",
        f"Model calls: {_boundary_state(boundaries.get('model_calls'))}",
        "File edits: disabled by dry-run importer",
        f"Autonomous loops: {_boundary_state(boundaries.get('autonomous_loops'))}",
        f"Workflow engine: {_boundary_state(boundaries.get('workflow_engine'))}",
    )
    return _one_line_list(boundary_rows)


def _candidate_prompt_preview(prompt: str) -> str:
    """Return a deterministic short preview of the proposed Codex prompt."""
    lines = prompt.splitlines()
    preview = tuple(lines[:12])
    suffix = ("...",) if len(lines) > len(preview) else ()
    return "\n".join((*preview, *suffix)) if preview else "(missing)"


def _boundary_state(value: object) -> str:
    """Render an execution boundary value."""
    if value is False:
        return "disabled"
    if value is True:
        return "enabled"
    return "missing"


def _candidate_task_mode(candidate: dict[object, object]) -> TaskMode:
    """Return a candidate task mode, falling back to review-only for malformed artifacts."""
    value = _candidate_dict_field(candidate, "task").get("mode")
    if isinstance(value, str) and value in TASK_MODE_VALUES:
        return cast("TaskMode", value)
    return "review_only"


def _candidate_task_refinement(candidate: dict[object, object]) -> TaskIntakeRefinement | None:
    """Return imported task-refinement metadata when present."""
    task = _candidate_dict_field(candidate, "task")
    refinement = _candidate_dict_field(task, "intake_refinement")
    if refinement.get("enabled") is not True:
        return None
    return TaskIntakeRefinement(
        raw_task=_candidate_string_field(task, "raw_operator_task"),
        evidence_paths=_candidate_string_sequence(refinement.get("evidence_paths")),
        evidence_notes=_candidate_string_sequence(refinement.get("evidence_notes")),
        status=_candidate_string_field(refinement, "status") or "needs_clarification",
        normalized_task=_candidate_string_field(refinement, "normalized_task"),
        target_surface=_candidate_string_field(refinement, "target_surface"),
        requested_change=_candidate_string_field(refinement, "requested_change"),
        observed_state=_candidate_string_field(refinement, "observed_state"),
        expected_state=_candidate_string_field(refinement, "expected_state"),
        explicit_non_goals=_candidate_string_sequence(refinement.get("explicit_non_goals")),
        preserved_behavior=_candidate_string_sequence(refinement.get("preserved_behavior")),
        suspected_risk_level=_candidate_string_field(refinement, "suspected_risk_level"),
        missing_details=_candidate_string_sequence(refinement.get("missing_details")),
    )


def _candidate_field(candidate: dict[object, object], key: str) -> object:
    """Read a candidate field without assuming a concrete JSON shape."""
    return candidate.get(key)


def _candidate_dict_field(candidate: dict[object, object], key: str) -> dict[object, object]:
    """Read an object field from a candidate."""
    value = candidate.get(key)
    return value if isinstance(value, dict) else {}


def _candidate_string_field(candidate: dict[object, object], key: str) -> str:
    """Read a string field from a candidate object."""
    value = candidate.get(key)
    return value if isinstance(value, str) else ""


def _candidate_string_sequence(value: object) -> tuple[str, ...]:
    """Read a string list from a candidate field."""
    if not _candidate_is_string_sequence(value):
        return ()
    return tuple(cast("Sequence[str]", value))


def _candidate_is_string_sequence(value: object) -> bool:
    """Return whether a value is a JSON-like string sequence."""
    return _candidate_is_sequence(value) and all(isinstance(item, str) for item in value)


def _candidate_is_sequence(value: object) -> bool:
    """Return whether a value is an imported JSON list or in-memory tuple."""
    return isinstance(value, (list, tuple))


def _candidate_display_value(value: object) -> str:
    """Render a scalar candidate value for diagnostics."""
    if isinstance(value, str) and value:
        return value
    if value is None or value == "":
        return "(missing)"
    return str(value)


def _yes_no(*, value: bool) -> str:
    """Render booleans for dry-run text."""
    return "yes" if value else "no"


def _automation_readiness_decision(
    *,
    selection: _ContextSelection,
    prompt_skills: tuple[PromptSkill, ...],
    task_mode: TaskMode,
    learning_signals: tuple[str, ...],
    task_refinement: TaskIntakeRefinement | None = None,
) -> AutomationReadinessDecision:
    """Classify whether a planned task is ready for future automation."""
    pack = discover_protector_pack(include_benchmarks=True)
    coverage = discover_pack_prompt_skill_benchmark_coverage()
    selected_pack_skills = tuple(skill for skill in prompt_skills if skill.source == "pack")
    selected_runtime_skills = tuple(skill for skill in prompt_skills if skill.source != "pack")
    missing_coverage = tuple(skill.name for skill in selected_pack_skills if not coverage.get(skill.name))
    problem_signals = _problem_learning_signals(learning_signals)
    record = _readiness_decision_record(
        pack=pack,
        missing_coverage=missing_coverage,
        selected_pack_skills=selected_pack_skills,
        selected_runtime_skills=selected_runtime_skills,
        selected_knowledge=selection.knowledge,
        problem_signals=problem_signals,
        task_mode=task_mode,
        task_refinement=task_refinement,
    )
    classification = _rdr_classification(record)
    return AutomationReadinessDecision(
        classification=classification,
        reasons=_readiness_summary_reasons(record),
        recommendation=_automation_readiness_recommendation(classification, record),
        record=record,
    )


def _automation_pack_reasons(pack: object) -> tuple[str, ...]:
    """Return concrete pack validation evidence for readiness."""
    return (
        f"Pack validation: {getattr(pack, 'validation_status', 'unknown')}.",
        f"Pack prompt-skill validation: {getattr(pack, 'prompt_skills_validation_status', 'unknown')}.",
        f"Pack capability validation: {getattr(pack, 'capabilities_validation_status', 'unknown')}.",
        f"Pack benchmark validation: {getattr(pack, 'benchmark_validation_status', 'unknown')}.",
    )


def _readiness_decision_record_payload(record: ReadinessDecisionRecord) -> dict[str, object]:
    """Return a JSON-ready Readiness Decision Record payload."""
    return {
        "evidence": record.evidence,
        "remaining_uncertainty": record.remaining_uncertainty,
        "automation_rationale": record.automation_rationale,
        "supervision_rationale": record.supervision_rationale,
        "blocking_rationale": record.blocking_rationale,
        "decision": record.decision,
    }


def _render_readiness_decision_record(record: ReadinessDecisionRecord) -> str:
    """Render an RDR for the human-readable automation readiness section."""
    evidence = record.evidence
    pack_state = _candidate_dict_field(evidence, "pack_validation_state")
    evidence_rows = (
        f"pack validation state: validation={_candidate_display_value(pack_state.get('validation'))}; "
        f"prompt_skills={_candidate_display_value(pack_state.get('prompt_skills'))}; "
        f"capabilities={_candidate_display_value(pack_state.get('capabilities'))}",
        f"benchmark validation state: {_candidate_display_value(evidence.get('benchmark_validation_state'))}",
        f"selected skills: {_inline_or_none(_candidate_skill_names(evidence.get('selected_skills')))}",
        f"selected knowledge: {_inline_or_none(_candidate_string_sequence(evidence.get('selected_knowledge')))}",
        f"learning signals used: {_inline_or_none(_candidate_string_sequence(evidence.get('learning_signals_used')))}",
        f"scope characteristics: {_inline_or_none(_candidate_string_sequence(evidence.get('scope_characteristics')))}",
    )
    rows = [f"Evidence: {_inline_or_none(evidence_rows)}"]
    rows.append(f"Remaining uncertainty: {_inline_or_none(record.remaining_uncertainty)}")
    rows.append(f"Automation rationale: {_inline_or_none(record.automation_rationale)}")
    rows.append(f"Supervision rationale: {_inline_or_none(record.supervision_rationale)}")
    rows.append(f"Blocking rationale: {_inline_or_none(record.blocking_rationale)}")
    rows.append(f"Decision: {record.decision}")
    return _one_line_list(tuple(rows))


def _candidate_skill_names(value: object) -> tuple[str, ...]:
    """Return `name [source]` rows from JSON-like selected skill evidence."""
    if not _candidate_is_sequence(value):
        return ()
    rows: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _candidate_string_field(item, "name")
        source = _candidate_string_field(item, "source")
        if name:
            rows.append(f"{name} [{source or 'unknown'}]")
    return tuple(rows)


def _readiness_decision_record(
    *,
    pack: object,
    missing_coverage: tuple[str, ...],
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
    selected_knowledge: tuple[str, ...],
    problem_signals: tuple[str, ...],
    task_mode: TaskMode,
    task_refinement: TaskIntakeRefinement | None,
) -> ReadinessDecisionRecord:
    """Build the operational decision record behind automation readiness."""
    scope = _readiness_scope_characteristics(task_mode, selected_pack_skills, selected_runtime_skills, task_refinement)
    automation_rationale = _readiness_automation_rationale(
        pack=pack,
        missing_coverage=missing_coverage,
        selected_pack_skills=selected_pack_skills,
        selected_runtime_skills=selected_runtime_skills,
        selected_knowledge=selected_knowledge,
        problem_signals=problem_signals,
        task_mode=task_mode,
    )
    supervision_rationale = _readiness_supervision_rationale(
        task_mode=task_mode,
        problem_signals=problem_signals,
        task_refinement=task_refinement,
    )
    blocking_rationale = _readiness_blocking_rationale(pack=pack, missing_coverage=missing_coverage, problem_signals=problem_signals)
    uncertainty = _readiness_remaining_uncertainty(task_mode=task_mode, problem_signals=problem_signals, task_refinement=task_refinement)
    evidence: dict[str, object] = {
        "pack_validation_state": {
            "validation": getattr(pack, "validation_status", "unknown"),
            "prompt_skills": getattr(pack, "prompt_skills_validation_status", "unknown"),
            "capabilities": getattr(pack, "capabilities_validation_status", "unknown"),
        },
        "benchmark_validation_state": getattr(pack, "benchmark_validation_status", "unknown"),
        "selected_skills": tuple({"name": skill.name, "source": skill.source} for skill in (*selected_pack_skills, *selected_runtime_skills)),
        "selected_knowledge": selected_knowledge,
        "learning_signals_used": problem_signals or ("No negative learning signals used.",),
        "scope_characteristics": scope,
    }
    classification = _readiness_record_classification(blocking_rationale, supervision_rationale)
    return ReadinessDecisionRecord(
        evidence=evidence,
        remaining_uncertainty=uncertainty,
        automation_rationale=automation_rationale,
        supervision_rationale=supervision_rationale,
        blocking_rationale=blocking_rationale,
        decision=_readiness_decision_text(classification, automation_rationale, supervision_rationale, blocking_rationale),
    )


def _readiness_scope_characteristics(
    task_mode: TaskMode,
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
    task_refinement: TaskIntakeRefinement | None,
) -> tuple[str, ...]:
    """Return deterministic scope characteristics for RDR evidence."""
    rows = [f"Task mode: {task_mode}."]
    if task_mode in {"implementation_fix", "ui_runtime_bug", "continuation_followup", "provider_api_bug"}:
        rows.append("Implementation scope is bounded by the proposed task and review contract.")
    if task_mode in {"review_only", "planning_only", "diagnostic_bootstrap"}:
        rows.append("Task mode requires human supervision before any candidate-backed execution.")
    if selected_pack_skills:
        rows.append(f"Pack specialization selected: {', '.join(skill.name for skill in selected_pack_skills)}.")
    if selected_runtime_skills:
        rows.append(f"Runtime generic skill selected: {', '.join(skill.name for skill in selected_runtime_skills)}.")
    if task_refinement is not None:
        rows.append(f"Task intake refinement status: {task_refinement.status}.")
        rows.append(f"Suspected risk level: {task_refinement.suspected_risk_level}.")
    return tuple(rows)


def _readiness_automation_rationale(
    *,
    pack: object,
    missing_coverage: tuple[str, ...],
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
    selected_knowledge: tuple[str, ...],
    problem_signals: tuple[str, ...],
    task_mode: TaskMode,
) -> tuple[str, ...]:
    """Return reasons that support candidate-backed execution after approval."""
    rows: list[str] = []
    if _pack_ready_for_automation(pack):
        rows.append("Pack, prompt-skill, capability, and benchmark validation are passing.")
    if selected_pack_skills and not missing_coverage:
        rows.append(f"Selected pack skills have benchmark coverage: {', '.join(skill.name for skill in selected_pack_skills)}.")
    if not selected_pack_skills and selected_runtime_skills:
        rows.append("No pack specialization was selected; generic runtime skills are acceptable because no pack-specific evidence is required.")
    if selected_knowledge:
        rows.append(f"Selected knowledge provides {len(selected_knowledge)} compact fact(s).")
    else:
        rows.append("No selected knowledge is required by the current narrow task evidence.")
    if not problem_signals:
        rows.append("No negative learning signals were found.")
    if task_mode in {"implementation_fix", "ui_runtime_bug", "continuation_followup", "provider_api_bug"}:
        rows.append("Task mode is compatible with a single candidate-backed execution after explicit human approval.")
    rows.append("automation_ready means candidate-backed execution after explicit approval, not autonomous execution.")
    return tuple(_unique_preserve_order(rows))


def _readiness_supervision_rationale(
    *,
    task_mode: TaskMode,
    problem_signals: tuple[str, ...],
    task_refinement: TaskIntakeRefinement | None,
) -> tuple[str, ...]:
    """Return concrete reasons supporting supervised_only."""
    rows: list[str] = []
    if task_refinement is not None and task_refinement.status == "needs_clarification":
        rows.append(f"Task intake needs clarification: {_inline_or_none(task_refinement.missing_details)}")
    if task_mode == "review_only":
        rows.append("Review-only task has no approved implementation intent for candidate execution.")
    if task_mode == "planning_only":
        rows.append("Planning-only task must settle the implementation target before candidate execution.")
    if task_mode == "diagnostic_bootstrap":
        rows.append("Diagnostic/bootstrap task may touch external or runtime state and needs human supervision before execution.")
    rows.extend(f"Negative learning signal must be resolved before automation_ready: {signal}" for signal in problem_signals)
    return tuple(_unique_preserve_order(rows))


def _readiness_blocking_rationale(
    *,
    pack: object,
    missing_coverage: tuple[str, ...],
    problem_signals: tuple[str, ...],
) -> tuple[str, ...]:
    """Return concrete missing evidence that blocks candidate-backed execution."""
    rows: list[str] = []
    if getattr(pack, "validation_status", "") != "valid":
        rows.append(f"Pack validation is missing or invalid: {getattr(pack, 'validation_status', 'unknown')}.")
    if getattr(pack, "prompt_skills_validation_status", "") != "valid":
        rows.append(f"Prompt-skill validation is missing or invalid: {getattr(pack, 'prompt_skills_validation_status', 'unknown')}.")
    if getattr(pack, "capabilities_validation_status", "") != "valid":
        rows.append(f"Capability validation is missing or invalid: {getattr(pack, 'capabilities_validation_status', 'unknown')}.")
    if getattr(pack, "benchmark_validation_status", "") != "passing":
        rows.append(f"Benchmark validation is missing or not passing: {getattr(pack, 'benchmark_validation_status', 'unknown')}.")
    if missing_coverage:
        rows.append(f"Selected pack skills missing benchmark coverage: {', '.join(missing_coverage)}.")
    rows.extend(signal for signal in problem_signals if signal.startswith("Skill repeatedly lacks benchmark coverage:"))
    return tuple(_unique_preserve_order(rows))


def _readiness_remaining_uncertainty(
    *,
    task_mode: TaskMode,
    problem_signals: tuple[str, ...],
    task_refinement: TaskIntakeRefinement | None,
) -> tuple[str, ...]:
    """Return concrete unresolved uncertainties for non-ready decisions."""
    rows: list[str] = []
    if task_refinement is not None and task_refinement.status == "needs_clarification":
        rows.extend(task_refinement.missing_details)
    if task_mode == "review_only":
        rows.append("Implementation intent is unresolved because the task asks for review output only.")
    if task_mode == "planning_only":
        rows.append("Implementation target is unresolved because the task asks for planning/design output only.")
    if task_mode == "diagnostic_bootstrap":
        rows.append("Runtime/configuration impact is unresolved until a human approves diagnostic execution boundaries.")
    rows.extend(f"Historical failure must be falsified by a clean supervised outcome: {signal}" for signal in problem_signals)
    return tuple(_unique_preserve_order(rows))


def _readiness_record_classification(blocking_rationale: tuple[str, ...], supervision_rationale: tuple[str, ...]) -> str:
    """Return the classification implied by RDR rationale."""
    if blocking_rationale:
        return "blocked"
    if supervision_rationale:
        return "supervised_only"
    return "automation_ready"


def _rdr_classification(record: ReadinessDecisionRecord) -> str:
    """Return the classification encoded by a readiness decision record."""
    return _readiness_record_classification(record.blocking_rationale, record.supervision_rationale)


def _readiness_decision_text(
    classification: str,
    automation_rationale: tuple[str, ...],
    supervision_rationale: tuple[str, ...],
    blocking_rationale: tuple[str, ...],
) -> str:
    """Return the RDR decision explanation."""
    if classification == "blocked":
        return f"blocked wins because required evidence is missing: {_inline_or_none(blocking_rationale)}"
    if classification == "supervised_only":
        return f"supervised_only wins because concrete uncertainty remains: {_inline_or_none(supervision_rationale)}"
    return f"automation_ready wins because required evidence is present: {_inline_or_none(automation_rationale)}"


def _readiness_summary_reasons(record: ReadinessDecisionRecord) -> tuple[str, ...]:
    """Return compact readiness reasons from the RDR."""
    evidence = record.evidence
    pack_state = _candidate_dict_field(evidence, "pack_validation_state")
    selected_skills = _candidate_skill_names(evidence.get("selected_skills"))
    rows = [
        f"Pack validation: {_candidate_display_value(pack_state.get('validation'))}.",
        f"Pack prompt-skill validation: {_candidate_display_value(pack_state.get('prompt_skills'))}.",
        f"Pack capability validation: {_candidate_display_value(pack_state.get('capabilities'))}.",
        f"Pack benchmark validation: {_candidate_display_value(evidence.get('benchmark_validation_state'))}.",
        f"Selected skills: {_inline_or_none(selected_skills)}.",
    ]
    rows.extend(record.blocking_rationale)
    rows.extend(record.supervision_rationale)
    if not record.blocking_rationale:
        rows.extend(record.automation_rationale)
    return tuple(_unique_preserve_order(rows))


def _automation_readiness_classification(
    *,
    pack: object,
    missing_coverage: tuple[str, ...],
    selected_pack_skills: tuple[PromptSkill, ...],
    has_knowledge: bool,
    problem_signals: tuple[str, ...],
    task_mode: TaskMode,
    task_refinement: TaskIntakeRefinement | None = None,
) -> str:
    """Return automation_ready, supervised_only, or blocked."""
    if not _pack_ready_for_automation(pack):
        return "blocked"
    if missing_coverage or any(signal.startswith("Skill repeatedly lacks benchmark coverage:") for signal in problem_signals):
        return "blocked"
    if task_refinement is not None and task_refinement.status == "needs_clarification":
        return "supervised_only"
    if task_mode in {"review_only", "planning_only", "diagnostic_bootstrap"} or problem_signals:
        return "supervised_only"
    _ = (selected_pack_skills, has_knowledge)
    return "automation_ready"


def _pack_ready_for_automation(pack: object) -> bool:
    """Return whether pack validation evidence is automation-ready."""
    return (
        getattr(pack, "validation_status", "") == "valid"
        and getattr(pack, "prompt_skills_validation_status", "") == "valid"
        and getattr(pack, "capabilities_validation_status", "") == "valid"
        and getattr(pack, "benchmark_validation_status", "") == "passing"
    )


def _automation_readiness_recommendation(
    classification: str,
    record: ReadinessDecisionRecord,
) -> str:
    """Return the next supervised step for the readiness decision."""
    if classification == "automation_ready":
        return "proceed with candidate-backed execution only after explicit human approval; autonomous execution remains disabled."
    if classification == "blocked":
        return f"restore missing readiness evidence first: {_inline_or_none(record.blocking_rationale)}"
    return f"resolve RDR uncertainty before automation_ready: {_inline_or_none(record.remaining_uncertainty)}"


def _problem_learning_signals(learning_signals: tuple[str, ...]) -> tuple[str, ...]:
    """Return learning signals that should affect readiness."""
    return tuple(
        signal
        for signal in learning_signals
        if not signal.startswith(("No outcome history entries", "No recurring outcome learning signals", "History unavailable"))
    )


def _expected_implementation_areas(skills: tuple[PromptSkill, ...], task_mode: TaskMode) -> tuple[str, ...]:
    """Return expected implementation areas from selected pack and fallback skills."""
    rows: list[str] = []
    names = {skill.name for skill in skills}
    if "localization_completion" in names:
        rows.append("Selected UI localization/resources/language-selector surface.")
    if "navigation_surface_convergence" in names or "mvp_surface_completion" in names:
        rows.append("Route, menu, index, dashboard, or promoted operator-view entry points.")
    if "provider_bootstrap_diagnostic" in names or "provider_api_bug" in names:
        rows.append("Provider/API boundary, diagnostic/config evidence, and workflow-state translation.")
    if "operational_workflow_convergence" in names:
        rows.append("Workflow state, action eligibility, and rendered operator action surface.")
    if "global_pattern_change" in names:
        rows.append("Shared or repeated pattern usages named by the task.")
    if "form_security_autofill_bug" in names:
        rows.append("Authentication/form field rendering and browser autofill behavior.")
    if not rows and task_mode != "review_only":
        rows.append("Narrow implementation surface required by the proposed Codex task.")
    if task_mode == "review_only":
        rows.append("Review-only evidence and risk analysis; no implementation area should be changed.")
    return tuple(_unique_preserve_order(rows))


def _expected_files_likely_to_change(task: str, selected_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return likely implementation files without treating guidance as targets."""
    task_tokens = _tokens(task)
    candidates = tuple(
        path
        for path in selected_paths
        if _is_concrete_implementation_path(path) and (_path_is_named_by_task(path, task_tokens) or _route_knowledge_supports_path(path, task_tokens))
    )
    if candidates:
        return candidates[:6]
    return ("Unknown until code inspection",)


def _is_concrete_implementation_path(path: str) -> bool:
    """Return whether a selected context path can be treated as an implementation target."""
    normalized = path.replace("\\", "/").lower()
    if normalized.endswith(("agents.md", "memory.md")):
        return False
    if any(
        part in normalized
        for part in (
            ".codex/",
            ".agents/",
            ".protector-harness/",
            "agent-workflow/",
            "docs/flows/",
            "knowledge/",
            "packs/",
            "skill.md",
            "feature-contract-template.md",
        )
    ):
        return False
    return normalized.endswith(
        (
            ".cs",
            ".cshtml",
            ".css",
            ".html",
            ".js",
            ".json",
            ".razor",
            ".razor.css",
            ".scss",
            ".ts",
            ".tsx",
        )
    )


def _path_is_named_by_task(path: str, task_tokens: frozenset[str]) -> bool:
    """Return whether the task text names a specific selected implementation path."""
    normalized = path.replace("\\", "/").lower()
    pieces = tuple(piece for piece in re.split(r"[/_.-]+", normalized) if len(piece) > 1)
    return bool(task_tokens & set(pieces))


def _route_knowledge_supports_path(path: str, task_tokens: frozenset[str]) -> bool:
    """Return whether route/view-style task terms support a selected implementation path."""
    normalized = path.replace("\\", "/").lower()
    if not any(part in normalized for part in ("/views/", "/pages/", "/components/", "/wwwroot/", "/styles/")):
        return False
    return bool(task_tokens & {"css", "razor", "style", "styles", "ui", "view", "views", "visual"})


def _expected_review_validation_scope(task: str, task_mode: TaskMode, skills: tuple[PromptSkill, ...]) -> tuple[str, ...]:
    """Return expected validation scope for the review contract."""
    if task_mode == "ui_visual_microfix":
        return ("verify the diff/static markup for the targeted visual element; build only if Razor syntax risk exists",)
    rows = list(_validation_expectations(task, task_mode))
    for skill in skills:
        rows.extend(skill.validation_expectations)
    rows.append("Confirm validation is proportional to the changed files and behavior claimed by Codex.")
    return tuple(_unique_preserve_order(rows))


def _validation_decision_record_payload() -> dict[str, str]:
    """Return the required Validation Decision Record fields for candidates."""
    return {
        "global_validation_law": GLOBAL_VALIDATION_LAW,
        "uncertainty": "State what remains uncertain before choosing validation.",
        "cheapest_falsifier": "Name the smallest credible check that could disprove the change.",
        "escalation_reason": "Explain why heavier validation is necessary when escalating beyond the cheapest falsifier.",
    }


def _validation_decision_record_rows() -> tuple[str, ...]:
    """Return review-contract rows for the Validation Decision Record."""
    record = _validation_decision_record_payload()
    return (
        f"Global Validation Law: {record['global_validation_law']}",
        f"VDR uncertainty: {record['uncertainty']}",
        f"VDR cheapest_falsifier: {record['cheapest_falsifier']}",
        f"VDR escalation_reason: {record['escalation_reason']}",
        "Reject validation escalation that lacks VDR evidence.",
    )


def _benchmark_relevance_rows(selected_pack_skills: tuple[PromptSkill, ...], coverage: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return benchmark relevance rows for selected pack skills."""
    if not selected_pack_skills:
        return ("No pack-specific benchmark is selected; rely on generic Protector review discipline.",)
    rows: list[str] = []
    for skill in selected_pack_skills:
        cases = coverage.get(skill.name, ())
        rows.append(f"{skill.name}: {', '.join(cases) if cases else 'missing benchmark coverage'}")
    return tuple(rows)


def _review_risk_rows(
    task_mode: TaskMode,
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
) -> tuple[str, ...]:
    """Return review risks from selected skills and task mode."""
    risks = [
        "Codex may broaden the task beyond the proposed prompt or selected pack capability.",
        "Codex may claim PASS without concrete validation evidence.",
    ]
    for skill in selected_pack_skills:
        risks.extend(skill.restriction_rules)
    if selected_runtime_skills:
        risks.append("Generic Protector fallback is present; verify it did not overpower pack-specific specialization.")
    if task_mode == "diagnostic_bootstrap":
        risks.append("Diagnostics may be mistaken for workflow completion.")
    return tuple(_unique_preserve_order(risks))


def _anti_drift_checks(selection: _ContextSelection, selected_pack_skills: tuple[PromptSkill, ...]) -> tuple[str, ...]:
    """Return anti-drift checks for the review contract."""
    checks = [
        "Reject changes outside the proposed Codex task, selected context, or explicitly justified adjacent files.",
        "Reject dashboard/session/workflow-engine/agent-loop additions unless the task explicitly requested them.",
        "Reject raw knowledge dumps; knowledge should stay compact and task-relevant.",
    ]
    if selection.knowledge:
        checks.append("Verify pack knowledge facts were applied only where relevant to the task.")
    for skill in selected_pack_skills:
        checks.extend(skill.forbidden_generic_wording)
    return tuple(_unique_preserve_order(checks))


def _review_pass_fail_criteria(task_mode: TaskMode, selected_pack_skills: tuple[PromptSkill, ...]) -> tuple[str, ...]:
    """Return PASS/FAIL criteria for the review contract."""
    criteria = [
        "PASS only if the Codex output addresses the proposed task and respects every selected pack-skill restriction.",
        "PASS only if validation evidence is concrete, scoped, and matches the claimed behavior.",
        "PASS only if escalated validation includes VDR uncertainty, cheapest_falsifier, and escalation_reason evidence.",
        "FAIL if Codex executes autonomous loops, creates background sessions, or introduces unrelated framework/platform pieces.",
        "FAIL if changed files or behavior drift beyond the task without explicit justification.",
    ]
    if task_mode == "review_only":
        criteria.append("FAIL if Codex implements changes for a review-only task.")
    if selected_pack_skills:
        criteria.append(f"PASS must account for benchmark-covered pack skills: {', '.join(skill.name for skill in selected_pack_skills)}.")
    return tuple(criteria)


def _knowledge_used_rows(selection: _ContextSelection) -> tuple[str, ...]:
    """Return plan rows for selected knowledge paths and compact knowledge facts."""
    rows = [f"Knowledge path: {path}" for path in _selected_knowledge_paths(selection)]
    rows.extend(f"Knowledge fact: {item}" for item in selection.knowledge)
    return tuple(rows)


def _selected_knowledge_paths(selection: _ContextSelection) -> tuple[str, ...]:
    """Return selected pack or legacy knowledge path rows."""
    paths: list[str] = []
    for item in selection.selected:
        normalized = item.replace("\\", "/")
        if "knowledge/" in normalized and (".protector-harness/" in normalized or "protector-financiacioncore/" in normalized):
            paths.append(item)
    return tuple(paths)


def _benchmark_confidence(pack: object, missing_coverage: tuple[str, ...]) -> str:
    """Return a compact confidence label for the supervised pilot."""
    pack_validation = getattr(pack, "validation_status", "")
    benchmark_validation = getattr(pack, "benchmark_validation_status", "")
    if pack_validation == "valid" and benchmark_validation == "passing" and not missing_coverage:
        return "high: verified pack and selected pack skills are benchmark-covered"
    return "reduced: review blockers or missing benchmark coverage before execution"


def _ecc_supervised_blockers(pack: object, missing_coverage: tuple[str, ...]) -> tuple[str, ...]:
    """Return blockers that keep the pilot from future execution."""
    blockers: list[str] = []
    if getattr(pack, "validation_status", "") != "valid":
        blockers.append("Pack validation is not valid.")
    if getattr(pack, "prompt_skills_validation_status", "") != "valid":
        blockers.append("Pack prompt-skill validation is not valid.")
    if getattr(pack, "capabilities_validation_status", "") != "valid":
        blockers.append("Pack capability validation is not valid.")
    if getattr(pack, "benchmark_validation_status", "") != "passing":
        blockers.append("Pack benchmark validation is not passing.")
    if missing_coverage:
        blockers.append(f"Selected pack skills missing benchmark coverage: {', '.join(missing_coverage)}.")
    return tuple(blockers) or ("None for read-only supervised planning.",)


def _ecc_supervised_review_criteria(
    task_mode: TaskMode,
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
) -> tuple[str, ...]:
    """Return review criteria for the supervised Codex handoff."""
    criteria = [
        "Confirm the proposed Codex prompt is task-specific and keeps pack knowledge compact.",
        "Confirm no Codex execution, file edits, autonomous loops, or background sessions are requested by this pilot.",
        "Review any future Codex output against the selected pack skill guardrails and prompt benchmark expectations.",
    ]
    if task_mode != "review_only":
        criteria.append("Require focused validation evidence before accepting PASS from a future Codex run.")
    if selected_pack_skills:
        criteria.append(f"Pack skills under review: {', '.join(skill.name for skill in selected_pack_skills)}.")
    if selected_runtime_skills:
        criteria.append(f"Generic Protector fallback used only where needed: {', '.join(skill.name for skill in selected_runtime_skills)}.")
    return tuple(criteria)


def _comma_or_none(items: tuple[str, ...]) -> str:
    """Render comma-separated names or `(none)`."""
    return ", ".join(items) if items else "(none)"


def _run_prompt_benchmark_case(path: Path, repo: Path | None, repo_alias: str | None) -> PromptBenchmarkResult:
    """Run one prompt benchmark fixture case."""
    task_path = path / "task.txt"
    expected_path = path / "expected_characteristics.md"
    if not task_path.is_file():
        return PromptBenchmarkResult(name=path.name, passed=False, failures=(f"missing fixture file: {task_path.name}",))
    if not expected_path.is_file():
        return PromptBenchmarkResult(name=path.name, passed=False, failures=(f"missing fixture file: {expected_path.name}",))

    task = task_path.read_text(encoding="utf-8").strip()
    expected = _parse_prompt_benchmark_expected(expected_path.read_text(encoding="utf-8"))
    selection = _select_context(repo, task, repo_alias=expected.repo_alias or repo_alias)
    prompt = _render_codex_prompt(task, selection)
    task_mode = _classify_task_mode(task)
    selected_skills = tuple(skill.name for skill in _selected_prompt_skills(task, task_mode))

    failures = _prompt_benchmark_failures(
        prompt=prompt,
        task_mode=task_mode,
        selected_skills=selected_skills,
        expected=expected,
    )
    return PromptBenchmarkResult(name=path.name, passed=not failures, failures=failures)


def _parse_prompt_benchmark_expected(text: str) -> PromptBenchmarkExpected:
    """Parse a Markdown expected-characteristics fixture."""
    sections = _parse_markdown_characteristic_sections(text)
    return PromptBenchmarkExpected(
        repo_alias=_parse_expected_repo_alias(sections.get("repo alias", ())),
        task_mode=_parse_expected_task_mode(sections.get("task mode", ())),
        required_skills=_section_items(sections.get("required skills", ())),
        forbidden_skills=_section_items(sections.get("forbidden skills", ())),
        required_prompt_text=_section_items(sections.get("required prompt text", ())),
        forbidden_prompt_text=_section_items(sections.get("forbidden prompt text", ())),
        english_labels=_section_enabled(sections.get("english labels", ())),
    )


def _parse_markdown_characteristic_sections(text: str) -> dict[str, tuple[str, ...]]:
    """Return `##` sections from a Markdown characteristics file."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(raw.rstrip())
    return {name: tuple(lines) for name, lines in sections.items()}


def _parse_expected_task_mode(lines: tuple[str, ...]) -> TaskMode | None:
    """Parse the expected task mode section."""
    items = _section_items(lines)
    if not items:
        return None
    mode = items[0]
    if mode not in {
        "implementation_fix",
        "review_only",
        "planning_only",
        "diagnostic_bootstrap",
        "continuation_followup",
        "ui_runtime_bug",
        "provider_api_bug",
    }:
        msg = f"unsupported prompt benchmark task mode: {mode}"
        raise HarnessUsageError(msg)
    return cast("TaskMode", mode)


def _parse_expected_repo_alias(lines: tuple[str, ...]) -> str | None:
    """Parse an optional benchmark repo alias section."""
    items = _section_items(lines)
    if not items:
        return None
    return items[0]


def _section_items(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Parse non-empty Markdown section lines as characteristic items."""
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        items.append(line)
    return tuple(items)


def _section_enabled(lines: tuple[str, ...]) -> bool:
    """Return whether a boolean-like section is enabled."""
    items = _section_items(lines)
    if not items:
        return False
    return items[0].lower() not in {"false", "no", "none", "off"}


def _prompt_benchmark_failures(
    *,
    prompt: str,
    task_mode: TaskMode,
    selected_skills: tuple[str, ...],
    expected: PromptBenchmarkExpected,
) -> tuple[str, ...]:
    """Return characteristic failures for a rendered prompt."""
    failures: list[str] = []
    if expected.task_mode is not None and task_mode != expected.task_mode:
        failures.append(f"task mode: expected {expected.task_mode}, got {task_mode}")
    failures.extend(_required_skill_failures(expected.required_skills, selected_skills))
    failures.extend(_forbidden_skill_failures(expected.forbidden_skills, selected_skills))
    if expected.english_labels:
        failures.extend(_english_label_failures(prompt))
    failures.extend(_required_text_failures(expected.required_prompt_text, prompt))
    failures.extend(_forbidden_text_failures(expected.forbidden_prompt_text, prompt))
    return tuple(failures)


def _required_skill_failures(required_skills: tuple[str, ...], selected_skills: tuple[str, ...]) -> tuple[str, ...]:
    """Return missing required skill failures."""
    selected_text = ", ".join(selected_skills)
    return tuple(f"required skill missing: {skill} (selected: {selected_text})" for skill in required_skills if skill not in selected_skills)


def _forbidden_skill_failures(forbidden_skills: tuple[str, ...], selected_skills: tuple[str, ...]) -> tuple[str, ...]:
    """Return selected forbidden skill failures."""
    return tuple(f"forbidden skill selected: {skill}" for skill in forbidden_skills if skill in selected_skills)


def _required_text_failures(required_text: tuple[str, ...], prompt: str) -> tuple[str, ...]:
    """Return missing required prompt text failures."""
    return tuple(f"required prompt text missing: {text}" for text in required_text if text not in prompt)


def _forbidden_text_failures(forbidden_text: tuple[str, ...], prompt: str) -> tuple[str, ...]:
    """Return present forbidden prompt text failures."""
    return tuple(f"forbidden prompt text present: {text}" for text in forbidden_text if text in prompt)


def _english_label_failures(prompt: str) -> tuple[str, ...]:
    """Return failures for required English Codex prompt labels."""
    required = (
        "Title:",
        "Task mode:",
        "Objective:",
        "Task details:",
        "Selected context paths:",
        "Scope boundaries:",
        "No-drift rules:",
        "Mode-specific requirements:",
        "Validation expectations:",
        "Mandatory output:",
    )
    missing = tuple(label for label in required if label not in prompt)
    return tuple(f"English label missing: {label}" for label in missing)


def _default_prompt_benchmarks_dir() -> Path:
    """Find the repository prompt benchmark fixture directory."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tests" / "prompt_benchmarks"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "tests" / "prompt_benchmarks"


def _default_prompt_benchmark_repo() -> Path | None:
    """Return a stable repo root for benchmark context selection when available."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file():
            return parent
    return None


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


def render_supervised_outcome_report(
    *,
    task: str,
    repo: Path | None,
    codex_output: str,
    source: str,
    repo_alias: str | None = None,
) -> OutcomeReport:
    """Render a supervised ECC outcome report without executing Codex or inspecting git."""
    selection = _select_context(repo, task, repo_alias=repo_alias)
    task_mode = _classify_task_mode(task)
    skills = _selected_prompt_skills(task, task_mode)
    coverage = discover_pack_prompt_skill_benchmark_coverage()
    pack = discover_protector_pack(include_benchmarks=True)
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    selected_pack_skills = tuple(skill for skill in skills if skill.source == "pack")
    selected_runtime_skills = tuple(skill for skill in skills if skill.source != "pack")
    expected_files = _expected_files_likely_to_change(task, selected_paths)
    actual_changed_files = _output_section_items(codex_output, "Files changed")
    expected_validation = _expected_review_validation_scope(task, task_mode, skills)
    actual_validation = _output_section_items(codex_output, "Validation")
    review = _review_codex_output(task, codex_output)
    selected_skill_deviations = _selected_skill_behavior_deviations(codex_output, selected_pack_skills, selected_runtime_skills)
    blocker_deviations = _outcome_blocker_deviations(pack, selected_pack_skills, coverage)
    file_deviations = _changed_file_deviations(expected_files, actual_changed_files)
    validation_gaps = _outcome_validation_gaps(expected_validation, actual_validation, review.validation_warnings)
    preliminary_deviations = _unique_preserve_order([*file_deviations, *selected_skill_deviations, *blocker_deviations, *review.drift_warnings])
    pass_fail_gaps = _pass_fail_consistency_gaps(codex_output, review, deviations=tuple(preliminary_deviations), validation_gaps=validation_gaps)
    deviations = _unique_preserve_order([*preliminary_deviations, *pass_fail_gaps])
    status = _outcome_status(review, deviations=tuple(deviations), validation_gaps=validation_gaps)
    follow_up = _outcome_follow_up_prompt(task, deviations=tuple(deviations), validation_gaps=validation_gaps, status=status)
    benchmark_additions = _suggested_benchmark_additions(selected_pack_skills, coverage, codex_output, deviations)
    summary = OutcomeSummary(
        timestamp=_utc_timestamp(),
        task=_task_objective(task),
        selected_pack=pack.name,
        selected_skills=tuple(OutcomeSkillSummary(name=skill.name, source=skill.source) for skill in skills),
        status=_history_status(status),
        changed_files=actual_changed_files,
        executed_validations=actual_validation,
        deviations=tuple(deviations),
        follow_up_prompt=follow_up,
        suggested_benchmark_additions=_meaningful_benchmark_additions(benchmark_additions),
    )
    text = f"""ECC Supervised Outcome Report
Status: {status}
Codex output: {source}
Original planned task:
- {_task_objective(task)}

Selected pack:
- Name: {pack.name}
- Validation: {pack.validation_status}
- Benchmark validation: {pack.benchmark_validation_status}

Selected skills:
{_render_skill_source_rows(skills, coverage)}

Knowledge used:
{_one_line_list(_knowledge_used_rows(selection))}

Expected files likely to change:
{_one_line_list(expected_files)}

Actual files changed:
{_one_line_list(actual_changed_files)}

Deviations from plan:
{_one_line_list(tuple(deviations))}

Expected validation scope:
{_one_line_list(expected_validation)}

Executed validation:
{_one_line_list(actual_validation)}

Validation gaps:
{_one_line_list(validation_gaps)}

PASS/FAIL consistency:
{_one_line_list(pass_fail_gaps)}

Suggested benchmark additions:
{_one_line_list(benchmark_additions)}

Follow-up prompt:
- {follow_up or "(none)"}

Supervision boundaries:
- Codex execution was not invoked.
- Git diffs were not inspected automatically.
- No memory/session persistence was written.
- No autonomous loop was started."""
    return OutcomeReport(text=text, status=status, follow_up=follow_up, summary=summary)


def outcome_history_path(repo: Path) -> Path:
    """Return the repo-scoped supervised outcome history path."""
    return repo / OUTCOME_HISTORY_RELATIVE_PATH


def append_outcome_history(repo: Path, report: OutcomeReport) -> Path:
    """Append one structured supervised outcome summary to repo-local JSONL history."""
    path = outcome_history_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{json.dumps(_outcome_summary_payload(report.summary), sort_keys=True)}\n")
    return path


def load_outcome_history(repo: Path, *, limit: int = 10) -> tuple[OutcomeSummary, ...]:
    """Load recent repo-scoped supervised outcome summaries."""
    path = outcome_history_path(repo)
    if not path.exists():
        return ()
    entries: list[OutcomeSummary] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary = _outcome_summary_from_payload(payload)
        if summary is not None:
            entries.append(summary)
    if limit <= 0:
        return ()
    return tuple(entries[-limit:][::-1])


def render_outcome_history(repo: Path, *, limit: int = 10) -> str:
    """Render recent repo-scoped supervised outcome history."""
    entries = load_outcome_history(repo, limit=limit)
    path = outcome_history_path(repo)
    rows = [
        "ECC Outcome History",
        f"Path: {path}",
        f"Entries shown: {len(entries)}",
    ]
    if not entries:
        rows.append("- (none)")
        return "\n".join(rows)
    for entry in entries:
        rows.append(f"- {entry.timestamp} | {entry.status} | {entry.selected_pack} | {_skill_names(entry.selected_skills)}")
        rows.append(f"  task: {entry.task}")
        rows.append(f"  changed files: {_inline_or_none(entry.changed_files)}")
        rows.append(f"  validations: {_inline_or_none(entry.executed_validations)}")
        rows.append(f"  deviations: {_inline_or_none(entry.deviations)}")
        if entry.follow_up_prompt:
            rows.append(f"  follow-up: {entry.follow_up_prompt}")
        if entry.suggested_benchmark_additions:
            rows.append(f"  benchmark additions: {_inline_or_none(entry.suggested_benchmark_additions)}")
    return "\n".join(rows)


def render_outcome_learning_signals(repo: Path, *, task: str | None = None, limit: int = 50) -> str:
    """Render deterministic learning signals from repo-scoped supervised outcome history."""
    prompt_skills: tuple[PromptSkill, ...] = ()
    if task:
        task_mode = _classify_task_mode(task)
        prompt_skills = _selected_prompt_skills(task, task_mode)
    signals = _outcome_learning_signals(repo, prompt_skills, limit=limit)
    path = outcome_history_path(repo)
    scope = "all pack history" if not prompt_skills else f"selected skills: {_inline_or_none(tuple(skill.name for skill in prompt_skills))}"
    rows = [
        "ECC Outcome Learning Signals",
        f"Path: {path}",
        f"Scope: {scope}",
    ]
    rows.extend(f"- {signal}" for signal in signals)
    return "\n".join(rows)


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


def render_codex_reviewer_prompt(
    *,
    task: str,
    repo: Path | None,
    codex_output: str,
    source: str,
    repo_alias: str | None = None,
) -> RenderedReviewerPrompt:
    """Render a prompt for a separate Codex reviewer without invoking a model."""
    selection = _select_context(repo, task, repo_alias=repo_alias)
    task_mode = _classify_task_mode(task)
    implementation_prompt = _render_codex_prompt(task, selection)
    findings = review_codex_output(task=task, output=codex_output, source=source)
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    reviewer_prompt = f"""Codex Reviewer Prompt

Task mode: {task_mode}

Selected context paths:
{_one_line_list(selected_paths)}

Generated implementation prompt:
{implementation_prompt}

Implementation Codex output:
{codex_output}

Deterministic reviewer findings:
{findings.text}

Reviewer rubric:
- Verify whether the implementation actually satisfies the generated implementation prompt.
- Check scope drift.
- Check whether protected behavior was touched.
- Check whether validation is proportional and credible.
- Check whether PASS is justified.
- Identify concrete missing evidence or follow-up.
- Do not require `Files read` or `Files changed` unless the generated implementation prompt explicitly asks for those sections.
- Do not request broad rewrites.
- Do not introduce new architecture, governance, or documentation.

Produce only:
- Review verdict: PASS / FAIL / NEEDS_FOLLOW_UP
- Findings
- Missing evidence
- Follow-up prompt if needed"""
    return RenderedReviewerPrompt(
        prompt=reviewer_prompt,
        findings=findings,
        selected_context_count=_selected_context_count(selection),
    )


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
    base = {token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(token) > 1}
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


def _select_context(repo: Path | None, task: str, *, repo_alias: str | None = None, task_mode: TaskMode | None = None) -> _ContextSelection:
    """Select bounded context rows for the handoff."""
    if repo is None:
        return _ContextSelection(
            selected=("repo not provided; no repo inspection performed",),
            not_selected=(),
        )

    context = _inspect_repo_context(repo)
    if task_mode == "ui_visual_microfix":
        return _select_visual_microfix_context(context, task)

    task_tokens = _tokens(task)
    selected_skills = _select_skills(context, task_tokens)
    selected_flows = _select_flows(context, task_tokens)
    selected_feature_contract = context.feature_contract if context.feature_contract is not None and _feature_contract_applies(task_tokens) else None
    knowledge = _load_repo_knowledge(context.root, task, repo_alias=repo_alias)

    selected: list[str] = []
    selected.extend(_relative_path(context.root, path) for path in context.mandatory)
    if knowledge is not None:
        selected.append(_relative_path(Path.cwd().resolve(), knowledge.path))
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
    if knowledge is not None:
        not_selected.extend(knowledge.warnings)
    return _ContextSelection(
        selected=tuple(selected),
        not_selected=tuple(not_selected),
        knowledge=knowledge.summary if knowledge is not None else (),
    )


def _select_visual_microfix_context(context: _RepoContext, task: str) -> _ContextSelection:
    """Select only a likely target file for a local visual microfix."""
    target_paths = _visual_microfix_target_paths(context.root, task)
    not_selected = (
        "AGENTS.md (visual microfix clamp: no global instructions needed)",
        "MEMORY.md (visual microfix clamp: no memory read needed)",
        ".codex/skills/*/SKILL.md (visual microfix clamp: no prompt skill routing needed)",
        "Docs/flows/*.md (visual microfix clamp: workflow/navigation context not needed)",
        ".codex/agent-workflow/feature-contract-template.md (visual microfix clamp: no feature contract change)",
    )
    return _ContextSelection(selected=target_paths or ("target file not resolved; use named surface from task",), not_selected=not_selected)


def _visual_microfix_target_paths(repo: Path, task: str) -> tuple[str, ...]:
    """Return likely target files for a visual microfix without broad repository search."""
    candidates = _visual_microfix_candidate_paths(repo, task)
    return tuple(_relative_path(repo, path) for path in candidates if path.is_file())[:3]


def _visual_microfix_candidate_paths(repo: Path, task: str) -> tuple[Path, ...]:
    """Return deterministic candidate paths derived from explicit route/surface text."""
    surfaces = _visual_microfix_surface_candidates(task)
    suffixes = (".razor", ".cshtml", ".razor.css", ".css")
    roots = (
        repo,
        repo / "Components" / "Pages",
        repo / "Pages",
        repo / "Views",
        repo / "Features",
        repo / "Areas",
        repo / "src",
    )
    candidates: list[Path] = []
    for surface in surfaces:
        normalized = surface.replace("\\", "/").strip("/")
        compact = normalized.replace(" ", "")
        pieces = tuple(piece for piece in compact.split("/") if piece)
        variants = tuple(_unique_preserve_order([normalized, compact, "/".join(pieces), "/".join(piece.replace(" ", "") for piece in pieces)]))
        for root in roots:
            for variant in variants:
                candidates.extend(root / f"{variant}{suffix}" for suffix in suffixes)
    return tuple(dict.fromkeys(candidates))


def _visual_microfix_surface_candidates(task: str) -> tuple[str, ...]:
    """Extract route-like UI surface references from visual microfix task text."""
    matches = [match.group(0).replace(" ", "") for match in re.finditer(r"\b[A-Z][A-Za-z0-9]+/[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?\b", task)]
    references = [reference for reference in _literal_references(task) if "/" in reference]
    return tuple(_unique_preserve_order([*matches, *references]))


def _render_context_route(selection: _ContextSelection) -> str:
    """Render bounded context routing metadata for the handoff."""
    return f"""Selected Context:
{_one_line_list(selection.selected)}

Not Selected:
{_one_line_list(selection.not_selected)}"""


def _load_repo_knowledge(repo: Path, task: str, *, repo_alias: str | None) -> _RepoKnowledge | None:
    """Load compact repo knowledge for the resolved repo alias when available."""
    alias_candidates = _repo_knowledge_alias_candidates(repo, repo_alias)
    for candidate in _repo_knowledge_candidates(alias_candidates):
        sections = _parse_repo_knowledge_sections(candidate.path.read_text(encoding="utf-8"))
        summary = _compact_repo_knowledge_summary(sections, task)
        if not summary:
            return None
        warnings = _repo_knowledge_duplicate_warnings(candidate)
        return _RepoKnowledge(alias=candidate.alias, path=candidate.path.resolve(), summary=summary, warnings=warnings)
    return None


def _repo_knowledge_alias_candidates(repo: Path, repo_alias: str | None) -> tuple[str, ...]:
    """Return ordered aliases that may have matching knowledge files."""
    candidates: list[str] = []
    if repo_alias and not any(separator in repo_alias for separator in ("/", "\\")):
        candidates.append(repo_alias)
    candidates.append(repo.name)
    return tuple(_unique_preserve_order(candidates))


def _knowledge_directories() -> tuple[Path, ...]:
    """Return possible Protector knowledge directories without scanning repos."""
    candidates = [base / ".protector-harness" / "knowledge" for base in (Path.cwd().resolve(), *Path(__file__).resolve().parents)]
    return tuple(path for path in _unique_paths(candidates) if path.is_dir())


@dataclass(frozen=True)
class _RepoKnowledgeCandidate:
    """One ordered repo knowledge candidate."""

    alias: str
    path: Path
    source: str
    duplicate: Path | None = None


def _repo_knowledge_candidates(alias_candidates: tuple[str, ...]) -> tuple[_RepoKnowledgeCandidate, ...]:
    """Return pack knowledge candidates before legacy duplicates."""
    candidates: list[_RepoKnowledgeCandidate] = []
    pack_directory = _valid_pack_knowledge_directory()
    legacy_directories = _knowledge_directories()

    if pack_directory is not None:
        for alias in alias_candidates:
            path = pack_directory / f"{alias}.md"
            if path.is_file():
                candidates.append(
                    _RepoKnowledgeCandidate(
                        alias=alias,
                        path=path.resolve(),
                        source="pack",
                        duplicate=_first_legacy_duplicate(alias, legacy_directories),
                    )
                )

    for directory in legacy_directories:
        for alias in alias_candidates:
            path = directory / f"{alias}.md"
            if path.is_file():
                candidates.append(_RepoKnowledgeCandidate(alias=alias, path=path.resolve(), source="legacy"))
    return _dedupe_knowledge_candidates(tuple(candidates))


def _valid_pack_knowledge_directory() -> Path | None:
    """Return the Protector pack knowledge directory when the pack is valid."""
    pack = discover_protector_pack()
    if not pack.found or pack.validation_status != "valid" or pack.path is None:
        return None
    directory = pack.path / "knowledge"
    return directory if directory.is_dir() else None


def _first_legacy_duplicate(alias: str, legacy_directories: tuple[Path, ...]) -> Path | None:
    """Return the first legacy knowledge duplicate for an alias."""
    for directory in legacy_directories:
        path = directory / f"{alias}.md"
        if path.is_file():
            return path.resolve()
    return None


def _repo_knowledge_duplicate_warnings(candidate: _RepoKnowledgeCandidate) -> tuple[str, ...]:
    """Return a warning when pack and legacy knowledge diverge."""
    if candidate.source != "pack" or candidate.duplicate is None:
        return ()
    if candidate.path.read_text(encoding="utf-8") == candidate.duplicate.read_text(encoding="utf-8"):
        return ()
    return (f"pack knowledge differs from legacy duplicate for {candidate.alias}: {candidate.path} vs {candidate.duplicate}",)


def _dedupe_knowledge_candidates(candidates: tuple[_RepoKnowledgeCandidate, ...]) -> tuple[_RepoKnowledgeCandidate, ...]:
    """Return candidates by first alias/source/path occurrence."""
    seen: set[tuple[str, Path]] = set()
    result: list[_RepoKnowledgeCandidate] = []
    for candidate in candidates:
        key = (candidate.alias, candidate.path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return tuple(result)


def _unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    """Return paths by first resolved occurrence."""
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


def _parse_repo_knowledge_sections(text: str) -> dict[str, tuple[str, ...]]:
    """Parse supported repo knowledge sections from Markdown."""
    supported = {heading.lower(): heading for heading in KNOWLEDGE_SECTION_HEADINGS}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            current = supported.get(line[3:].strip().lower())
            if current is not None:
                sections.setdefault(current, [])
            continue
        if current is not None and line:
            sections[current].append(_strip_markdown_bullet(line))
    return {heading: tuple(items) for heading, items in sections.items() if items}


def _strip_markdown_bullet(line: str) -> str:
    """Strip simple Markdown bullet markers from a knowledge line."""
    if line.startswith(("- ", "* ")):
        return line[2:].strip()
    return line


def _compact_repo_knowledge_summary(sections: dict[str, tuple[str, ...]], task: str) -> tuple[str, ...]:
    """Render compact synthesized knowledge without dumping the source file."""
    all_text = " ".join(item for items in sections.values() for item in items).lower()
    task_tokens = _tokens(task)
    summary: list[str] = []
    if "mvp" in all_text or "saas" in all_text:
        summary.append("Align with FinanciacionCore MVP convergence toward a usable SaaS surface.")
    if "contract-first" in all_text:
        summary.append("Preserve contract-first company and financer onboarding as the promoted workflow.")
    if "legacy" in all_text and "direct route" in all_text:
        summary.append("Legacy direct routes may stay backend-compatible, but should not be promoted in normal UI.")
    if "broad audit" in all_text or "broad audits" in all_text:
        summary.append(
            "Avoid broad audits unless the task explicitly requests one. Do not touch Contabilidad, telemetry, or resilience unless targeted."
        )

    priorities = _knowledge_priority_summary(sections.get("Current priorities", ()), task_tokens)
    if priorities is not None:
        summary.append(priorities)
    route_summary = _knowledge_route_summary(sections.get("Known useful routes/views/tests", ()), task_tokens)
    if route_summary is not None:
        summary.append(route_summary)
    if {"contabilidad", "telemetry", "resilience"} & _tokens(all_text) and not any("Contabilidad" in item for item in summary):
        summary.append("Do not touch Contabilidad, telemetry, or resilience unless this task targets them.")
    return tuple(_unique_preserve_order(summary[:6]))


def _knowledge_priority_summary(priorities: tuple[str, ...], task_tokens: frozenset[str]) -> str | None:
    """Return one task-relevant priority line when repo knowledge has one."""
    matched = [priority.rstrip(".") for priority in priorities if _tokens(priority) & task_tokens]
    if matched:
        return f"Relevant repo priority: {'; '.join(matched[:2])}."
    if priorities:
        return "Current repo priorities include legacy onboarding reachability, accounting gaps, visual fixes, languages, telemetry, and resilience."
    return None


def _knowledge_route_summary(routes: tuple[str, ...], task_tokens: frozenset[str]) -> str | None:
    """Return compact route/view/test hints when the task is route or onboarding related."""
    if not routes:
        return None
    if not (task_tokens & {"legacy", "onboarding", "route", "routes", "view", "views", "test", "tests", "workflow"}):
        return None
    return f"Useful route/view/test hints: {'; '.join(route.rstrip('.') for route in routes[:3])}."


def _title_from_task(task: str) -> str:
    """Infer a compact prompt title from task text."""
    if _needs_english_summary(task):
        return _compact_title(_english_task_summary(task, _classify_task_mode(task)))
    sections = _parse_task_sections(task)
    explicit_title = sections.get("Title")
    if explicit_title:
        return _compact_title(explicit_title)
    objective = _task_objective(task)
    return _compact_title(objective)


def _compact_title(text: str) -> str:
    """Return a compact title from text."""
    words = re.findall(r"[\w-]+", text, flags=re.UNICODE)
    title = " ".join(words[:8]).strip()
    return title or "Engineering Harness Task"


def _task_objective(task: str) -> str:
    """Return a concise objective without leading section-label noise."""
    if _needs_english_summary(task):
        return _english_task_summary(task, _classify_task_mode(task))

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


def _task_details(task: str, task_mode: TaskMode) -> str:
    """Render an operational task brief for Codex."""
    if task_mode == "ui_visual_microfix":
        return _visual_microfix_task_details(task)
    if _should_render_operational_brief(task, task_mode):
        return _operational_task_brief(task, task_mode, _selected_prompt_skills(task, task_mode))
    if task_mode == "review_only":
        return f"Review focus: {_task_objective(task)}"
    if task_mode == "planning_only":
        return f"Planning focus: {_task_objective(task)}"
    return f"Task summary: {_task_objective(task)}"


def _should_render_operational_brief(task: str, task_mode: TaskMode) -> bool:
    """Return whether task details should be expanded into operational sections."""
    if task_mode == "ui_visual_microfix":
        return False
    if _parse_task_sections(task):
        return True
    if _needs_english_summary(task) or _has_concrete_ui_or_identifier_details(task):
        return True
    tokens = _tokens(task)
    return task_mode in {"diagnostic_bootstrap", "implementation_fix", "continuation_followup", "ui_runtime_bug", "provider_api_bug"} or bool(
        tokens & {"bug", "regression", "error", "falla"}
    )


def _selected_prompt_skills(task: str, task_mode: TaskMode) -> tuple[PromptSkill, ...]:
    """Return prompt skills that shape Codex prompt rendering."""
    if task_mode == "ui_visual_microfix":
        return select_prompt_skills(
            task_mode=task_mode,
            task_tokens=frozenset(),
            has_spanish_text=False,
            task_text=task,
        )
    return select_prompt_skills(
        task_mode=task_mode,
        task_tokens=_tokens(task),
        has_spanish_text=_needs_english_summary(task),
        task_text=task,
    )


def _operational_task_brief(task: str, task_mode: TaskMode, skills: tuple[PromptSkill, ...]) -> str:
    """Render bug/fix task details in a handoff-style operational brief."""
    sections = _parse_task_sections(task)
    rows = [
        ("Observed state", _brief_observed_state(task, sections, skills)),
        ("Expected behavior", _brief_expected_behavior(task, sections, skills)),
        ("Objective", _task_objective(task)),
        ("Scope", _brief_scope(task, sections, skills)),
        ("Restrictions", _brief_restrictions(task, task_mode, sections, skills)),
        ("Validation", _brief_validation(task, sections, skills)),
    ]
    pass_criteria = sections.get("PASS")
    if pass_criteria:
        rows.append(("PASS criteria", pass_criteria))
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in rows if body)


def _brief_observed_state(task: str, sections: dict[str, str], skills: tuple[PromptSkill, ...]) -> str:
    """Return observed state text for the operational brief."""
    if sections.get("Current regression"):
        return sections["Current regression"]
    if sections.get("State"):
        return sections["State"]

    tokens = _tokens(task)
    observed = "The requested behavior is currently wrong or regressed."
    skill_observed = _first_skill_text(skills, "observed_state")
    if _skill_selected(skills, "form_security_autofill_bug") and skill_observed is not None:
        observed = skill_observed
    elif "spinner" in tokens:
        observed = "The visible UI remains in a spinner/loading state around the provider-hosted send flow."
    elif tokens & {"signature", "firma"} and tokens & {"action", "button", "botón", "resend", "send"}:
        observed = "A signature send/resend action is missing even though the workflow still needs operator action."
    elif skill_observed is not None:
        observed = skill_observed
    return observed


def _brief_expected_behavior(task: str, sections: dict[str, str], skills: tuple[PromptSkill, ...]) -> str:
    """Return expected behavior text for the operational brief."""
    if sections.get("Expected behavior"):
        return sections["Expected behavior"]

    skill_expected = _first_skill_text(skills, "expected_behavior")
    if skill_expected is not None:
        return skill_expected
    if "spinner" in _tokens(task):
        return "The spinner/loading state should clear when the provider-hosted send step reaches its expected terminal UI state."
    return "The scoped workflow behavior should match the requested task without changing unrelated behavior."


def _brief_scope(task: str, sections: dict[str, str], skills: tuple[PromptSkill, ...]) -> str:
    """Return scoped implementation text for the operational brief."""
    if sections.get("Scope"):
        return sections["Scope"]

    tokens = _tokens(task)
    references = _literal_references(task)
    scope = "Inspect only enough code to locate the faulty condition, then fix surgically."
    skill_scope = tuple(rule for skill in skills for rule in skill.scope_rules)
    if skill_scope:
        scope += f" {' '.join(skill_scope)}"
    if references:
        scope += f" Preserve exact references: {', '.join(references)}."
    if tokens & {"workflow", "firma", "signature"}:
        scope += " Do not redesign the workflow."
    return scope


def _brief_restrictions(task: str, task_mode: TaskMode, sections: dict[str, str], skills: tuple[PromptSkill, ...]) -> str:
    """Return restrictions and validated-behavior guardrails for the operational brief."""
    restrictions: list[str] = []
    if sections.get("Restrictions"):
        restrictions.append(sections["Restrictions"])

    restrictions.extend(rule for skill in skills for rule in skill.restriction_rules)
    restrictions.extend(_validated_behavior_guardrails(task, task_mode))
    if not restrictions:
        restrictions.append("Preserve unrelated behavior and public contracts.")
    return _render_rule_lines(tuple(_unique_preserve_order(restrictions)))


def _validated_behavior_guardrails(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return guardrails for behavior that already works and must stay intact."""
    lowered = task.lower()
    guardrails: list[str] = []
    if "regression" in _tokens(task) or task_mode in {"continuation_followup", "provider_api_bug"}:
        guardrails.append("Protect any behavior already validated as working; do not trade one fixed path for another regression.")
    if task_mode == "provider_api_bug" or any(term in lowered for term in ("lleida", "providerstatus", "providercorrelationid")):
        guardrails.append("Provider dispatch success only proves provider acceptance; it does not prove signature completion or workflow completion.")
    if "providerstatus" in lowered or "providercorrelationid" in lowered:
        guardrails.append("Preserve ProviderStatus/ProviderCorrelationId handling that already records successful dispatch.")
    if "ver estado firma" in lowered:
        guardrails.append('Keep "Ver estado firma" working.')
    if "set_config" in lowered or "configid" in lowered or "config_id" in lowered:
        guardrails.append("Do not modify ConfigId or SET_CONFIG handling unless explicitly targeted.")
    if "start_signature" in lowered or "payload" in lowered:
        guardrails.append("Do not modify START_SIGNATURE payloads unless explicitly targeted.")
    return tuple(guardrails)


def _brief_validation(task: str, sections: dict[str, str], skills: tuple[PromptSkill, ...]) -> str:
    """Return proportional validation text for the operational brief."""
    validation: list[str] = []
    if sections.get("Validation"):
        validation.append(sections["Validation"])

    tokens = _tokens(task)
    validation.extend(rule for skill in skills for rule in skill.validation_expectations)
    if tokens & {"signature", "firma"} and tokens & {"action", "button", "botón", "resend", "send"}:
        validation.append("Verify the signature send/resend action renders when the signature remains pending.")
    if not validation:
        validation.append("Run the smallest focused build/test/smoke that proves the requested behavior.")
    return _render_rule_lines(tuple(_unique_preserve_order(validation)))


def _first_skill_text(skills: tuple[PromptSkill, ...], field: str) -> str | None:
    """Return first non-empty text field from selected non-base skills."""
    for skill in skills:
        if skill.name == "base_prompt_quality":
            continue
        value = getattr(skill, field)
        if isinstance(value, str) and value:
            return value
    return None


def _skill_selected(skills: tuple[PromptSkill, ...], name: str) -> bool:
    """Return whether a prompt skill was selected by name."""
    return any(skill.name == name for skill in skills)


def _render_rule_lines(items: tuple[str, ...]) -> str:
    """Render rules as bullets unless caller supplied bullet formatting."""
    return "\n".join(f"- {item}" if not item.lstrip().startswith(("-", "*")) else item for item in items)


def _unique_preserve_order(items: list[str]) -> tuple[str, ...]:
    """Return unique non-empty items while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _validation_expectations(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return deterministic validation expectations scaled to task wording."""
    task_tokens = _tokens(task)
    if task_mode == "ui_visual_microfix":
        return ("verify the diff/static markup for the targeted visual element; build only if Razor syntax risk exists",)
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
    if task_mode == "ui_visual_microfix":
        return "verify the diff/static markup for the targeted visual element; build only if Razor syntax risk exists"
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
        "localization_completion": _has_localization_completion_intent(task),
        "diagnostic_bootstrap": _has_diagnostic_bootstrap_intent(task, task_tokens),
        "continuation_followup": _has_continuation_followup_intent(task, task_tokens),
        "ui_visual_microfix": _has_ui_visual_microfix_intent(task, task_tokens),
        "ui_runtime_bug": bool(task_tokens & UI_RUNTIME_BUG_TERMS),
        "provider_api_bug": bool(task_tokens & PROVIDER_API_BUG_TERMS) or "start_signature" in lowered or "set_config" in lowered,
    }
    ordered_rules: tuple[tuple[bool, TaskMode], ...] = (
        (signals["review_only"] and not implementation_intent, "review_only"),
        (signals["localization_completion"], "implementation_fix"),
        (signals["planning_only"] and not implementation_intent, "planning_only"),
        (signals["diagnostic_bootstrap"], "diagnostic_bootstrap"),
        (signals["continuation_followup"], "continuation_followup"),
        (signals["provider_api_bug"] and implementation_intent, "provider_api_bug"),
        (signals["ui_visual_microfix"] and implementation_intent, "ui_visual_microfix"),
        (signals["ui_runtime_bug"] and implementation_intent, "ui_runtime_bug"),
        (implementation_intent, "implementation_fix"),
        (signals["provider_api_bug"], "provider_api_bug"),
        (signals["ui_visual_microfix"], "ui_visual_microfix"),
        (signals["ui_runtime_bug"], "ui_runtime_bug"),
        (signals["planning_only"], "planning_only"),
    )
    for matches, task_mode in ordered_rules:
        if matches:
            return task_mode
    return "review_only"


def _has_ui_visual_microfix_intent(task: str, task_tokens: frozenset[str]) -> bool:
    """Return whether task text asks for a local visual-only UI edit."""
    _ = task_tokens
    lowered = task.lower()
    positive_text = _visual_microfix_positive_text(task)
    positive_tokens = _tokens(positive_text)
    visual_terms = {"visual", "style", "styling", "css", "class", "classes", "icon", "tailwind", "taildwind", "color", "rounded"}
    ui_terms = {"ui", "view", "vista", "razor", "button", "boton", "botón", "icon", "table", "clients", "client", "receipts"}
    behavior_terms = {"workflow", "route", "routes", "navigation", "runtime", "handler", "form", "forms", "eligibility", "state"}
    provider_terms = PROVIDER_API_BUG_TERMS | {"provider", "signature", "firma", "set_config", "start_signature"}
    if positive_tokens & (behavior_terms | provider_terms | LOCALIZATION_TASK_TERMS):
        return False
    if any(phrase in lowered for phrase in ("visual-only", "visual only", "styling only", "style only", "icon only")):
        return True
    if positive_tokens & visual_terms and positive_tokens & ui_terms:
        return True
    return bool(
        positive_tokens & {"delete", "trash"}
        and positive_tokens & {"icon", "button"}
        and positive_tokens & {"tailwind", "taildwind", "style", "styling"}
    )


def _visual_microfix_positive_text(task: str) -> str:
    """Return task text excluding preservation/non-goal clauses."""
    kept: list[str] = []
    for line in task.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith(("- do not ", "do not ", "- preserve ", "preserve ", "non-goals:", "preserved behavior:")):
            continue
        if "must remain unchanged" in lowered or "must keep working" in lowered:
            continue
        kept.append(stripped)
    return " ".join(kept) or task


def _has_diagnostic_bootstrap_intent(task: str, task_tokens: frozenset[str]) -> bool:
    """Return whether task text targets bounded diagnostic/bootstrap work."""
    lowered = task.lower()
    if task_tokens & DIAGNOSTIC_BOOTSTRAP_TERMS:
        return True
    if "smoke real provider" in lowered:
        return True
    if "set_config" not in lowered:
        return False
    return not any(phrase in lowered for phrase in NEGATIVE_PROVIDER_CONFIG_BOUNDARY_PHRASES)


def _has_localization_completion_intent(task: str) -> bool:
    """Return whether task text targets implementation of localized UI completion."""
    lowered = task.lower()
    raw_tokens = _raw_tokens(task)
    return bool(raw_tokens & LOCALIZATION_STRONG_TASK_TERMS) or any(phrase in lowered for phrase in LOCALIZATION_LANGUAGE_PAIR_PHRASES)


def _raw_tokens(text: str) -> frozenset[str]:
    """Return deterministic lowercase task tokens without alias expansion."""
    return frozenset(token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(token) > 1)


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


def _scope_boundaries(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return scope boundaries for the selected task mode."""
    skills = _selected_prompt_skills(task, task_mode)
    if task_mode == "ui_runtime_bug" and not _skill_selected(skills, "ui_runtime_bug"):
        boundaries = [
            "Inspect only enough code to locate the faulty condition, then fix surgically.",
            "Scoped edits are allowed when needed to fix the requested UI/runtime issue.",
        ]
        boundaries.extend(rule for skill in skills if skill.name != "base_prompt_quality" for rule in skill.scope_rules[:1])
        return tuple(_unique_preserve_order(boundaries))
    return SCOPE_BOUNDARY_BY_MODE[task_mode]


def _mode_requirements(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return extra task-mode-specific requirements."""
    skills = tuple(skill for skill in _selected_prompt_skills(task, task_mode) if skill.name != "base_prompt_quality")
    if skills:
        requirements = tuple(rule for skill in skills for rule in (*skill.scope_rules, *skill.restriction_rules, *skill.validation_expectations))
        return tuple(_unique_preserve_order(list(requirements)))
    return MODE_REQUIREMENTS_BY_MODE.get(task_mode, ())


def _output_requirements(task: str, task_mode: TaskMode) -> tuple[str, ...]:
    """Return minimal output requirements for generated Codex prompts."""
    requirements = ["Summary", "Validation", "PASS/FAIL"]
    if _explicitly_requests_legacy_output_sections(task):
        requirements = ["Files read", "Files changed", *requirements]
    if task_mode == "planning_only":
        return ("Plan", "Risks", "Validation", "PASS/FAIL")
    if task_mode == "review_only":
        return ("Findings", "Risks", "Validation", "PASS/FAIL")
    if task_mode == "diagnostic_bootstrap":
        return ("Diagnostics run", "Evidence", "Validation", "PASS/FAIL")
    return tuple(requirements)


def _explicitly_requests_legacy_output_sections(task: str) -> bool:
    """Return whether task text explicitly asks for legacy output sections."""
    lowered = task.lower()
    return "files read" in lowered or "files changed" in lowered


def _needs_english_summary(task: str) -> bool:
    """Return whether task text needs deterministic English summary rendering."""
    return bool(_tokens(task) & SPANISH_SUMMARY_TERMS) or re.search(r"[áéíóúñÁÉÍÓÚÑ]", task) is not None


def _english_task_summary(task: str, task_mode: TaskMode) -> str:
    """Return a deterministic English objective summary for non-English task text."""
    tokens = _tokens(task)
    if task_mode == "review_only":
        target = _english_target_from_tokens(tokens)
        references = _reference_suffix(task)
        return f"Review {target} without implementing changes{references}."

    action = _english_action_from_tokens(tokens)
    target = _english_target_from_tokens(tokens)
    references = _reference_suffix(task)
    return f"{action} {target}{references}."


def _english_task_details(task: str, task_mode: TaskMode) -> str:
    """Return English task details while preserving literal task evidence separately."""
    summary = _english_task_summary(task, task_mode)
    references = _literal_references(task)
    if references:
        return f"Implementation summary: {summary}\nPreserve exact referenced text and identifiers: {', '.join(references)}."
    return f"Implementation summary: {summary}"


def _english_action_from_tokens(tokens: frozenset[str]) -> str:
    """Return an English action phrase from task tokens."""
    if tokens & {"eliminar", "eliminemos", "quitar", "quitemos", "remove"}:
        return "Remove"
    if tokens & {"ocultar", "hide"}:
        return "Hide"
    if tokens & {"reemplazar", "replace"}:
        return "Replace"
    if tokens & {"cambiar", "modificar", "ajustar", "change", "modify"}:
        return "Update"
    if tokens & IMPLEMENTATION_INTENT_TERMS:
        return "Fix"
    return "Handle"


def _english_target_from_tokens(tokens: frozenset[str]) -> str:
    """Return an English target phrase from task tokens."""
    if "cuenta" in tokens or "cuentas" in tokens:
        if tokens & {"añadir", "mas", "más"}:
            return "the option to add more than one account"
        return "the account UI behavior"
    if "contraseña" in tokens and "login" in tokens:
        return "the login/password UI behavior"
    if tokens & UI_RUNTIME_BUG_TERMS:
        return "the requested UI behavior"
    return "the requested behavior"


def _reference_suffix(task: str) -> str:
    """Return an English suffix listing exact literals to preserve."""
    references = _literal_references(task)
    if not references:
        return ""
    return f" while preserving exact references: {', '.join(references)}"


def _literal_references(task: str) -> tuple[str, ...]:
    """Return task identifiers and literal UI text that must stay verbatim."""
    candidates: list[tuple[int, str]] = []
    for reference in SPANISH_LITERAL_REFERENCES:
        index = task.find(reference)
        if index >= 0:
            candidates.append((index, reference))

    candidates.extend((match.start(), match.group(0)) for match in re.finditer(r"\b[A-Za-z]+[A-Z][A-Za-z0-9]*\b", task))

    seen: set[str] = set()
    ordered: list[str] = []
    for _index, reference in sorted(candidates, key=lambda item: item[0]):
        if reference not in seen:
            seen.add(reference)
            ordered.append(reference)
    return tuple(ordered)


def _has_concrete_ui_or_identifier_details(task: str) -> bool:
    """Return whether unstructured task text has UI details worth preserving."""
    tokens = _tokens(task)
    return bool(tokens & UI_RUNTIME_BUG_TERMS) or re.search(r"\b[A-Za-z]+[A-Z][A-Za-z0-9]*\b", task) is not None


def _render_bullets(items: tuple[str, ...]) -> str:
    """Render a bullet list from plain text items."""
    return "\n".join(f"- {item}" for item in items)


def _render_codex_prompt(task: str, selection: _ContextSelection) -> str:
    """Render the compact prompt intended for Codex."""
    selected_paths = tuple(item for item in selection.selected if not item.startswith("repo not provided"))
    objective = _task_objective(task)
    task_mode = _classify_task_mode(task)
    if task_mode == "ui_visual_microfix":
        return _render_visual_microfix_codex_prompt(task, selected_paths)
    task_details_text = _task_details(task, task_mode)
    mode_requirements = _mode_requirements(task, task_mode)
    mode_requirements_text = _render_bullets(mode_requirements) if mode_requirements else "- (none)"
    knowledge_text = ""
    if selection.knowledge:
        knowledge_text = f"""
Repo knowledge:
{_one_line_list(selection.knowledge)}"""
    return f"""Codex Prompt:
Title: {_title_from_task(task)}
Task mode: {task_mode}
Objective: {objective}
Task details:
{task_details_text}
Selected context paths:
{_one_line_list(selected_paths)}
{knowledge_text}
Scope boundaries:
{_render_bullets(_scope_boundaries(task, task_mode))}
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
{_render_bullets(_output_requirements(task, task_mode))}"""


def _render_visual_microfix_codex_prompt(task: str, selected_paths: tuple[str, ...]) -> str:
    """Render the intentionally small prompt for visual-only UI fixes."""
    target = _visual_microfix_target_text(task, selected_paths)
    change = _visual_microfix_change_text(task)
    return f"""Codex Prompt:
Title: {_title_from_task(task)}
Task mode: ui_visual_microfix
Target file/surface:
- {target}
Exact visual change:
- {change}
Preserve:
- Preserve handlers, forms, routes, data binding, authorization, submitted actions, and delete semantics.
- Do not change persistence, providers, runtime behavior, or non-target UI.
Exploration clamp:
- Inspect only the target file/surface and the nearest existing visual pattern if needed.
- Exclude candidate.json, candidate.codex-output.txt, .protector-harness, bin, obj, generated Connected Services, migrations, and Docs from searches.
- Do not search prompt/log artifacts; avoid matching this task text or Codex output.
Validation:
- Review the diff/static markup for the targeted visual element.
- Run a build only if the Razor markup or syntax risk is non-trivial.
Mandatory output:
- Summary
- Validation
- PASS/FAIL"""


def _visual_microfix_target_text(task: str, selected_paths: tuple[str, ...]) -> str:
    """Return the target file or surface for a visual microfix prompt."""
    concrete_paths = tuple(path for path in selected_paths if path != "target file not resolved; use named surface from task")
    if concrete_paths:
        return ", ".join(concrete_paths[:3])
    surfaces = _visual_microfix_surface_candidates(task)
    if surfaces:
        return ", ".join(surfaces[:3])
    return "Named UI surface from the task"


def _visual_microfix_change_text(task: str) -> str:
    """Return a compact visual-only change statement."""
    refinement = refine_operator_task(task)
    if refinement.requested_change:
        return refinement.requested_change
    return _task_objective(task)


def _visual_microfix_task_details(task: str) -> str:
    """Render only target/change details for visual microfix metadata."""
    return f"Target: {_visual_microfix_target_text(task, ())}\nChange: {_visual_microfix_change_text(task)}"


def _contains_section(text: str, section: str) -> bool:
    """Return whether `text` contains a required output section label."""
    wanted = _normalized_output_header(section)
    return any(_normalized_output_header(line) == wanted for line in text.splitlines())


def _first_nonempty_tuple(*items: tuple[str, ...]) -> tuple[str, ...]:
    """Return the first non-empty tuple after removing display placeholders."""
    for item in items:
        meaningful = _meaningful_output_items(item)
        if meaningful:
            return meaningful
    return ()


def _meaningful_output_items(items: tuple[str, ...]) -> tuple[str, ...]:
    """Return section items that represent real output, not placeholders."""
    return tuple(item for item in items if item.strip().lower() not in {"(none)", "none"})


def _pollution_filtered_codex_output(text: str) -> str:
    """Remove known self-matched or generated search-output lines from a transcript."""
    return "\n".join(line for line in text.splitlines() if not _is_polluted_transcript_line(line))


def _is_polluted_transcript_line(line: str) -> bool:
    """Return whether one transcript line is from ignored self-match/generated output."""
    normalized = _normalize_outcome_path(line)
    if _is_ignored_outcome_path(normalized):
        return True
    return len(line) > OUTCOME_LARGE_GENERATED_LINE_LENGTH and any(
        term in normalized for term in OUTCOME_LARGE_GENERATED_LINE_MARKERS
    )


def _normalize_outcome_path(text: str) -> str:
    """Normalize path-like text for outcome filtering."""
    return text.replace("\\", "/").lower()


def _is_ignored_outcome_path(path: str) -> bool:
    """Return whether an outcome path should be ignored for implementation evidence."""
    normalized = f"/{_normalize_outcome_path(path).lstrip('./')}"
    if any(marker in normalized for marker in OUTCOME_IGNORED_PATH_MARKERS):
        return True
    return normalized.endswith(OUTCOME_IGNORED_SOURCE_MAP_SUFFIXES)


def _changed_files_from_unified_diff(text: str) -> tuple[str, ...]:
    """Extract changed file paths from embedded unified diff headers."""
    files: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line.strip())
        if match is not None:
            _append_outcome_file(files, match.group(2))
            continue
        match = re.match(r"^\+\+\+ b/(.+)$", line.strip())
        if match is not None:
            _append_outcome_file(files, match.group(1))
    return tuple(files)


def _changed_file_section_items(text: str) -> tuple[str, ...]:
    """Extract only path-like entries from a `Files changed` section."""
    files: list[str] = []
    for item in _output_section_items(text, "Files changed"):
        _append_outcome_file(files, item.strip("`"))
    return tuple(files)


def _append_outcome_file(files: list[str], path: str) -> None:
    """Append one meaningful outcome file path, preserving order."""
    cleaned = path.strip().strip('"').replace("\\", "/")
    if not cleaned or cleaned == "/dev/null" or _is_ignored_outcome_path(cleaned):
        return
    if " " in cleaned:
        return
    if "/" not in cleaned and "." not in Path(cleaned).name:
        return
    if cleaned not in files:
        files.append(cleaned)


def _repo_diff_changed_files(repo: Path | None) -> tuple[str, ...]:
    """Return current repo diff files when outcome transcript parsing is ambiguous."""
    if repo is None:
        return ()
    try:
        completed = subprocess.run(  # noqa: S603  # fixed git command with repo passed as one argv value
            ("git", "-C", str(repo), "diff", "--name-only", "--"),
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    files: list[str] = []
    for line in completed.stdout.splitlines():
        _append_outcome_file(files, line)
    return tuple(files)


def _validation_items_from_narrative(text: str) -> tuple[str, ...]:
    """Recover validation evidence from malformed final assistant sections."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        if not stripped or _is_known_output_section_header(stripped) or _is_polluted_transcript_line(stripped):
            continue
        lowered = stripped.lower()
        if any(term in lowered for term in VALIDATION_EVIDENCE_TERMS):
            items.append(stripped)
    return tuple(_unique_preserve_order(items[-5:]))


def _has_execution_activity(text: str) -> bool:
    """Return whether the transcript shows actual assistant or local command activity."""
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "\nassistant\n",
            "\ncodex\n",
            "\nexec\n",
            "pwsh.exe ",
            "powershell.exe ",
            " succeeded in ",
            "diff --git ",
        )
    )


def _normalized_output_header(line: str) -> str:
    """Normalize markdown/plain Codex section headings."""
    normalized = line.strip()
    normalized = re.sub(r"^\s*[-*]\s+", "", normalized)
    normalized = normalized.strip("#").strip()
    normalized = normalized.strip("*`_ ").strip()
    normalized = normalized.rstrip(":").strip()
    return normalized.lower()


def _output_section_items(text: str, section: str) -> tuple[str, ...]:
    """Extract simple bullet/plain lines from one Codex output section."""
    lines = text.splitlines()
    start = _section_start_index(lines, section)
    if start is None:
        return ()
    items: list[str] = []
    for raw in lines[start + 1 :]:
        stripped = raw.strip()
        if _is_known_output_section_header(stripped):
            break
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        items.append(stripped)
    return _meaningful_output_items(tuple(items))


def _candidate_executor_failed(output: str, *, repo_changed_files: tuple[str, ...] = ()) -> bool:
    """Return whether candidate execution failed before implementation review."""
    review_text = _pollution_filtered_codex_output(_codex_output_after_prompt_contract(output))
    if (
        _completed_codex_result_present(review_text)
        or _changed_files_from_unified_diff(review_text)
        or (repo_changed_files and _has_execution_activity(review_text))
    ):
        return False
    lowered = (review_text if "Codex Prompt:" in output else output).lower()
    return (
        _codex_transcript_contains_only_user_prompt(output)
        or "local_executor_unavailable" in lowered
        or "failure: executor failure" in lowered
        or ("candidate execution status: failure" in lowered and "process tree status: process_tree_terminated" in lowered)
    )


def _completed_codex_result_present(output: str) -> bool:
    """Return whether the transcript includes a real assistant implementation result."""
    if not output.strip():
        return False
    if _output_section_items(output, "Files changed"):
        return True
    if _output_section_items(output, "Validation"):
        return True
    if _changed_files_from_unified_diff(output):
        return True
    return _status_token(output) is not None


def _codex_transcript_contains_only_user_prompt(output: str) -> bool:
    """Return whether raw Codex output captured only the submitted prompt."""
    lines = [line.strip() for line in output.splitlines()]
    return "OpenAI Codex" in output and "user" in lines and "assistant" not in lines and "Codex Prompt:" in output


def _codex_output_after_prompt_contract(output: str) -> str:
    """Return only the execution result area, excluding echoed prompt contract text."""
    lines = output.splitlines()
    prompt_index = _last_line_index(lines, "Codex Prompt:")
    if prompt_index is None:
        return output
    turn_index = _first_codex_result_turn_index(lines, start=prompt_index + 1)
    if turn_index is not None:
        return "\n".join(lines[turn_index:])
    policy_index = _last_line_index(lines, "Local executor policy:", start=prompt_index)
    if policy_index is not None:
        result_index = _first_result_section_after_policy(lines, policy_index)
        if result_index is not None:
            return "\n".join(lines[result_index:])
        first_blank = _first_blank_line_after(lines, policy_index)
        if first_blank is not None:
            return "\n".join(lines[first_blank + 1 :])
    result_index = _first_actual_output_section_index(lines, start=prompt_index + 1)
    if result_index is None:
        return ""
    return "\n".join(lines[result_index:])


def _last_line_index(lines: list[str], prefix: str, *, start: int = 0) -> int | None:
    """Return the last line index whose stripped text starts with `prefix`."""
    index: int | None = None
    for current in range(start, len(lines)):
        if lines[current].strip().startswith(prefix):
            index = current
    return index


def _first_codex_result_turn_index(lines: list[str], *, start: int) -> int | None:
    """Return the first Codex assistant/local-command turn marker after a prompt."""
    for index in range(start, len(lines)):
        if lines[index].strip().lower() in {"assistant", "codex", "exec"}:
            return index
    return None


def _first_result_section_after_policy(lines: list[str], policy_index: int) -> int | None:
    """Return the first actual result section after the local executor policy block."""
    first_blank = _first_blank_line_after(lines, policy_index)
    if first_blank is None:
        return None
    return _first_actual_output_section_index(lines, start=first_blank + 1)


def _first_blank_line_after(lines: list[str], start: int) -> int | None:
    """Return the first blank line index after `start`."""
    for index in range(start + 1, len(lines)):
        if not lines[index].strip():
            return index
    return None


def _first_actual_output_section_index(lines: list[str], *, start: int) -> int | None:
    """Return the first non-bulleted output section header after `start`."""
    for index in range(start, len(lines)):
        if _is_actual_output_section_header(lines[index].strip()):
            return index
    return None


def _is_actual_output_section_header(line: str) -> bool:
    """Return whether `line` is an implementation output section header."""
    return _normalized_output_header(line) in _KNOWN_OUTPUT_HEADERS


def _section_start_index(lines: list[str], section: str) -> int | None:
    """Return the last index of a section header line."""
    wanted = _normalized_output_header(section)
    found: int | None = None
    for index, line in enumerate(lines):
        if _normalized_output_header(line) == wanted:
            found = index
    return found


def _is_known_output_section_header(line: str) -> bool:
    """Return whether a line starts a known Codex output section."""
    return _normalized_output_header(line) in _KNOWN_OUTPUT_HEADERS


def _has_pass_or_fail(text: str) -> bool:
    """Return whether `text` contains an explicit PASS or FAIL token."""
    return _status_token(text) is not None


def _status_token(text: str) -> str | None:
    """Return the last standalone PASS/FAIL verdict line found."""
    verdict: str | None = None
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(?:[-*]\s*)?(PASS|FAIL)\s*:?\s*", line, flags=re.IGNORECASE)
        if match is not None:
            verdict = match.group(1).upper()
    return verdict


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


def _changed_file_deviations(expected_files: tuple[str, ...], actual_files: tuple[str, ...]) -> tuple[str, ...]:
    """Return file-change deviations from the supervised plan."""
    if not actual_files:
        return ("Codex output did not report changed files.",)
    if expected_files and expected_files[0].startswith("Unknown until"):
        return ()
    expected_tokens = _tokens(" ".join(expected_files))
    deviations: list[str] = []
    for file in actual_files:
        if file.lower() in {"(none)", "none"}:
            continue
        if not (_tokens(file) & expected_tokens):
            deviations.append(f"Changed file outside expected plan context: {file}")
    return tuple(deviations)


def _outcome_validation_gaps(
    expected_validation: tuple[str, ...],
    actual_validation: tuple[str, ...],
    deterministic_warnings: tuple[str, ...],
) -> tuple[str, ...]:
    """Return validation gaps compared with the review contract."""
    gaps = list(deterministic_warnings)
    if not actual_validation:
        gaps.append("Codex output did not report executed validation.")
    elif not _validation_evidence_present(" ".join(actual_validation)):
        gaps.append("Reported validation lacks build/test/smoke/check evidence.")
    expected_tokens = _tokens(" ".join(expected_validation))
    actual_tokens = _tokens(" ".join(actual_validation))
    missing_terms = tuple(sorted((expected_tokens & {"build", "check", "smoke", "test", "verified", "workflow", "ui"}) - actual_tokens))
    if missing_terms:
        gaps.append(f"Validation may not cover expected scope terms: {', '.join(missing_terms)}.")
    gaps.extend(_validation_decision_record_gaps(actual_validation))
    return tuple(_unique_preserve_order(gaps))


def _validation_decision_record_gaps(actual_validation: tuple[str, ...]) -> tuple[str, ...]:
    """Return VDR gaps when validation escalates without recorded decision evidence."""
    text = " ".join(actual_validation).lower()
    if not text or not any(term in text for term in VALIDATION_ESCALATION_TERMS):
        return ()
    missing = tuple(field for field, evidence_terms in VDR_FIELD_EVIDENCE_TERMS.items() if not any(term in text for term in evidence_terms))
    if not missing:
        return ()
    return (f"Validation escalated without VDR evidence: missing {', '.join(missing)}.",)


def _selected_skill_behavior_deviations(
    output: str,
    selected_pack_skills: tuple[PromptSkill, ...],
    selected_runtime_skills: tuple[PromptSkill, ...],
) -> tuple[str, ...]:
    """Return deviations between selected skills and described Codex behavior."""
    output_tokens = _tokens(output)
    deviations: list[str] = []
    for skill in selected_pack_skills:
        skill_terms = _tokens(" ".join((skill.name, skill.expected_behavior or "", *skill.scope_rules, *skill.restriction_rules)))
        if skill_terms and not (output_tokens & skill_terms):
            deviations.append(f"Codex output does not mention behavior tied to selected pack skill: {skill.name}")
    if selected_runtime_skills and not any(skill.name == "base_prompt_quality" for skill in selected_runtime_skills):
        deviations.append("Runtime fallback was selected; verify generic behavior did not replace pack-specific constraints.")
    return tuple(deviations)


def _outcome_blocker_deviations(
    pack: object,
    selected_pack_skills: tuple[PromptSkill, ...],
    coverage: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return pack or benchmark blockers visible at outcome time."""
    missing_coverage = tuple(skill.name for skill in selected_pack_skills if not coverage.get(skill.name))
    blockers = _ecc_supervised_blockers(pack, missing_coverage)
    return tuple(item for item in blockers if item != "None for read-only supervised planning.")


def _pass_fail_consistency_gaps(
    output: str,
    review: _ReviewResult,
    *,
    deviations: tuple[str, ...],
    validation_gaps: tuple[str, ...],
) -> tuple[str, ...]:
    """Return PASS/FAIL consistency findings."""
    verdict = _status_token(output)
    if verdict is None:
        return ("Codex output did not include PASS or FAIL.",)
    if verdict == "PASS" and (review.status != "PASS" or deviations or validation_gaps):
        return ("Codex claimed PASS but deterministic review found unresolved gaps.",)
    if verdict == "FAIL" and review.status == "FAIL":
        return ("Codex reported FAIL; outcome cannot be accepted.",)
    return ()


def _outcome_status(
    review: _ReviewResult,
    *,
    deviations: tuple[str, ...],
    validation_gaps: tuple[str, ...],
) -> str:
    """Return accepted / needs review / failed for an outcome report."""
    if review.status == "FAIL" or any("reported FAIL" in item for item in deviations):
        return "failed"
    meaningful_deviations = tuple(item for item in deviations if item != "None for read-only supervised planning.")
    if review.status != "PASS" or meaningful_deviations or validation_gaps:
        return "needs review"
    return "accepted"


def _outcome_follow_up_prompt(
    task: str,
    *,
    deviations: tuple[str, ...],
    validation_gaps: tuple[str, ...],
    status: str,
) -> str | None:
    """Return a compact follow-up prompt when the outcome is not accepted."""
    if status == "accepted":
        return None
    issues = tuple(item for item in (*deviations, *validation_gaps) if item != "None for read-only supervised planning.")
    issue_text = "; ".join(issues[:4]) or "Outcome requires supervised review."
    return f"Revise or justify the Codex result for: {task}. Address these outcome gaps: {issue_text}."


def _suggested_benchmark_additions(
    selected_pack_skills: tuple[PromptSkill, ...],
    coverage: dict[str, tuple[str, ...]],
    output: str,
    deviations: tuple[str, ...],
) -> tuple[str, ...]:
    """Suggest benchmark additions only for real uncovered behavior described by the output."""
    suggestions: list[str] = []
    output_tokens = _tokens(output)
    for skill in selected_pack_skills:
        if coverage.get(skill.name):
            continue
        skill_terms = _tokens(" ".join((skill.name, skill.expected_behavior or "")))
        if output_tokens & skill_terms:
            suggestions.append(f"Add prompt benchmark coverage for uncovered pack skill: {skill.name}.")
    if any("does not mention behavior tied to selected pack skill" in item for item in deviations):
        suggestions.append("No benchmark addition suggested until the real uncovered behavior is clarified.")
    return tuple(suggestions) or ("(none)",)


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp for local operational history."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _history_status(status: str) -> str:
    """Normalize rendered outcome status for structured history."""
    return status.replace(" ", "_")


def _meaningful_benchmark_additions(items: tuple[str, ...]) -> tuple[str, ...]:
    """Return real benchmark suggestions, excluding display placeholders."""
    return tuple(item for item in items if item != "(none)")


def _outcome_summary_payload(summary: OutcomeSummary) -> dict[str, object]:
    """Return a JSON-safe outcome summary payload."""
    return {
        "timestamp": summary.timestamp,
        "task": summary.task,
        "selected_pack": summary.selected_pack,
        "selected_skills": tuple({"name": skill.name, "source": skill.source} for skill in summary.selected_skills),
        "status": summary.status,
        "changed_files": summary.changed_files,
        "executed_validations": summary.executed_validations,
        "deviations": summary.deviations,
        "follow_up_prompt": summary.follow_up_prompt,
        "suggested_benchmark_additions": summary.suggested_benchmark_additions,
    }


def _outcome_summary_from_payload(payload: object) -> OutcomeSummary | None:
    """Parse one outcome history JSON object."""
    if not isinstance(payload, dict):
        return None
    selected_skills = _outcome_skill_summaries(payload.get("selected_skills"))
    return OutcomeSummary(
        timestamp=_string_field(payload, "timestamp"),
        task=_string_field(payload, "task"),
        selected_pack=_string_field(payload, "selected_pack"),
        selected_skills=selected_skills,
        status=_string_field(payload, "status"),
        changed_files=_string_tuple_field(payload, "changed_files"),
        executed_validations=_string_tuple_field(payload, "executed_validations"),
        deviations=_string_tuple_field(payload, "deviations"),
        follow_up_prompt=_optional_string_field(payload, "follow_up_prompt"),
        suggested_benchmark_additions=_string_tuple_field(payload, "suggested_benchmark_additions"),
    )


def _outcome_skill_summaries(value: object) -> tuple[OutcomeSkillSummary, ...]:
    """Parse selected skill history rows."""
    if not isinstance(value, list):
        return ()
    rows: list[OutcomeSkillSummary] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        source = item.get("source")
        if isinstance(name, str) and isinstance(source, str):
            rows.append(OutcomeSkillSummary(name=name, source=source))
    return tuple(rows)


def _string_field(payload: dict[object, object], key: str) -> str:
    """Read a string field from a history payload."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _optional_string_field(payload: dict[object, object], key: str) -> str | None:
    """Read an optional string field from a history payload."""
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _string_tuple_field(payload: dict[object, object], key: str) -> tuple[str, ...]:
    """Read a string tuple field from a history payload."""
    value = payload.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _recent_outcome_signals(repo: Path | None, prompt_skills: tuple[PromptSkill, ...]) -> tuple[str, ...]:
    """Return recent relevant outcome history signals for the selected pack/skills."""
    if repo is None:
        return ("History unavailable: repo path was not provided.",)
    pack = discover_protector_pack(include_benchmarks=False)
    selected_names = {skill.name for skill in prompt_skills}
    signals: list[str] = []
    for entry in load_outcome_history(repo, limit=20):
        if entry.selected_pack != pack.name:
            continue
        entry_skill_names = {skill.name for skill in entry.selected_skills}
        overlap = tuple(sorted(selected_names & entry_skill_names))
        if not overlap:
            continue
        signals.append(_outcome_signal_row(entry, overlap))
        if len(signals) >= OUTCOME_HISTORY_SIGNAL_LIMIT:
            break
    return tuple(signals) or ("No relevant repo-scoped supervised outcome history for selected pack/skills.",)


def _outcome_signal_row(entry: OutcomeSummary, overlap: tuple[str, ...]) -> str:
    """Render one compact outcome signal for a future plan."""
    parts = [
        f"{entry.timestamp} {entry.status}",
        f"skills={', '.join(overlap)}",
        f"files={_inline_or_none(entry.changed_files)}",
    ]
    if entry.deviations:
        parts.append(f"deviations={_inline_or_none(entry.deviations[:2])}")
    if entry.follow_up_prompt:
        parts.append("follow-up=yes")
    return " | ".join(parts)


def _outcome_learning_signals(repo: Path | None, prompt_skills: tuple[PromptSkill, ...], *, limit: int = 50) -> tuple[str, ...]:
    """Derive deterministic planning warnings from repeated supervised outcomes."""
    if repo is None:
        return ("History unavailable: repo path was not provided.",)
    entries = _matching_outcome_history(repo, prompt_skills, limit=limit)
    if not entries:
        return ("No outcome history entries match the selected pack/skills.",)

    signals: list[str] = []
    signals.extend(_recurring_validation_signals(entries))
    signals.extend(_recurring_changed_file_mismatch_signals(entries))
    signals.extend(_recurring_benchmark_coverage_signals(entries))
    signals.extend(_recurring_follow_up_signals(entries))
    signals.extend(_recurring_drift_deviation_signals(entries))
    return tuple(signals) or ("No recurring outcome learning signals for selected pack/skills.",)


def _adaptive_planning_adjustments(learning_signals: tuple[str, ...]) -> tuple[str, ...]:
    """Map deterministic learning signals to explainable planning adaptations."""
    adjustments: list[str] = []
    for signal in learning_signals:
        adjustment = _adaptive_planning_adjustment(signal)
        if adjustment is not None:
            adjustments.append(adjustment)
    return tuple(adjustments) or ("No adaptive planning adjustments applied.",)


def _adaptive_planning_adjustment(signal: str) -> str | None:
    """Return one planning adaptation triggered by one learning signal."""
    if signal.startswith("Recurring failed validation:"):
        return (
            "Strengthen validation expectations: require explicit build/test/smoke evidence in the Codex handoff and review contract. "
            f"Triggered by learning signal: {signal}"
        )
    if signal.startswith("Skill repeatedly lacks benchmark coverage:"):
        return (
            "Increase benchmark emphasis: call out benchmark coverage risk before relying on the affected pack skill. "
            f"Triggered by learning signal: {signal}"
        )
    if signal.startswith("Repeated drift/deviation pattern:"):
        return (
            "Increase anti-drift guidance: restate the exact task boundary and reject unrelated platform/dashboard/session/workflow expansion. "
            f"Triggered by learning signal: {signal}"
        )
    if signal.startswith("Recurring changed-file mismatch:"):
        file_hint = _changed_file_hint_from_signal(signal)
        return (
            f"Surface likely changed-file area: {file_hint}. Require Codex to either inspect this area or justify why it is out of scope. "
            f"Triggered by learning signal: {signal}"
        )
    if signal.startswith("Repeated follow-up prompt:"):
        return (
            "Pre-apply repeated follow-up: include the recurring correction in the initial supervised handoff. "
            f"Triggered by learning signal: {signal}"
        )
    return None


def _changed_file_hint_from_signal(signal: str) -> str:
    """Extract a file hint from a changed-file mismatch signal."""
    match = re.search(r"context:\s*(.+?)\s*\(\d+\s+outcomes\)", signal)
    if match is not None:
        return match.group(1).strip()
    if "did not report changed files" in signal:
        return "changed files were omitted from prior Codex summaries"
    return "the recurring mismatch path from outcome history"


def _matching_outcome_history(repo: Path, prompt_skills: tuple[PromptSkill, ...], *, limit: int) -> tuple[OutcomeSummary, ...]:
    """Return history entries matching the current pack and selected skills."""
    pack = discover_protector_pack(include_benchmarks=False)
    selected_names = {skill.name for skill in prompt_skills}
    entries: list[OutcomeSummary] = []
    for entry in load_outcome_history(repo, limit=limit):
        if entry.selected_pack != pack.name:
            continue
        if selected_names:
            entry_skill_names = {skill.name for skill in entry.selected_skills}
            if not (selected_names & entry_skill_names):
                continue
        entries.append(entry)
    return tuple(entries)


def _recurring_validation_signals(entries: tuple[OutcomeSummary, ...]) -> tuple[str, ...]:
    """Return recurring failed-validation warnings."""
    counter: dict[str, int] = {}
    for entry in entries:
        for validation in entry.executed_validations:
            if not _validation_evidence_present(validation):
                _increment(counter, _learning_pattern(validation))
        for deviation in entry.deviations:
            lowered = deviation.lower()
            if "validation" in lowered or "pass claimed without validation evidence" in lowered:
                _increment(counter, _learning_pattern(deviation))
    return tuple(
        f"Recurring failed validation: {pattern} ({count} outcomes). Recommendation: require concrete build/test/smoke evidence before PASS."
        for pattern, count in _recurring_items(counter)
    )


def _recurring_changed_file_mismatch_signals(entries: tuple[OutcomeSummary, ...]) -> tuple[str, ...]:
    """Return recurring changed-file mismatch warnings."""
    counter: dict[str, int] = {}
    for entry in entries:
        for deviation in entry.deviations:
            if deviation.startswith(("Changed file outside expected plan context:", "Codex output did not report changed files.")):
                _increment(counter, _learning_pattern(deviation))
    return tuple(
        f"Recurring changed-file mismatch: {pattern} ({count} outcomes). Recommendation: make expected files explicit in the next Codex handoff."
        for pattern, count in _recurring_items(counter)
    )


def _recurring_benchmark_coverage_signals(entries: tuple[OutcomeSummary, ...]) -> tuple[str, ...]:
    """Return recurring benchmark-coverage warnings."""
    counter: dict[str, int] = {}
    for entry in entries:
        for suggestion in entry.suggested_benchmark_additions:
            skill = _benchmark_skill_name(suggestion)
            _increment(counter, skill or _learning_pattern(suggestion))
        for deviation in entry.deviations:
            if "missing benchmark coverage" in deviation.lower():
                _increment(counter, _learning_pattern(deviation))
    recommendation = "Recommendation: add pack benchmark coverage before relying on this specialization."
    return tuple(
        f"Skill repeatedly lacks benchmark coverage: {pattern} ({count} outcomes). {recommendation}" for pattern, count in _recurring_items(counter)
    )


def _recurring_follow_up_signals(entries: tuple[OutcomeSummary, ...]) -> tuple[str, ...]:
    """Return recurring follow-up prompt warnings."""
    counter: dict[str, int] = {}
    for entry in entries:
        if entry.follow_up_prompt:
            _increment(counter, _learning_pattern(entry.follow_up_prompt))
    return tuple(
        f"Repeated follow-up prompt: {pattern} ({count} outcomes). Recommendation: include this correction in the initial plan."
        for pattern, count in _recurring_items(counter)
    )


def _recurring_drift_deviation_signals(entries: tuple[OutcomeSummary, ...]) -> tuple[str, ...]:
    """Return recurring drift/deviation warnings."""
    counter: dict[str, int] = {}
    for entry in entries:
        for deviation in entry.deviations:
            lowered = deviation.lower()
            drift_terms = ("mentioned without task scope", "does not mention behavior tied to selected pack skill", "runtime fallback")
            if any(term in lowered for term in drift_terms):
                _increment(counter, _learning_pattern(deviation))
    recommendation = "Recommendation: state the anti-drift constraint explicitly before Codex runs."
    return tuple(f"Repeated drift/deviation pattern: {pattern} ({count} outcomes). {recommendation}" for pattern, count in _recurring_items(counter))


def _increment(counter: dict[str, int], key: str) -> None:
    """Increment a string counter."""
    if not key:
        return
    counter[key] = counter.get(key, 0) + 1


def _recurring_items(counter: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """Return recurring counter items sorted by frequency then name."""
    return tuple(
        sorted(
            ((pattern, count) for pattern, count in counter.items() if count >= OUTCOME_LEARNING_SIGNAL_MIN_COUNT),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _learning_pattern(text: str) -> str:
    """Normalize one history value into a concise repeated pattern."""
    compact = " ".join(text.strip().split())
    if len(compact) > OUTCOME_LEARNING_PATTERN_LIMIT:
        return f"{compact[: OUTCOME_LEARNING_PATTERN_LIMIT - 3]}..."
    return compact


def _benchmark_skill_name(text: str) -> str | None:
    """Extract a skill name from a benchmark suggestion when present."""
    match = re.search(r"uncovered pack skill:\s*([\w-]+)", text, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1)
    return None


def _skill_names(skills: tuple[OutcomeSkillSummary, ...]) -> str:
    """Render skill names with sources."""
    return _inline_or_none(tuple(f"{skill.name} [{skill.source}]" for skill in skills))


def _inline_or_none(items: tuple[str, ...]) -> str:
    """Render a compact comma-separated list."""
    return ", ".join(items) if items else "(none)"
