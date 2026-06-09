"""Compatibility-only controlled planning models pending ECC migration.

Do not expand these into Protector-owned generic registries. New generic agent,
skill, profile, sandbox, review-chain, or loop capabilities belong in ECC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from deepagents.harnesses.protector._prompt_skills import PromptSkill

TaskMode = Literal[
    "implementation_fix",
    "review_only",
    "planning_only",
    "diagnostic_bootstrap",
    "continuation_followup",
    "ui_runtime_bug",
    "provider_api_bug",
]
ExecutionProfileName = Literal["prompt_only", "review_only", "supervised_implementation", "bounded_loop_candidate"]
SandboxAccess = Literal["read_only", "supervised_workspace_write"]
LoopMode = Literal["disabled", "candidate_only"]


@dataclass(frozen=True)
class AgentDefinition:
    """Narrow role that may participate in a controlled Protector run."""

    name: str
    purpose: str
    may_execute_codex: bool
    may_edit_files: bool


@dataclass(frozen=True)
class SkillDefinition:
    """Registered prompt or orchestration capability with clear triggers."""

    name: str
    trigger: str
    role: str


@dataclass(frozen=True)
class SandboxPolicy:
    """Sandbox boundary for a generated execution plan."""

    name: str
    access: SandboxAccess
    codex_execution_enabled: bool
    shell_execution_enabled: bool
    file_write_enabled: bool
    gates: tuple[str, ...]


@dataclass(frozen=True)
class ReviewerChain:
    """Reviewer sequence required before an execution plan can be trusted."""

    name: str
    reviewers: tuple[str, ...]
    gates: tuple[str, ...]


@dataclass(frozen=True)
class LoopPolicy:
    """Bounded-loop policy for future automation."""

    name: str
    mode: LoopMode
    max_iterations: int
    stop_conditions: tuple[str, ...]
    gates: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionProfile:
    """Execution profile that selects agents and safety models."""

    name: ExecutionProfileName
    description: str
    agent_names: tuple[str, ...]
    sandbox_policy: SandboxPolicy
    reviewer_chain: ReviewerChain
    loop_policy: LoopPolicy


@dataclass(frozen=True)
class ExecutionPlan:
    """Selected controlled execution plan for one task."""

    profile: ExecutionProfile
    agents: tuple[AgentDefinition, ...]
    skills: tuple[SkillDefinition, ...]
    safety_gates: tuple[str, ...]


AGENT_REGISTRY: dict[str, AgentDefinition] = {
    "planner": AgentDefinition(
        name="planner",
        purpose="Break the request into a bounded implementation or review plan.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
    "context-loader": AgentDefinition(
        name="context-loader",
        purpose="Select only repo, knowledge, skill, route, and contract context needed by the task.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
    "implementer": AgentDefinition(
        name="implementer",
        purpose="Future executor for scoped code changes after approval and sandbox gates.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        purpose="Review produced work against prompt requirements and protected behavior.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
    "validation-runner": AgentDefinition(
        name="validation-runner",
        purpose="Future executor for approved build, test, and smoke commands.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
    "drift-guard": AgentDefinition(
        name="drift-guard",
        purpose="Detect scope expansion, protected-area touches, and missing validation evidence.",
        may_execute_codex=False,
        may_edit_files=False,
    ),
}

SKILL_REGISTRY: dict[str, SkillDefinition] = {
    "base_prompt_quality": SkillDefinition(
        name="base_prompt_quality",
        trigger="Always active.",
        role="Keep prompts compact, scoped, and validation-oriented.",
    ),
    "implementation_fix": SkillDefinition(
        name="implementation_fix",
        trigger="Implementation terms such as fix, bug, change, remove, or repair.",
        role="Permit scoped implementation planning while preventing audit-only drift.",
    ),
    "regression_fix": SkillDefinition(
        name="regression_fix",
        trigger="Regression, missing behavior, disappeared action, or restore wording.",
        role="Protect previously working behavior while restoring the narrow regression.",
    ),
    "continuation_followup": SkillDefinition(
        name="continuation_followup",
        trigger="Continuation, follow-up, previous fix, or after-fix regression wording.",
        role="Preserve validated previous fixes and continue from the named gap.",
    ),
    "ui_runtime_bug": SkillDefinition(
        name="ui_runtime_bug",
        trigger="Concrete UI, Razor, JS, DOM, modal, view, table, or spinner terms.",
        role="Require proof of the rendered condition before UI changes.",
    ),
    "navigation_surface_convergence": SkillDefinition(
        name="navigation_surface_convergence",
        trigger="Navigation, menu, dashboard, index, view, or legacy surface terms.",
        role="Converge route/menu/view entry points without dashboard redesign.",
    ),
    "form_security_autofill_bug": SkillDefinition(
        name="form_security_autofill_bug",
        trigger="Autofill, autocomplete, login, Email, password, or `contraseña` terms.",
        role="Keep authentication field behavior specific instead of generic UI work.",
    ),
    "provider_api_bug": SkillDefinition(
        name="provider_api_bug",
        trigger="Provider, API, dispatch, payload, signature, or Lleida state terms.",
        role="Protect provider/config/payload boundaries while fixing workflow state mapping.",
    ),
    "provider_bootstrap_diagnostic": SkillDefinition(
        name="provider_bootstrap_diagnostic",
        trigger="Bootstrap, diagnostic, probe, config, SET_CONFIG, or smoke provider terms.",
        role="Keep provider diagnostics bounded and evidence-driven.",
    ),
    "operational_workflow_convergence": SkillDefinition(
        name="operational_workflow_convergence",
        trigger="Workflow action, activation, pending, signature, account, or entity terms.",
        role="Trace workflow state and operator action eligibility before changing behavior.",
    ),
    "mvp_surface_completion": SkillDefinition(
        name="mvp_surface_completion",
        trigger="MVP, useful, operational, operator, surface, or complete terms.",
        role="Complete only the requested daily operator surface.",
    ),
    "global_pattern_change": SkillDefinition(
        name="global_pattern_change",
        trigger="Global, everywhere, all usages, entities, or repeated pattern terms.",
        role="Change targeted repeated usages consistently without broadening scope.",
    ),
    "spanish_implementation_task_preservation": SkillDefinition(
        name="spanish_implementation_task_preservation",
        trigger="Spanish UI labels, identifiers, accents, or task wording.",
        role="Preserve literal UI labels and identifiers while rendering English prompt structure.",
    ),
}

_NO_CODEX_SANDBOX = SandboxPolicy(
    name="no_codex_execution",
    access="read_only",
    codex_execution_enabled=False,
    shell_execution_enabled=False,
    file_write_enabled=False,
    gates=(
        "Codex execution disabled in v1.",
        "No file writes by planned agents.",
        "No shell commands by planned agents.",
    ),
)
_SUPERVISED_WRITE_SANDBOX = SandboxPolicy(
    name="future_supervised_workspace_write",
    access="supervised_workspace_write",
    codex_execution_enabled=False,
    shell_execution_enabled=False,
    file_write_enabled=False,
    gates=(
        "Codex execution disabled until a later approved slice.",
        "Future writes require explicit operator approval.",
        "Future shell commands require sandbox policy approval.",
    ),
)
_PROMPT_REVIEWERS = ReviewerChain(
    name="prompt_safety_review",
    reviewers=("drift-guard",),
    gates=("Prompt must keep task-specific request dominant.", "No autonomous execution may be implied."),
)
_IMPLEMENTATION_REVIEWERS = ReviewerChain(
    name="implementation_safety_review",
    reviewers=("reviewer", "validation-runner", "drift-guard"),
    gates=(
        "Reviewer must check requested behavior and protected decisions.",
        "Validation-runner must identify focused checks before PASS.",
        "Drift-guard must block broad audits and unrelated architecture.",
    ),
)
_NO_LOOP = LoopPolicy(
    name="no_loop",
    mode="disabled",
    max_iterations=0,
    stop_conditions=("Stop after prompt or plan generation.",),
    gates=("No retry loop may run in v1.",),
)
_BOUNDED_LOOP_CANDIDATE = LoopPolicy(
    name="future_bounded_loop_candidate",
    mode="candidate_only",
    max_iterations=2,
    stop_conditions=(
        "Stop on protected-area drift.",
        "Stop on missing validation evidence.",
        "Stop after reviewer requests human decision.",
    ),
    gates=("Loop candidate is described only; it does not execute in v1.",),
)

EXECUTION_PROFILE_REGISTRY: dict[str, ExecutionProfile] = {
    "prompt_only": ExecutionProfile(
        name="prompt_only",
        description="Generate a Codex prompt or plan only.",
        agent_names=("planner", "context-loader", "drift-guard"),
        sandbox_policy=_NO_CODEX_SANDBOX,
        reviewer_chain=_PROMPT_REVIEWERS,
        loop_policy=_NO_LOOP,
    ),
    "review_only": ExecutionProfile(
        name="review_only",
        description="Review provided output without implementation.",
        agent_names=("context-loader", "reviewer", "drift-guard"),
        sandbox_policy=_NO_CODEX_SANDBOX,
        reviewer_chain=_PROMPT_REVIEWERS,
        loop_policy=_NO_LOOP,
    ),
    "supervised_implementation": ExecutionProfile(
        name="supervised_implementation",
        description="Prepare for a future supervised implementation run.",
        agent_names=("planner", "context-loader", "implementer", "reviewer", "validation-runner", "drift-guard"),
        sandbox_policy=_SUPERVISED_WRITE_SANDBOX,
        reviewer_chain=_IMPLEMENTATION_REVIEWERS,
        loop_policy=_NO_LOOP,
    ),
    "bounded_loop_candidate": ExecutionProfile(
        name="bounded_loop_candidate",
        description="Mark a task as a future bounded-loop candidate without running it.",
        agent_names=("planner", "context-loader", "implementer", "validation-runner", "reviewer", "drift-guard"),
        sandbox_policy=_SUPERVISED_WRITE_SANDBOX,
        reviewer_chain=_IMPLEMENTATION_REVIEWERS,
        loop_policy=_BOUNDED_LOOP_CANDIDATE,
    ),
}


def build_execution_plan(*, task_mode: TaskMode, prompt_skills: tuple[PromptSkill, ...]) -> ExecutionPlan:
    """Build a controlled execution plan without invoking Codex or shell tools."""
    profile = _select_execution_profile(task_mode)
    skills = tuple(SKILL_REGISTRY[skill.name] for skill in prompt_skills if skill.name in SKILL_REGISTRY)
    agents = tuple(AGENT_REGISTRY[name] for name in profile.agent_names)
    safety_gates = _execution_safety_gates(profile)
    return ExecutionPlan(profile=profile, agents=agents, skills=skills, safety_gates=safety_gates)


def render_execution_plan(plan: ExecutionPlan) -> str:
    """Render a human-readable execution plan for CLI and Operator UI."""
    return f"""Execution Plan:
Profile: {plan.profile.name}
Profile description: {plan.profile.description}

Agents:
{_render_agents(plan.agents)}

Skills:
{_render_skills(plan.skills)}

Sandbox policy:
- Name: {plan.profile.sandbox_policy.name}
- Access: {plan.profile.sandbox_policy.access}
- Codex execution enabled: {_yes_no(value=plan.profile.sandbox_policy.codex_execution_enabled)}
- Shell execution enabled: {_yes_no(value=plan.profile.sandbox_policy.shell_execution_enabled)}
- File write enabled: {_yes_no(value=plan.profile.sandbox_policy.file_write_enabled)}

Reviewer chain:
- Name: {plan.profile.reviewer_chain.name}
- Reviewers: {', '.join(plan.profile.reviewer_chain.reviewers)}

Loop policy:
- Name: {plan.profile.loop_policy.name}
- Mode: {plan.profile.loop_policy.mode}
- Max iterations: {plan.profile.loop_policy.max_iterations}
- Stop conditions: {'; '.join(plan.profile.loop_policy.stop_conditions)}

Safety gates:
{_render_bullets(plan.safety_gates)}

Execution:
- Codex execution: disabled
- Autonomous code execution: disabled"""


def _select_execution_profile(task_mode: TaskMode) -> ExecutionProfile:
    """Select the first v1 execution profile for a task mode."""
    if task_mode == "review_only":
        return EXECUTION_PROFILE_REGISTRY["review_only"]
    if task_mode == "diagnostic_bootstrap":
        return EXECUTION_PROFILE_REGISTRY["bounded_loop_candidate"]
    if task_mode == "planning_only":
        return EXECUTION_PROFILE_REGISTRY["prompt_only"]
    return EXECUTION_PROFILE_REGISTRY["supervised_implementation"]


def _execution_safety_gates(profile: ExecutionProfile) -> tuple[str, ...]:
    """Combine profile safety gates in deterministic order."""
    gates = [
        *profile.sandbox_policy.gates,
        *profile.reviewer_chain.gates,
        *profile.loop_policy.gates,
    ]
    return tuple(dict.fromkeys(gates))


def _render_agents(agents: tuple[AgentDefinition, ...]) -> str:
    """Render selected agent roles."""
    return "\n".join(
        f"- {agent.name}: {agent.purpose} Codex={_yes_no(value=agent.may_execute_codex)}, edits={_yes_no(value=agent.may_edit_files)}"
        for agent in agents
    )


def _render_skills(skills: tuple[SkillDefinition, ...]) -> str:
    """Render selected skills and their triggers."""
    if not skills:
        return "- (none)"
    return "\n".join(f"- {skill.name}: {skill.role} Trigger: {skill.trigger}" for skill in skills)


def _render_bullets(items: tuple[str, ...]) -> str:
    """Render bullet lines."""
    return "\n".join(f"- {item}" for item in items)


def _yes_no(*, value: bool) -> str:
    """Render booleans for plan text."""
    return "yes" if value else "no"
