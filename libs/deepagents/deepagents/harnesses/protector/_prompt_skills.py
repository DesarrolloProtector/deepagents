"""Prompt quality skill definitions for Protector Codex prompts."""

from __future__ import annotations

from dataclasses import dataclass

_NAVIGATION_STRONG_TERMS = frozenset({"accounting", "dashboard", "index", "legacy", "menu", "menus", "nav", "navigation"})
_MVP_SURFACE_STRONG_TERMS = frozenset({"complete", "mvp", "useful"})
_PROVIDER_BOOTSTRAP_TARGET_TERMS = frozenset({"bootstrap", "configure", "diagnostic", "probe", "smoke", "verify"})
_PROVIDER_CONFIG_TERMS = frozenset({"config", "config_id", "configid", "set_config"})
_NEGATIVE_PROVIDER_CONFIG_PHRASES = (
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


@dataclass(frozen=True)
class PromptSkill:
    """Prompt-shaping rules selected by task mode and keywords."""

    name: str
    trigger_keywords: frozenset[str]
    task_modes: frozenset[str]
    observed_state: str | None = None
    expected_behavior: str | None = None
    scope_rules: tuple[str, ...] = ()
    restriction_rules: tuple[str, ...] = ()
    validation_expectations: tuple[str, ...] = ()
    forbidden_generic_wording: tuple[str, ...] = ()

    def applies_to(self, task_mode: str, task_tokens: frozenset[str]) -> bool:
        """Return whether this skill applies to a classified task."""
        mode_matches = not self.task_modes or task_mode in self.task_modes
        keyword_matches = not self.trigger_keywords or bool(task_tokens & self.trigger_keywords)
        return mode_matches and keyword_matches


BASE_PROMPT_QUALITY_SKILL = PromptSkill(
    name="base_prompt_quality",
    trigger_keywords=frozenset(),
    task_modes=frozenset(),
    scope_rules=(
        "Inspect only enough code to locate the faulty condition, then fix surgically.",
        "Preserve explicit scope, restrictions, and validation as hard boundaries.",
        "Keep the prompt compact and operational; avoid boilerplate and empty task details.",
    ),
    restriction_rules=(
        "Preserve unrelated behavior and public contracts.",
        "Do not include `Files read` or `Files changed` unless explicitly requested.",
    ),
    validation_expectations=("Run the smallest focused build/test/smoke that proves the requested behavior.",),
    forbidden_generic_wording=("Task details: (none)",),
)

IMPLEMENTATION_FIX_SKILL = PromptSkill(
    name="implementation_fix",
    trigger_keywords=frozenset({"fix", "bug", "change", "modify", "remove", "eliminar", "quitar", "reparar"}),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "provider_api_bug", "continuation_followup"}),
    scope_rules=("Scoped edits are allowed only where needed for the requested behavior change.",),
    restriction_rules=("Do not turn the task into an audit-only response.",),
    validation_expectations=("Verify the changed behavior through the narrowest credible check.",),
)

REGRESSION_FIX_SKILL = PromptSkill(
    name="regression_fix",
    trigger_keywords=frozenset({"regression", "restore", "after", "missing", "disappeared", "desaparece"}),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "provider_api_bug", "continuation_followup"}),
    scope_rules=("Find the regression condition and restore the previous working behavior surgically.",),
    restriction_rules=("Protect behavior already validated as working; do not trade one fixed path for another regression.",),
    validation_expectations=("Prove both the restored behavior and the previously working behavior still hold.",),
)

CONTINUATION_FOLLOWUP_SKILL = PromptSkill(
    name="continuation_followup",
    trigger_keywords=frozenset({"continuation", "follow", "follow-up", "previous", "after"}),
    task_modes=frozenset({"continuation_followup"}),
    scope_rules=("Preserve validated previous fixes and continue only from the named remaining gap.",),
    restriction_rules=("Honor restrictions and PASS boundaries from the current prompt as must-not-touch items.",),
    validation_expectations=("Show that the follow-up fix did not reopen the previous regression.",),
)

UI_RUNTIME_BUG_SKILL = PromptSkill(
    name="ui_runtime_bug",
    trigger_keywords=frozenset(
        {
            "button",
            "boton",
            "botón",
            "dom",
            "inputs",
            "js",
            "modal",
            "razor",
            "spinner",
            "textbox",
            "ui",
            "view",
            "vista",
        }
    ),
    task_modes=frozenset({"ui_runtime_bug"}),
    observed_state="The visible UI state does not match the operator action requested by the task.",
    expected_behavior="The expected UI state should render the requested controls and stop hiding/loading them incorrectly.",
    scope_rules=("Stay on the real rendered UI path and its handler/render condition.",),
    restriction_rules=("Do not redesign the UI or navigation; change only the wrong render/hide/loading condition.",),
    validation_expectations=("Prove the exact condition that hides or renders the button/modal/spinner/view before changing it.",),
)

NAVIGATION_SURFACE_CONVERGENCE_SKILL = PromptSkill(
    name="navigation_surface_convergence",
    trigger_keywords=frozenset(
        {
            "accounting",
            "dashboard",
            "index",
            "legacy",
            "menu",
            "menus",
            "navigation",
            "nav",
            "surface",
            "view",
            "views",
            "vista",
            "vistas",
        }
    ),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "planning_only"}),
    observed_state="Navigation, menu, dashboard, or view surfaces are not converging on one coherent operator path.",
    expected_behavior="Operators should reach the same useful workflow surface consistently from navigation, menus, indexes, and legacy views.",
    scope_rules=("Trace route/menu/view entry points and converge only the requested navigation surface.",),
    restriction_rules=("Do not redesign dashboards or unrelated navigation; preserve existing permissions and routes.",),
    validation_expectations=("Verify the affected navigation entries land on the intended operational view.",),
    forbidden_generic_wording=("spinner", "button render condition"),
)

FORM_SECURITY_AUTOFILL_SKILL = PromptSkill(
    name="form_security_autofill_bug",
    trigger_keywords=frozenset(
        {
            "autocomplete",
            "autofill",
            "autofilled",
            "autofills",
            "autocompleta",
            "autocompletan",
            "contraseña",
            "email",
            "login",
            "password",
            "security",
        }
    ),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug"}),
    observed_state="Email/password fields are being autofilled or prepopulated when the requested login flow should not do that.",
    expected_behavior="The login UI should preserve the requested `Email` and `contraseña` behavior without unwanted autofill side effects.",
    scope_rules=("Stay on authentication/form field behavior; do not use generic UI runtime rules unless that state is actually involved.",),
    restriction_rules=("Do not redesign login, account creation, or authentication flow semantics.",),
    validation_expectations=("Verify the `Email` and `contraseña` fields render with the expected autofill behavior.",),
    forbidden_generic_wording=("render/hide/loading condition", "spinner/loading state"),
)

PROVIDER_API_BUG_SKILL = PromptSkill(
    name="provider_api_bug",
    trigger_keywords=frozenset(
        {
            "api",
            "dispatch",
            "lleida",
            "payload",
            "provider",
            "providerstatus",
            "providercorrelationid",
            "signature",
            "start_signature",
        }
    ),
    task_modes=frozenset({"implementation_fix", "provider_api_bug", "continuation_followup"}),
    observed_state="Provider/API state is not being translated correctly into the surrounding workflow state.",
    expected_behavior="Provider dispatch success means the provider accepted the request; it must not be treated as workflow completion.",
    scope_rules=("Stay on the provider/API-to-workflow boundary.",),
    restriction_rules=(
        "Provider dispatch success only proves provider acceptance; it does not prove signature completion or workflow completion.",
        "Protect config, payload, and dispatch behavior unless the task explicitly targets them.",
        "Do not modify ConfigId, SET_CONFIG, START_SIGNATURE, payload, or dispatch behavior unless the task specifically targets it.",
    ),
    validation_expectations=("Verify provider dispatch success remains intact and the workflow state is not incorrectly marked complete.",),
)

PROVIDER_BOOTSTRAP_DIAGNOSTIC_SKILL = PromptSkill(
    name="provider_bootstrap_diagnostic",
    trigger_keywords=frozenset({"bootstrap", "config", "config_id", "configure", "diagnostic", "probe", "set_config", "smoke"}),
    task_modes=frozenset({"diagnostic_bootstrap", "provider_api_bug", "implementation_fix"}),
    observed_state="Provider bootstrap/configuration needs bounded verification before changing runtime workflow behavior.",
    expected_behavior="Diagnostics should prove provider configuration and bootstrap state without treating acceptance as workflow completion.",
    scope_rules=("Limit work to explicit provider bootstrap, config, SET_CONFIG, or smoke-diagnostic checks.",),
    restriction_rules=("Do not modify provider payloads or workflow lifecycle unless the diagnostic proves that exact target.",),
    validation_expectations=("Report exact bounded diagnostic/config evidence and whether provider calls were made.",),
)

OPERATIONAL_WORKFLOW_CONVERGENCE_SKILL = PromptSkill(
    name="operational_workflow_convergence",
    trigger_keywords=frozenset(
        {
            "action",
            "actions",
            "activation",
            "account",
            "accounts",
            "bank",
            "cuenta",
            "cuentas",
            "entity",
            "entidad",
            "entidades",
            "pending",
            "resend",
            "signature",
            "workflow",
        }
    ),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "provider_api_bug", "continuation_followup"}),
    observed_state="The operational workflow has a dead end or inconsistent operator action path.",
    expected_behavior="The operator should have the correct next action while the workflow remains pending.",
    scope_rules=("Trace the real workflow state, action eligibility, and rendered operator surface before editing.",),
    restriction_rules=("Do not redesign the workflow or activate/complete work earlier than the existing lifecycle allows.",),
    validation_expectations=("Verify the relevant operator action appears only in the correct workflow state.",),
)

MVP_SURFACE_COMPLETION_SKILL = PromptSkill(
    name="mvp_surface_completion",
    trigger_keywords=frozenset({"complete", "mvp", "operational", "operator", "surface", "useful", "views", "vista", "vistas"}),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "planning_only"}),
    observed_state="The MVP/operator surface is incomplete or not useful enough for the daily workflow.",
    expected_behavior="The MVP surface should expose the minimum useful operational path without adding dashboard sprawl.",
    scope_rules=("Complete only the requested operator-facing surface needed for the daily workflow.",),
    restriction_rules=("Do not add dashboards, background workers, storage, autonomy, or unrelated navigation redesign.",),
    validation_expectations=("Smoke the completed surface through the operator entry point that matters.",),
)

GLOBAL_PATTERN_CHANGE_SKILL = PromptSkill(
    name="global_pattern_change",
    trigger_keywords=frozenset(
        {
            "all",
            "cualquier",
            "entidad",
            "entidades",
            "entities",
            "everywhere",
            "global",
            "lados",
            "pattern",
            "todas",
            "todos",
            "usages",
        }
    ),
    task_modes=frozenset({"implementation_fix", "ui_runtime_bug", "provider_api_bug", "continuation_followup"}),
    observed_state="The requested behavior is a repeated pattern and inconsistent implementations can drift across usages.",
    expected_behavior="All targeted usages should follow one consistent behavior while unrelated surfaces remain unchanged.",
    scope_rules=("Find the shared pattern or all targeted usages; change them consistently without broad redesign.",),
    restriction_rules=("Do not expand scope beyond the named global pattern or entity set.",),
    validation_expectations=("Verify representative usages plus at least one guard against missed pattern drift.",),
)

SPANISH_TASK_PRESERVATION_SKILL = PromptSkill(
    name="spanish_implementation_task_preservation",
    trigger_keywords=frozenset(
        {
            "añadir",
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
        }
    ),
    task_modes=frozenset(),
    scope_rules=("Preserve exact Spanish UI text and identifiers without reintroducing the raw user task.",),
    restriction_rules=("Do not translate identifiers or literal UI labels such as `Email`, `contraseña`, `login`, or button text.",),
    validation_expectations=("Verify Spanish accents and `ñ` remain intact in generated prompt text.",),
)

_SPECIFIC_SKILLS = (
    FORM_SECURITY_AUTOFILL_SKILL,
    NAVIGATION_SURFACE_CONVERGENCE_SKILL,
    PROVIDER_BOOTSTRAP_DIAGNOSTIC_SKILL,
    GLOBAL_PATTERN_CHANGE_SKILL,
    MVP_SURFACE_COMPLETION_SKILL,
    OPERATIONAL_WORKFLOW_CONVERGENCE_SKILL,
    PROVIDER_API_BUG_SKILL,
    UI_RUNTIME_BUG_SKILL,
    CONTINUATION_FOLLOWUP_SKILL,
    REGRESSION_FIX_SKILL,
    IMPLEMENTATION_FIX_SKILL,
)


def select_prompt_skills(*, task_mode: str, task_tokens: frozenset[str], has_spanish_text: bool, task_text: str = "") -> tuple[PromptSkill, ...]:
    """Select prompt skills with deterministic precedence and generic suppression."""
    selected: list[PromptSkill] = [BASE_PROMPT_QUALITY_SKILL]
    specific = [skill for skill in _SPECIFIC_SKILLS if _skill_applies(skill, task_mode, task_tokens, task_text)]
    names = {skill.name for skill in specific}

    if "form_security_autofill_bug" in names:
        specific = [skill for skill in specific if skill.name != "ui_runtime_bug"]
    if "navigation_surface_convergence" in names:
        specific = [skill for skill in specific if skill.name != "ui_runtime_bug"]
    if "global_pattern_change" in names:
        specific = [skill for skill in specific if skill.name != "ui_runtime_bug"]
    if "provider_bootstrap_diagnostic" in names:
        specific = [skill for skill in specific if skill.name != "provider_api_bug"]

    selected.extend(_dedupe_skills(specific))
    if has_spanish_text:
        selected.append(SPANISH_TASK_PRESERVATION_SKILL)
    return tuple(_dedupe_skills(selected))


def _skill_applies(skill: PromptSkill, task_mode: str, task_tokens: frozenset[str], task_text: str) -> bool:
    """Return whether a skill applies after taxonomy-specific routing rules."""
    if skill.name == "provider_bootstrap_diagnostic":
        if not skill.applies_to(task_mode, task_tokens):
            return False
        lowered = task_text.lower()
        has_negative_config_boundary = any(phrase in lowered for phrase in _NEGATIVE_PROVIDER_CONFIG_PHRASES)
        return bool(task_tokens & _PROVIDER_BOOTSTRAP_TARGET_TERMS) or bool(
            (task_tokens & _PROVIDER_CONFIG_TERMS) and not has_negative_config_boundary
        )
    if skill.name == "navigation_surface_convergence":
        return skill.applies_to(task_mode, task_tokens) and bool(task_tokens & _NAVIGATION_STRONG_TERMS)
    if skill.name == "mvp_surface_completion":
        return skill.applies_to(task_mode, task_tokens) and bool(task_tokens & _MVP_SURFACE_STRONG_TERMS)
    return skill.applies_to(task_mode, task_tokens)


def _dedupe_skills(skills: list[PromptSkill]) -> list[PromptSkill]:
    """Return skills by first occurrence of name."""
    seen: set[str] = set()
    result: list[PromptSkill] = []
    for skill in skills:
        if skill.name not in seen:
            seen.add(skill.name)
            result.append(skill)
    return result
