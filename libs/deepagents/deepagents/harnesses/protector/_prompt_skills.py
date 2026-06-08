"""Prompt quality skill definitions for Protector Codex prompts."""

from __future__ import annotations

from dataclasses import dataclass


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
        "If concrete scope, restrictions, or validation were provided, preserve them as hard boundaries.",
        "Keep the prompt compact and operational; avoid boilerplate and avoid generic empty task details.",
    ),
    restriction_rules=(
        "Preserve unrelated behavior and public contracts.",
        "Do not include `Files read` or `Files changed` unless explicitly requested.",
    ),
    validation_expectations=("Run the smallest focused build/test/smoke that proves the requested behavior.",),
    forbidden_generic_wording=("Task details: (none)",),
)

UI_RUNTIME_BUG_SKILL = PromptSkill(
    name="ui_runtime_bug",
    trigger_keywords=frozenset(
        {
            "button",
            "boton",
            "botón",
            "dom",
            "form",
            "inputs",
            "js",
            "modal",
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
    ),
    task_modes=frozenset({"ui_runtime_bug"}),
    observed_state="The visible UI state does not match the operator action requested by the task.",
    expected_behavior="The expected UI state should render the requested controls and stop hiding/loading them incorrectly.",
    scope_rules=("Stay on the real rendered UI path and its handler/render condition.",),
    restriction_rules=("Do not redesign the UI or navigation; change only the condition that causes the wrong render/hide/loading state.",),
    validation_expectations=("Prove the exact condition that hides or renders the button/modal/spinner/view before changing it.",),
)

PROVIDER_API_BUG_SKILL = PromptSkill(
    name="provider_api_bug",
    trigger_keywords=frozenset(
        {
            "api",
            "config",
            "config_id",
            "dispatch",
            "lleida",
            "payload",
            "provider",
            "providerstatus",
            "providercorrelationid",
            "set_config",
            "start_signature",
        }
    ),
    task_modes=frozenset({"provider_api_bug", "continuation_followup"}),
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

FORM_SECURITY_AUTOFILL_SKILL = PromptSkill(
    name="form_security_autofill_bug",
    trigger_keywords=frozenset({"autofill", "autofilled", "autofills", "autocompleta", "autocompletan", "contraseña", "email", "login"}),
    task_modes=frozenset({"ui_runtime_bug", "implementation_fix"}),
    observed_state="Email/password fields are being autofilled or prepopulated when the requested login flow should not do that.",
    expected_behavior="The login UI should preserve the requested `Email` and `contraseña` behavior without unwanted autofill side effects.",
    scope_rules=("Stay on authentication/form field behavior; do not use generic UI loading rules unless an actual loading state is involved.",),
    restriction_rules=("Do not redesign login or authentication flow semantics.",),
    validation_expectations=("Verify the `Email` and `contraseña` fields render with the expected autofill behavior.",),
    forbidden_generic_wording=("render/hide/loading condition", "spinner/loading state"),
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
    scope_rules=("Preserve the original Spanish task as evidence and keep exact Spanish UI text and identifiers uncorrupted.",),
    restriction_rules=("Do not translate identifiers or literal UI labels such as `Email`, `contraseña`, `login`, or button text.",),
    validation_expectations=("Verify Spanish accents and `ñ` remain intact in generated prompt text.",),
)

PROMPT_SKILLS = (
    BASE_PROMPT_QUALITY_SKILL,
    FORM_SECURITY_AUTOFILL_SKILL,
    UI_RUNTIME_BUG_SKILL,
    PROVIDER_API_BUG_SKILL,
    SPANISH_TASK_PRESERVATION_SKILL,
)


def select_prompt_skills(*, task_mode: str, task_tokens: frozenset[str], has_spanish_text: bool) -> tuple[PromptSkill, ...]:
    """Select prompt skills, with specific skills suppressing less precise generic skills."""
    selected: list[PromptSkill] = [BASE_PROMPT_QUALITY_SKILL]
    form_security = FORM_SECURITY_AUTOFILL_SKILL.applies_to(task_mode, task_tokens)
    for skill in PROMPT_SKILLS[1:]:
        if skill is UI_RUNTIME_BUG_SKILL and form_security:
            continue
        if skill is SPANISH_TASK_PRESERVATION_SKILL and has_spanish_text:
            selected.append(skill)
            continue
        if skill.applies_to(task_mode, task_tokens):
            selected.append(skill)
    return tuple(selected)
