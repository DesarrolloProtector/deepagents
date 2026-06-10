"""Prompt quality skill definitions for Protector Codex prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_NAVIGATION_STRONG_TERMS = frozenset({"accounting", "dashboard", "index", "legacy", "menu", "menus", "nav", "navigation"})
_MVP_SURFACE_STRONG_TERMS = frozenset({"complete", "mvp", "useful"})
_PROVIDER_BOOTSTRAP_TARGET_TERMS = frozenset({"bootstrap", "configure", "diagnostic", "probe", "smoke", "verify"})
_PROVIDER_CONFIG_TERMS = frozenset({"config", "config_id", "configid", "set_config"})
_LOCALIZATION_STRONG_TERMS = frozenset(
    {
        "english",
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
_LOCALIZATION_LANGUAGE_PAIR_PHRASES = ("es/en", "en/es", "spanish/english", "english/spanish")
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
_PROTECTOR_PACK_RELATIVE_PATH = Path("packs") / "protector-financiacioncore"
_PACK_PROMPT_SKILLS_RELATIVE_PATH = Path("verification") / "prompt-skills.json"
_PACK_OWNED_PROMPT_SKILL_NAMES = frozenset(
    {
        "form_security_autofill_bug",
        "navigation_surface_convergence",
        "provider_bootstrap_diagnostic",
        "provider_api_bug",
        "operational_workflow_convergence",
        "global_pattern_change",
        "localization_completion",
        "mvp_surface_completion",
        "spanish_implementation_task_preservation",
    }
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
    source: str = "runtime_generic"

    def applies_to(self, task_mode: str, task_tokens: frozenset[str]) -> bool:
        """Return whether this skill applies to a classified task."""
        mode_matches = not self.task_modes or task_mode in self.task_modes
        keyword_matches = not self.trigger_keywords or bool(task_tokens & self.trigger_keywords)
        return mode_matches and keyword_matches


def _load_pack_prompt_skill(name: str) -> PromptSkill:
    """Load one pack-owned prompt skill from the Protector ECC pack."""
    metadata = _load_pack_prompt_skill_metadata()
    for item in _metadata_items(metadata):
        if item.get("name") == name:
            return _prompt_skill_from_metadata(item)
    msg = f"Protector pack prompt skill metadata is missing required skill: {name}"
    raise RuntimeError(msg)


def _load_pack_prompt_skill_metadata() -> dict[str, object]:
    """Return pack-owned prompt skill metadata."""
    path = _find_pack_prompt_skill_metadata()
    if path is None:
        msg = f"Protector pack prompt skill metadata not found: {_PACK_PROMPT_SKILLS_RELATIVE_PATH}"
        raise RuntimeError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid Protector pack prompt skill metadata JSON at {path}: {exc.msg}"
        raise RuntimeError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Protector pack prompt skill metadata must be a JSON object: {path}"
        raise TypeError(msg)
    return data


def _find_pack_prompt_skill_metadata() -> Path | None:
    """Find the repo-local Protector pack prompt-skill metadata file."""
    for base in (Path.cwd(), *_package_roots()):
        for parent in (base, *base.parents):
            candidate = parent / _PROTECTOR_PACK_RELATIVE_PATH / _PACK_PROMPT_SKILLS_RELATIVE_PATH
            if candidate.is_file():
                return candidate.resolve()
    return None


def _package_roots() -> tuple[Path, ...]:
    """Return package roots used only for locating the repo-local pack."""
    current = Path(__file__).resolve()
    return tuple(current.parents[:8])


def _metadata_items(data: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Return prompt skill metadata entries."""
    value = data.get("prompt_skills")
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _prompt_skill_from_metadata(item: dict[str, object]) -> PromptSkill:
    """Convert pack metadata into the runtime prompt-skill model."""
    name = _metadata_string(item, "name")
    if name not in _PACK_OWNED_PROMPT_SKILL_NAMES:
        msg = f"unexpected pack-owned prompt skill: {name}"
        raise RuntimeError(msg)
    return PromptSkill(
        name=name,
        trigger_keywords=frozenset(_metadata_string_items(item, "trigger_keywords")),
        task_modes=frozenset(_metadata_string_items(item, "task_modes")),
        observed_state=_metadata_optional_string(item, "observed_state"),
        expected_behavior=_metadata_optional_string(item, "expected_behavior"),
        scope_rules=_metadata_string_items(item, "scope_rules"),
        restriction_rules=_metadata_string_items(item, "restriction_rules"),
        validation_expectations=_metadata_string_items(item, "validation_expectations"),
        forbidden_generic_wording=_metadata_string_items(item, "forbidden_generic_wording"),
        source="pack",
    )


def _metadata_string(item: dict[str, object], key: str) -> str:
    """Return a required metadata string."""
    value = item.get(key)
    if isinstance(value, str) and value:
        return value
    msg = f"Protector pack prompt skill metadata must include string field: {key}"
    raise RuntimeError(msg)


def _metadata_optional_string(item: dict[str, object], key: str) -> str | None:
    """Return an optional metadata string."""
    value = item.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"Protector pack prompt skill metadata field must be string or null: {key}"
    raise RuntimeError(msg)


def _metadata_string_items(item: dict[str, object], key: str) -> tuple[str, ...]:
    """Return a required metadata list of strings."""
    value = item.get(key)
    if isinstance(value, list) and all(isinstance(entry, str) for entry in value):
        return tuple(value)
    msg = f"Protector pack prompt skill metadata must include string list field: {key}"
    raise RuntimeError(msg)


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
    trigger_keywords=frozenset(
        {"fix", "bug", "change", "complete", "completion", "modify", "remove", "eliminar", "quitar", "reparar"}
    ),
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

NAVIGATION_SURFACE_CONVERGENCE_SKILL = _load_pack_prompt_skill("navigation_surface_convergence")

FORM_SECURITY_AUTOFILL_SKILL = _load_pack_prompt_skill("form_security_autofill_bug")

PROVIDER_API_BUG_SKILL = _load_pack_prompt_skill("provider_api_bug")

PROVIDER_BOOTSTRAP_DIAGNOSTIC_SKILL = _load_pack_prompt_skill("provider_bootstrap_diagnostic")

OPERATIONAL_WORKFLOW_CONVERGENCE_SKILL = _load_pack_prompt_skill("operational_workflow_convergence")

MVP_SURFACE_COMPLETION_SKILL = _load_pack_prompt_skill("mvp_surface_completion")

GLOBAL_PATTERN_CHANGE_SKILL = _load_pack_prompt_skill("global_pattern_change")

LOCALIZATION_COMPLETION_SKILL = _load_pack_prompt_skill("localization_completion")

SPANISH_TASK_PRESERVATION_SKILL = _load_pack_prompt_skill("spanish_implementation_task_preservation")

_PACK_SPECIALIZATION_SKILLS = (
    FORM_SECURITY_AUTOFILL_SKILL,
    NAVIGATION_SURFACE_CONVERGENCE_SKILL,
    PROVIDER_BOOTSTRAP_DIAGNOSTIC_SKILL,
    GLOBAL_PATTERN_CHANGE_SKILL,
    MVP_SURFACE_COMPLETION_SKILL,
    OPERATIONAL_WORKFLOW_CONVERGENCE_SKILL,
    PROVIDER_API_BUG_SKILL,
    LOCALIZATION_COMPLETION_SKILL,
)
_RUNTIME_GENERIC_FALLBACK_SKILLS = (
    UI_RUNTIME_BUG_SKILL,
    CONTINUATION_FOLLOWUP_SKILL,
    REGRESSION_FIX_SKILL,
    IMPLEMENTATION_FIX_SKILL,
)


def available_prompt_skills() -> tuple[PromptSkill, ...]:
    """Return all prompt skills that an exported candidate may reference."""
    return (
        BASE_PROMPT_QUALITY_SKILL,
        *_PACK_SPECIALIZATION_SKILLS,
        SPANISH_TASK_PRESERVATION_SKILL,
        *_RUNTIME_GENERIC_FALLBACK_SKILLS,
    )


def select_prompt_skills(*, task_mode: str, task_tokens: frozenset[str], has_spanish_text: bool, task_text: str = "") -> tuple[PromptSkill, ...]:
    """Select pack-owned specialization skills first, then runtime generic fallback skills."""
    selected: list[PromptSkill] = [BASE_PROMPT_QUALITY_SKILL]
    if task_mode == "ui_visual_microfix":
        return tuple(selected)
    pack_specific = [skill for skill in _PACK_SPECIALIZATION_SKILLS if _skill_applies(skill, task_mode, task_tokens, task_text)]
    runtime_fallback = [skill for skill in _RUNTIME_GENERIC_FALLBACK_SKILLS if _skill_applies(skill, task_mode, task_tokens, task_text)]
    specific = [*pack_specific, *runtime_fallback]
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
    if skill.name == "localization_completion":
        lowered = task_text.lower()
        has_language_pair = any(phrase in lowered for phrase in _LOCALIZATION_LANGUAGE_PAIR_PHRASES)
        raw_tokens = _raw_task_tokens(task_text)
        return skill.applies_to(task_mode, task_tokens) and (bool(raw_tokens & _LOCALIZATION_STRONG_TERMS) or has_language_pair)
    return skill.applies_to(task_mode, task_tokens)


def _raw_task_tokens(text: str) -> frozenset[str]:
    """Return lowercase task tokens without alias expansion."""
    return frozenset(token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if len(token) > 1)


def _dedupe_skills(skills: list[PromptSkill]) -> list[PromptSkill]:
    """Return skills by first occurrence of name."""
    seen: set[str] = set()
    result: list[PromptSkill] = []
    for skill in skills:
        if skill.name not in seen:
            seen.add(skill.name)
            result.append(skill)
    return result
