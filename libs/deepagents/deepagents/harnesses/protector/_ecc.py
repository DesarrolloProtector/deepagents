"""Read-only ECC discovery for the Protector harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

ECC_REPO_ENV_VAR = "PROTECTOR_HARNESS_ECC_REPO"
ECC_CONFIG_ENV_VAR = "PROTECTOR_HARNESS_ECC_CONFIG"
_PROTECTOR_CONFIG_DIR = ".protector-harness"
_ECC_CONFIG_FILE = "ecc.json"
_RELEVANT_TERMS = frozenset(
    {
        "agentic",
        "codex",
        "context",
        "drift",
        "gate",
        "harness",
        "loop",
        "orchestrat",
        "plan",
        "planner",
        "prompt",
        "quality",
        "review",
        "reviewer",
        "session",
        "skill",
        "verify",
        "verification",
        "workflow",
    }
)
_MAX_DISPLAY_ITEMS = 12


@dataclass(frozen=True)
class EccCapability:
    """One ECC-discovered capability file."""

    name: str
    path: Path


@dataclass(frozen=True)
class EccInstallProfile:
    """One ECC install profile discovered from the ECC manifest."""

    name: str
    description: str


@dataclass(frozen=True)
class EccDiscovery:
    """Read-only ECC discovery result for status surfaces."""

    path: Path | None
    source: str
    found: bool
    agents: tuple[EccCapability, ...]
    skills: tuple[EccCapability, ...]
    profiles: tuple[EccInstallProfile, ...]
    warnings: tuple[str, ...]


def default_ecc_config_path() -> Path:
    """Return the local Protector ECC config path."""
    override = os.environ.get(ECC_CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    profile = os.environ.get("USERPROFILE")
    root = Path(profile) if profile else Path.home()
    return root / _PROTECTOR_CONFIG_DIR / _ECC_CONFIG_FILE


def discover_ecc() -> EccDiscovery:
    """Discover an ECC checkout without importing or duplicating ECC registries."""
    path, source, warnings = _locate_ecc_path()
    if path is None:
        return EccDiscovery(path=None, source=source, found=False, agents=(), skills=(), profiles=(), warnings=warnings)
    if not _looks_like_ecc_repo(path):
        warning = f"configured ECC path does not look like an ECC repo: {path}"
        return EccDiscovery(path=path, source=source, found=False, agents=(), skills=(), profiles=(), warnings=(*warnings, warning))

    agents = _discover_capability_files(path / "agents", "*.md")
    skills = _discover_skill_dirs(path / "skills")
    profiles = _discover_profiles(path / "manifests" / "install-profiles.json")
    return EccDiscovery(
        path=path,
        source=source,
        found=True,
        agents=agents,
        skills=skills,
        profiles=profiles,
        warnings=warnings,
    )


def render_ecc_status(discovery: EccDiscovery | None = None) -> str:
    """Render compact ECC discovery status for `ph ecc-status`."""
    result = discovery or discover_ecc()
    path = str(result.path) if result.path is not None else "(not found)"
    rows = [
        "Protector ECC Status",
        f"ECC path: {path}",
        f"ECC path source: {result.source}",
        f"ECC discovered: {_yes_no(value=result.found)}",
        f"Relevant agents: {len(result.agents)}",
        f"Relevant skills: {len(result.skills)}",
        f"Relevant profiles: {len(result.profiles)}",
        "Codex execution: disabled",
        "Model calls: disabled",
        "Autonomous loops: disabled",
        "Protector role: ECC-backed specialization layer",
        "Agents:",
        _render_names(tuple(agent.name for agent in result.agents)),
        "Skills:",
        _render_names(tuple(skill.name for skill in result.skills)),
        "Profiles:",
        _render_names(tuple(profile.name for profile in result.profiles)),
        "Warnings:",
        _render_names(result.warnings),
    ]
    return "\n".join(rows)


def _locate_ecc_path() -> tuple[Path | None, str, tuple[str, ...]]:
    """Locate the ECC repo from env, config, or sibling checkout."""
    env_path = os.environ.get(ECC_REPO_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve(), f"env:{ECC_REPO_ENV_VAR}", ()

    config_path = default_ecc_config_path()
    if config_path.exists():
        try:
            path = _path_from_config(config_path)
        except ValueError as exc:
            return None, f"config:{config_path}", (str(exc),)
        if path is not None:
            return path.expanduser().resolve(), f"config:{config_path}", ()

    sibling = _find_sibling_ecc()
    if sibling is not None:
        return sibling.resolve(), "sibling:ECC", ()
    return None, "not configured", ()


def _path_from_config(path: Path) -> Path | None:
    """Read an ECC path from a local JSON config."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid ECC config JSON at {path}: {exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"ECC config must be a JSON object: {path}"
        raise TypeError(msg)
    value = data.get("path") or data.get("repo") or data.get("ecc_repo")
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"ECC config path must be a string: {path}"
        raise TypeError(msg)
    return Path(value)


def _find_sibling_ecc() -> Path | None:
    """Return a sibling ECC checkout near the current working directory if present."""
    for base in (Path.cwd(), *_package_roots()):
        for parent in (base, *base.parents):
            candidate = parent / "ECC"
            if _looks_like_ecc_repo(candidate):
                return candidate
    return None


def _package_roots() -> tuple[Path, ...]:
    """Return local package roots used only for sibling ECC discovery."""
    current = Path(__file__).resolve()
    return tuple(current.parents[:8])


def _looks_like_ecc_repo(path: Path) -> bool:
    """Return whether `path` has the minimal ECC repository shape."""
    return path.is_dir() and (path / "agents").is_dir() and (path / "skills").is_dir()


def _discover_capability_files(root: Path, pattern: str) -> tuple[EccCapability, ...]:
    """List relevant ECC capability files from a directory."""
    if not root.is_dir():
        return ()
    capabilities = [
        EccCapability(name=path.stem, path=path)
        for path in sorted(root.glob(pattern), key=lambda item: item.name.lower())
        if _is_relevant(path.stem)
    ]
    return tuple(capabilities)


def _discover_skill_dirs(root: Path) -> tuple[EccCapability, ...]:
    """List relevant ECC skill directories with `SKILL.md` files."""
    if not root.is_dir():
        return ()
    capabilities = [
        EccCapability(name=path.name, path=path / "SKILL.md")
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_dir() and (path / "SKILL.md").is_file() and _is_relevant(path.name)
    ]
    return tuple(capabilities)


def _discover_profiles(path: Path) -> tuple[EccInstallProfile, ...]:
    """List relevant ECC install profiles from the ECC manifest."""
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        return ()

    discovered: list[EccInstallProfile] = []
    for name, value in sorted(profiles.items(), key=lambda item: item[0].lower()):
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        description = value.get("description", "")
        modules = value.get("modules", ())
        module_items = tuple(modules) if isinstance(modules, list) else ()
        haystack = " ".join(_string_items((name, description, *module_items)))
        if _is_relevant(haystack):
            discovered.append(EccInstallProfile(name=name, description=description if isinstance(description, str) else ""))
    return tuple(discovered)


def _is_relevant(text: str) -> bool:
    """Return whether a discovered ECC item is relevant to Protector's thin integration."""
    lowered = text.lower()
    return any(term in lowered for term in _RELEVANT_TERMS)


def _string_items(items: Iterable[object]) -> tuple[str, ...]:
    """Return only string items from a mixed iterable."""
    return tuple(item for item in items if isinstance(item, str))


def _render_names(names: tuple[str, ...]) -> str:
    """Render a compact list of discovered names."""
    if not names:
        return "- (none)"
    visible = names[:_MAX_DISPLAY_ITEMS]
    rows = [f"- {name}" for name in visible]
    remaining = len(names) - len(visible)
    if remaining > 0:
        rows.append(f"- ... {remaining} more")
    return "\n".join(rows)


def _yes_no(*, value: bool) -> str:
    """Render a boolean value for status output."""
    return "yes" if value else "no"
