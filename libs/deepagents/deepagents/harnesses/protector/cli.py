"""Command-line entry point for the Protector engineering harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents._version import __version__
from deepagents.harnesses.protector._engineering import (
    HARNESS_PROFILE,
    HarnessUsageError,
    build_read_only_agent,
    render_output,
    render_review_findings,
    resolve_harness_profile,
    write_prompt_output,
)
from deepagents.profiles.harness.harness_profiles import _get_harness_profile

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOS_CONFIG_ENV_VAR = "PROTECTOR_HARNESS_REPOS"
_REPOS_CONFIG_DIR = ".protector-harness"
_REPOS_CONFIG_FILE = "repos.json"
_BUILT_IN_REPO_ALIASES = {
    "FinanciacionCore": Path(r"C:\Users\DesarrolladorProtect\source\repos\FinanciacionCore"),
    "deepagents": Path(r"C:\Users\DesarrolladorProtect\source\repos\deepagents"),
}
_MIN_POSITIONAL_REPO_TASK_ARGS = 2


@dataclass(frozen=True)
class _ClipboardCopyResult:
    """Result of copying generated prompt text to the system clipboard."""

    copied: bool
    error: str | None = None


def _repos_config_path() -> Path:
    """Return the local repo-alias config path."""
    override = os.environ.get(REPOS_CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    profile = os.environ.get("USERPROFILE")
    root = Path(profile) if profile else Path.home()
    return root / _REPOS_CONFIG_DIR / _REPOS_CONFIG_FILE


def _repo_alias_output(path: Path) -> str:
    """Format an alias path for JSON or display output."""
    return str(path)


def _default_repo_alias_config() -> dict[str, str]:
    """Return the built-in aliases in the supported config JSON shape."""
    return {name: _repo_alias_output(path) for name, path in _BUILT_IN_REPO_ALIASES.items()}


def _load_config_aliases(parser: argparse.ArgumentParser) -> dict[str, Path]:
    """Load local repo aliases from JSON without scanning repositories."""
    path = _repos_config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"invalid repo alias config JSON at {path}: {exc.msg}")
    if not isinstance(data, dict):
        parser.error(f"repo alias config must be a JSON object: {path}")

    aliases: dict[str, Path] = {}
    for name, value in data.items():
        if not isinstance(name, str) or not isinstance(value, str):
            parser.error(f"repo alias config entries must be string path mappings: {path}")
        aliases[name] = Path(value).expanduser()
    return aliases


def _repo_aliases(parser: argparse.ArgumentParser) -> dict[str, Path]:
    """Return config aliases over built-in fallback aliases."""
    return {**_BUILT_IN_REPO_ALIASES, **_load_config_aliases(parser)}


def _resolve_explicit_repo(repo: str, parser: argparse.ArgumentParser) -> Path:
    """Resolve `--repo` as an explicit path."""
    resolved = Path(repo).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        parser.error(f"repo path does not exist or is not a directory: {repo}")
    return resolved


def _resolve_positional_repo(repo: str, parser: argparse.ArgumentParser) -> Path:
    """Resolve a positional repo argument as alias or existing path."""
    aliases = _repo_aliases(parser)
    alias = aliases.get(repo)
    candidate = alias if alias is not None else Path(repo)
    resolved = candidate.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        parser.error(f"unknown repo alias or missing repo path: {repo}")
    return resolved


def _resolve_repo_and_task(
    *,
    explicit_repo: str | None,
    positional: list[str],
    parser: argparse.ArgumentParser,
) -> tuple[Path | None, str]:
    """Resolve repo from `--repo` or leading positional alias/path plus task."""
    if explicit_repo is not None:
        task = _task_text(positional)
        if not task:
            parser.error("task text is required")
        return _resolve_explicit_repo(explicit_repo, parser), task

    if len(positional) < _MIN_POSITIONAL_REPO_TASK_ARGS:
        parser.error("repo alias/path and task text are required unless --repo is used")
    repo = _resolve_positional_repo(positional[0], parser)
    task = _task_text(positional[1:])
    if not task:
        parser.error("task text is required")
    return repo, task


def _task_text(parts: list[str]) -> str:
    """Join task words from argparse into a compact task string."""
    return " ".join(parts).strip()


def _build_parser() -> argparse.ArgumentParser:
    """Build the `ph` argument parser."""
    parser = argparse.ArgumentParser(
        prog="ph",
        description="Protector engineering harness utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    task = subparsers.add_parser("task", help="Generate a Codex-ready engineering prompt.")
    task.add_argument("--repo", default=None, help="Explicit target repository path.")
    task.add_argument("--output", type=Path, default=None, help="Optional file path for writing only the generated Codex Prompt section.")
    task.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file.")
    task.add_argument("--no-copy", action="store_true", help="Print the generated prompt instead of copying it to the clipboard.")
    task.add_argument("repo_or_task", help="Repo alias/path, or the first task word when --repo is used.")
    task.add_argument("task", nargs="*", help="Task text to turn into a compact Codex handoff.")

    review = subparsers.add_parser("review", help="Review a pasted Codex output deterministically.")
    review.add_argument("--repo", default=None, help="Explicit target repository path.")
    review.add_argument("--codex-output", type=Path, required=True, help="Text file containing pasted Codex output to review.")
    review.add_argument("repo_or_task", help="Repo alias/path, or the first task word when --repo is used.")
    review.add_argument("task", nargs="*", help="Task text used to detect scope drift.")

    repos = subparsers.add_parser("repos", help="Show or initialize local repo aliases.")
    repos.add_argument("--init", action="store_true", help="Create the local repo alias config with built-in aliases.")
    repos.add_argument("--overwrite", action="store_true", help="Allow --init to replace an existing config file.")

    subparsers.add_parser("status", help="Show Protector harness CLI status.")
    return parser


def _run_task(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph task`."""
    repo, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)

    try:
        harness_profile = resolve_harness_profile(HARNESS_PROFILE)
    except HarnessUsageError as exc:
        parser.error(str(exc))

    profile = _get_harness_profile(harness_profile)
    if profile is None:
        parser.error(f"built-in harness profile {harness_profile!r} is not registered")

    agent = build_read_only_agent(harness_profile)
    rendered = render_output(
        task=task,
        repo=repo,
        mode="auto",
        harness_profile=harness_profile,
        real_model=None,
        agent_type=type(agent).__name__,
        output=args.output,
    )
    if args.output is not None:
        try:
            write_prompt_output(args.output, rendered.codex_prompt, overwrite=args.overwrite)
        except HarnessUsageError as exc:
            parser.error(str(exc))
    if args.no_copy:
        sys.stdout.write(rendered.payload)
        sys.stdout.write("\n")
        return 0

    copy_result = _copy_to_clipboard(rendered.codex_prompt)
    if copy_result.copied:
        sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex prompt copied to clipboard"))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex prompt was not copied"))
    sys.stdout.write(f"\nClipboard unavailable: {copy_result.error or 'no clipboard backend available'}\n\n")
    sys.stdout.write(rendered.payload)
    sys.stdout.write("\n")
    return 0


def _copy_to_clipboard(text: str) -> _ClipboardCopyResult:
    """Copy `text` to the system clipboard using a fixed local command."""
    command = _clipboard_command()
    if command is None:
        return _ClipboardCopyResult(copied=False, error="no clipboard command found")

    try:
        subprocess.run(  # noqa: S603  # Command is selected from fixed clipboard backends with `shell=False`.
            command,
            input=text,
            text=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        msg = stderr or f"{Path(command[0]).name} exited with status {exc.returncode}"
        return _ClipboardCopyResult(copied=False, error=msg)
    except OSError as exc:
        return _ClipboardCopyResult(copied=False, error=str(exc) or type(exc).__name__)

    return _ClipboardCopyResult(copied=True)


def _clipboard_command() -> tuple[str, ...] | None:
    """Return the first available local clipboard command."""
    if sys.platform == "win32":
        candidates = (("clip.exe",), ("clip",))
    elif sys.platform == "darwin":
        candidates = (("pbcopy",),)
    else:
        candidates = (
            ("wl-copy",),
            ("xclip", "-selection", "clipboard"),
            ("xsel", "--clipboard", "--input"),
        )

    for candidate in candidates:
        executable = shutil.which(candidate[0])
        if executable is not None:
            return (executable, *candidate[1:])
    return None


def _render_task_confirmation(repo: Path | None, selected_context_count: int, status: str) -> str:
    """Render compact `ph task` success metadata."""
    repo_text = str(repo.resolve()) if repo is not None else "(not provided)"
    return f"""Repo: {repo_text}
Selected context count: {selected_context_count}
{status}"""


def _run_review(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph review`."""
    _repo, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)

    text = args.codex_output.read_text(encoding="utf-8")
    sys.stdout.write(render_review_findings(args.codex_output, task, text))
    sys.stdout.write("\n")
    return 0


def _run_status() -> int:
    """Run `ph status`."""
    profile = _get_harness_profile(HARNESS_PROFILE)
    registered = "yes" if profile is not None else "no"
    sys.stdout.write(
        f"""Protector Harness Status
deepagents version: {__version__}
profile {HARNESS_PROFILE} registered: {registered}
cwd: {Path.cwd()}
cli module: {Path(__file__).resolve()}
"""
    )
    return 0


def _run_repos(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph repos`."""
    path = _repos_config_path()
    if args.init:
        if path.exists() and not args.overwrite:
            parser.error(f"repo alias config already exists: {path}; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{json.dumps(_default_repo_alias_config(), indent=2)}\n", encoding="utf-8")

    aliases = _repo_aliases(parser)
    status = "exists" if path.exists() else "missing"
    rows = "\n".join(f"- {name}: {alias.expanduser().resolve()}" for name, alias in sorted(aliases.items()))
    sys.stdout.write(
        f"""Protector Harness Repos
config: {path} ({status})
Aliases:
{rows}
"""
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `ph` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "task":
        return _run_task(args, parser)
    if args.command == "review":
        return _run_review(args, parser)
    if args.command == "status":
        return _run_status()
    if args.command == "repos":
        return _run_repos(args, parser)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
