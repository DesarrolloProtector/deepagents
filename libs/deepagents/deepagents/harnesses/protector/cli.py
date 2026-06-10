"""Command-line entry point for the Protector engineering harness."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents._version import __version__
from deepagents.harnesses.protector._ecc import render_ecc_status
from deepagents.harnesses.protector._engineering import (
    HARNESS_PROFILE,
    HarnessUsageError,
    RenderedOutput,
    append_outcome_history,
    automation_candidate_approval_sha,
    automation_candidate_codex_prompt,
    automation_candidate_readiness_classification,
    build_read_only_agent,
    render_automation_candidate_dry_run,
    render_candidate_outcome_report,
    render_codex_reviewer_prompt,
    render_controlled_execution_plan,
    render_outcome_history,
    render_outcome_learning_signals,
    render_output,
    render_prompt_benchmark_report,
    render_review_findings,
    render_supervised_outcome_report,
    resolve_harness_profile,
    review_codex_output,
    run_prompt_benchmarks,
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
_INTERACTIVE_SENTINEL = "END"
_GMEM_MOVEABLE = 0x0002
_CF_UNICODETEXT = 13
_DEFAULT_CODEX_COMMAND = "codex exec -"
STABLE_ECC_PACK_COMMANDS = (
    "task",
    "review",
    "review-codex",
    "outcome",
    "outcome-history",
    "outcome-learning",
    "candidate-dry-run",
    "candidate-outcome",
    "candidate-execute",
    "benchmark",
    "ecc-status",
)
DEPRECATED_PLATFORM_COMPATIBILITY_COMMANDS = ("plan",)


@dataclass(frozen=True)
class _ClipboardCopyResult:
    """Result of copying generated prompt text to the system clipboard."""

    copied: bool
    error: str | None = None


@dataclass(frozen=True)
class _ResolvedRepo:
    """Resolved repository path plus the alias token when one was used."""

    path: Path
    alias: str | None = None


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
    return _resolve_positional_repo_with_alias(repo, parser).path


def _resolve_positional_repo_with_alias(repo: str, parser: argparse.ArgumentParser) -> _ResolvedRepo:
    """Resolve a positional repo argument as alias or existing path."""
    aliases = _repo_aliases(parser)
    alias = aliases.get(repo)
    candidate = alias if alias is not None else Path(repo)
    resolved = candidate.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        parser.error(f"unknown repo alias or missing repo path: {repo}")
    return _ResolvedRepo(path=resolved, alias=repo if alias is not None else None)


def _resolve_repo_and_task(
    *,
    explicit_repo: str | None,
    positional: list[str],
    parser: argparse.ArgumentParser,
) -> tuple[Path | None, str | None, str]:
    """Resolve repo from `--repo` or leading positional alias/path plus task."""
    if explicit_repo is not None:
        task = _task_text(positional)
        if not task:
            parser.error("task text is required")
        return _resolve_explicit_repo(explicit_repo, parser), None, task

    if len(positional) < _MIN_POSITIONAL_REPO_TASK_ARGS:
        parser.error("repo alias/path and task text are required unless --repo is used")
    resolved = _resolve_positional_repo_with_alias(positional[0], parser)
    task = _task_text(positional[1:])
    if not task:
        parser.error("task text is required")
    return resolved.path, resolved.alias, task


def _resolve_history_repo(
    *,
    explicit_repo: str | None,
    positional_repo: str | None,
    parser: argparse.ArgumentParser,
) -> Path:
    """Resolve a repo for repo-scoped outcome history commands."""
    if explicit_repo is not None:
        return _resolve_explicit_repo(explicit_repo, parser)
    if positional_repo is None:
        parser.error("repo alias/path is required unless --repo is used")
    return _resolve_positional_repo(positional_repo, parser)


def _resolve_history_repo_and_optional_task(
    *,
    explicit_repo: str | None,
    positional: list[str],
    parser: argparse.ArgumentParser,
) -> tuple[Path, str | None]:
    """Resolve a repo plus optional task text for history-learning commands."""
    if explicit_repo is not None:
        task = _task_text(positional)
        return _resolve_explicit_repo(explicit_repo, parser), task or None
    if not positional:
        parser.error("repo alias/path is required unless --repo is used")
    repo = _resolve_positional_repo(positional[0], parser)
    task = _task_text(positional[1:])
    return repo, task or None


def _task_text(parts: list[str]) -> str:
    """Join task words from argparse into a compact task string."""
    return " ".join(parts).strip()


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915  # explicit subcommand declarations keep CLI contracts inspectable
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

    review_codex = subparsers.add_parser("review-codex", help="Generate a Codex reviewer prompt for a Codex output.")
    review_codex.add_argument("--repo", default=None, help="Explicit target repository path.")
    review_codex.add_argument("--codex-output", type=Path, required=True, help="Text file containing Codex implementation output to review.")
    review_codex.add_argument("--output", type=Path, default=None, help="Optional file path for writing only the generated Codex reviewer prompt.")
    review_codex.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file.")
    review_codex.add_argument("--no-copy", action="store_true", help="Print the generated reviewer prompt instead of copying it to the clipboard.")
    review_codex.add_argument("repo_or_task", help="Repo alias/path, or the first task word when --repo is used.")
    review_codex.add_argument("task", nargs="*", help="Original task text used to build the reviewer prompt.")

    outcome = subparsers.add_parser("outcome", help="Compare pasted Codex output against the ECC supervised plan/review contract.")
    outcome.add_argument("--repo", default=None, help="Explicit target repository path.")
    outcome.add_argument("--codex-output", type=Path, required=True, help="Text file containing Codex implementation output to capture.")
    outcome.add_argument("--save-history", action="store_true", help="Append a structured outcome summary to repo-local ECC history.")
    outcome.add_argument("repo_or_task", help="Repo alias/path, or the first task word when --repo is used.")
    outcome.add_argument("task", nargs="*", help="Original planned task text used to compare the outcome.")

    outcome_history = subparsers.add_parser("outcome-history", help="List recent repo-local ECC supervised outcome history.")
    outcome_history.add_argument("--repo", default=None, help="Explicit target repository path.")
    outcome_history.add_argument("--limit", type=int, default=10, help="Maximum number of recent history entries to show.")
    outcome_history.add_argument("repo_or_alias", nargs="?", help="Repo alias/path when --repo is not used.")

    outcome_learning = subparsers.add_parser("outcome-learning", help="Inspect derived ECC supervised outcome learning signals.")
    outcome_learning.add_argument("--repo", default=None, help="Explicit target repository path.")
    outcome_learning.add_argument("--limit", type=int, default=50, help="Maximum number of recent history entries to analyze.")
    outcome_learning.add_argument("repo_or_task", nargs="?", help="Repo alias/path, or the first task word when --repo is used.")
    outcome_learning.add_argument("task", nargs="*", help="Optional task text used to filter signals to selected skills.")

    plan = subparsers.add_parser(
        "plan",
        help="Compatibility-only execution-plan view; ECC owns future planning/orchestration.",
    )
    plan.add_argument("--repo", default=None, help="Explicit target repository path.")
    plan.add_argument("--with-history", action="store_true", help="Include recent repo-scoped supervised outcome signals.")
    plan.add_argument("--candidate-json", type=Path, default=None, help="Optional path for writing a structured automation candidate JSON.")
    plan.add_argument("--overwrite", action="store_true", help="Allow --candidate-json to replace an existing file.")
    plan.add_argument("repo_or_task", help="Repo alias/path, or the first task word when --repo is used.")
    plan.add_argument("task", nargs="*", help="Task text to turn into a controlled execution plan.")

    candidate_dry_run = subparsers.add_parser(
        "candidate-dry-run",
        help="Validate an exported automation candidate JSON without execution.",
    )
    candidate_dry_run.add_argument("candidate_json", type=Path, help="Exported automation candidate JSON to validate.")

    candidate_outcome = subparsers.add_parser(
        "candidate-outcome",
        help="Review Codex output against an exported automation candidate JSON.",
    )
    candidate_outcome.add_argument("--repo", default=None, help="Optional repository path or alias for --save-history.")
    candidate_outcome.add_argument("--codex-output", type=Path, required=True, help="Text file containing Codex implementation output to review.")
    candidate_outcome.add_argument("--save-history", action="store_true", help="Append a structured outcome summary to repo-local ECC history.")
    candidate_outcome.add_argument("candidate_json", type=Path, help="Exported automation candidate JSON to use as the review contract.")

    candidate_execute = subparsers.add_parser(
        "candidate-execute",
        help="Run one explicitly approved foreground Codex execution for an automation-ready candidate.",
    )
    candidate_execute.add_argument("--approve-sha", default=None, help="Required canonical SHA-256 approval token for the candidate JSON.")
    candidate_execute.add_argument(
        "--codex-cmd",
        default=_DEFAULT_CODEX_COMMAND,
        help=f"Codex command to invoke once. Default: {_DEFAULT_CODEX_COMMAND!r}.",
    )
    candidate_execute.add_argument("--output", type=Path, default=None, help="Optional file path for persisting raw Codex output.")
    candidate_execute.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file.")
    candidate_execute.add_argument("--repo", default=None, help="Optional repository path or alias shown in the next candidate-outcome command.")
    candidate_execute.add_argument("candidate_json", type=Path, help="Automation-ready candidate JSON to execute once.")

    run = subparsers.add_parser("run", help="Run the interactive prompt/review workflow without invoking Codex.")
    run.add_argument("repo", help="Repo alias/path to target.")

    repos = subparsers.add_parser("repos", help="Show or initialize local repo aliases.")
    repos.add_argument("--init", action="store_true", help="Create the local repo alias config with built-in aliases.")
    repos.add_argument("--overwrite", action="store_true", help="Allow --init to replace an existing config file.")

    benchmark = subparsers.add_parser("benchmark", help="Run prompt-quality benchmark fixtures.")
    benchmark.add_argument("--benchmarks", type=Path, default=None, help="Optional prompt benchmark fixture directory.")
    benchmark.add_argument("--repo", default=None, help="Optional repository path or alias used for context selection.")

    subparsers.add_parser("ecc-status", help="Show read-only ECC discovery status.")
    subparsers.add_parser("status", help="Show Protector harness CLI status.")
    return parser


def _run_task(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph task`."""
    repo, repo_alias, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)
    rendered = _render_task_prompt(repo=repo, repo_alias=repo_alias, task=task, output=args.output, parser=parser)
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


def _run_interactive(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph run`."""
    _ = args, parser
    sys.stdout.write(
        "ph run is temporarily disabled. Use ph task to generate/copy prompts and ph review/ph review-codex for review.\n",
    )
    return 1


def _run_interactive_enabled(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the disabled interactive workflow implementation."""
    repo = _resolve_positional_repo(args.repo, parser)
    task = _read_interactive_task(parser)
    task = _confirm_interactive_task(task, parser)
    if task is None:
        sys.stdout.write("Cancelled.\n")
        return 0

    rendered = _render_task_prompt(repo=repo, repo_alias=args.repo, task=task, output=None, parser=parser)
    copy_result = _copy_to_clipboard(rendered.codex_prompt)
    if copy_result.copied:
        sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex prompt copied to clipboard"))
    else:
        sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex prompt was not copied"))
        sys.stdout.write(f"\nClipboard unavailable: {copy_result.error or 'no clipboard backend available'}\n\n")
        sys.stdout.write(rendered.payload)
    sys.stdout.write("\nPaste this into your already-open Codex session.\n")

    codex_output = _read_until_sentinel(
        f"Paste Codex output. End input with a line containing only {_INTERACTIVE_SENTINEL}.\n",
    )
    findings = review_codex_output(task=task, output=codex_output, source="interactive paste")
    sys.stdout.write(findings.text)
    sys.stdout.write("\n")

    if findings.status == "REVIEW_NEEDED" and findings.follow_up is not None:
        follow_up_copy = _copy_to_clipboard(findings.follow_up)
        if follow_up_copy.copied:
            sys.stdout.write("Follow-up prompt copied to clipboard.\n")
        else:
            sys.stdout.write(f"Follow-up prompt was not copied: {follow_up_copy.error or 'no clipboard backend available'}\n")
        sys.stdout.write("Follow-up prompt:\n")
        sys.stdout.write(f"{findings.follow_up}\n")
    elif findings.status == "PASS":
        sys.stdout.write("PASS\n")
    return 0


def _read_interactive_task(parser: argparse.ArgumentParser) -> str:
    """Read a non-empty interactive task from stdin."""
    task = _read_until_sentinel(
        f"Paste or type the task. End input with a line containing only {_INTERACTIVE_SENTINEL}.\n",
    )
    if not task:
        parser.error("task text is required")
    return task


def _confirm_interactive_task(task: str, parser: argparse.ArgumentParser) -> str | None:
    """Let the operator generate, edit, re-enter, or cancel the task."""
    current = task
    while True:
        sys.stdout.write("Task to generate:\n")
        sys.stdout.write(f"{current}\n")
        sys.stdout.write("Choose: G generate, E edit, R re-enter, C cancel: ")
        choice = sys.stdin.readline().strip().upper()
        if choice == "G":
            return current
        if choice == "E":
            current = _edit_interactive_task(current, parser)
            continue
        if choice == "R":
            current = _read_interactive_task(parser)
            continue
        if choice == "C":
            return None
        if choice == "":
            parser.error("task confirmation choice is required")
        sys.stdout.write("Choose G, E, R, or C.\n")


def _edit_interactive_task(task: str, parser: argparse.ArgumentParser) -> str:
    """Edit task text in the configured editor and return the UTF-8 result."""
    with tempfile.TemporaryDirectory(prefix="protector-harness-task-") as tmp:
        path = Path(tmp) / "task.md"
        path.write_text(f"{task.rstrip()}\n", encoding="utf-8")
        command = _editor_command()
        try:
            _run_editor(command, path)
        except (OSError, subprocess.CalledProcessError) as exc:
            parser.error(f"editor failed: {exc}")
        edited = path.read_text(encoding="utf-8").strip()
    if not edited:
        parser.error("edited task text is empty")
    return edited


def _editor_command() -> tuple[str, ...]:
    """Return the configured editor command for interactive task edits."""
    for env_var in ("PROTECTOR_HARNESS_EDITOR", "VISUAL", "EDITOR"):
        value = os.environ.get(env_var)
        if value:
            return tuple(shlex.split(value, posix=sys.platform != "win32"))
    code = shutil.which("code")
    if code is not None:
        return (code, "--wait")
    return ("notepad",)


def _run_editor(command: tuple[str, ...], path: Path) -> None:
    """Run an editor command against the temp task file."""
    subprocess.run(  # noqa: S603  # Editor command is explicit operator configuration or fixed fallback.
        [*command, str(path)],
        check=True,
    )


def _render_task_prompt(
    *,
    repo: Path | None,
    repo_alias: str | None = None,
    task: str,
    output: Path | None,
    parser: argparse.ArgumentParser,
) -> RenderedOutput:
    """Build the Codex prompt using the harness profile dry-run path."""
    try:
        harness_profile = resolve_harness_profile(HARNESS_PROFILE)
    except HarnessUsageError as exc:
        parser.error(str(exc))

    profile = _get_harness_profile(harness_profile)
    if profile is None:
        parser.error(f"built-in harness profile {harness_profile!r} is not registered")

    agent = build_read_only_agent(harness_profile)
    return render_output(
        task=task,
        repo=repo,
        repo_alias=repo_alias,
        mode="auto",
        harness_profile=harness_profile,
        real_model=None,
        agent_type=type(agent).__name__,
        output=output,
    )


def _read_until_sentinel(prompt: str) -> str:
    """Read stdin until the interactive sentinel line."""
    sys.stdout.write(prompt)
    lines: list[str] = []
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
        if line.rstrip("\r\n") == _INTERACTIVE_SENTINEL:
            break
        lines.append(line)
    return "".join(lines).strip()


def _copy_to_clipboard(text: str) -> _ClipboardCopyResult:
    """Copy `text` to the system clipboard using a fixed local command."""
    if sys.platform == "win32":
        return _copy_to_windows_clipboard(text)

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


def _copy_to_windows_clipboard(text: str) -> _ClipboardCopyResult:
    """Copy text to the Windows Unicode clipboard."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_int
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.restype = ctypes.c_int
    data = f"{text}\0".encode("utf-16-le")
    handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
    if not handle:
        return _ClipboardCopyResult(copied=False, error="GlobalAlloc failed")

    locked = kernel32.GlobalLock(handle)
    if not locked:
        kernel32.GlobalFree(handle)
        return _ClipboardCopyResult(copied=False, error="GlobalLock failed")

    ctypes.memmove(locked, data, len(data))
    kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        kernel32.GlobalFree(handle)
        return _ClipboardCopyResult(copied=False, error="OpenClipboard failed")

    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(_CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            return _ClipboardCopyResult(copied=False, error="SetClipboardData failed")
    finally:
        user32.CloseClipboard()
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
    _repo, _repo_alias, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)

    text = args.codex_output.read_text(encoding="utf-8")
    sys.stdout.write(render_review_findings(args.codex_output, task, text))
    sys.stdout.write("\n")
    return 0


def _run_review_codex(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph review-codex`."""
    repo, repo_alias, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)
    text = args.codex_output.read_text(encoding="utf-8")
    rendered = render_codex_reviewer_prompt(
        task=task,
        repo=repo,
        repo_alias=repo_alias,
        codex_output=text,
        source=str(args.codex_output.resolve()),
    )
    if args.output is not None:
        try:
            write_prompt_output(args.output, rendered.prompt, overwrite=args.overwrite)
        except HarnessUsageError as exc:
            parser.error(str(exc))
    if args.no_copy:
        sys.stdout.write(rendered.prompt)
        sys.stdout.write("\n")
        return 0

    copy_result = _copy_to_clipboard(rendered.prompt)
    if copy_result.copied:
        sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex reviewer prompt copied to clipboard"))
        sys.stdout.write("\n")
        return 0

    sys.stdout.write(_render_task_confirmation(repo, rendered.selected_context_count, "Codex reviewer prompt was not copied"))
    sys.stdout.write(f"\nClipboard unavailable: {copy_result.error or 'no clipboard backend available'}\n\n")
    sys.stdout.write(rendered.prompt)
    sys.stdout.write("\n")
    return 0


def _run_outcome(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph outcome`."""
    repo, repo_alias, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)
    text = args.codex_output.read_text(encoding="utf-8")
    report = render_supervised_outcome_report(
        task=task,
        repo=repo,
        repo_alias=repo_alias,
        codex_output=text,
        source=str(args.codex_output.resolve()),
    )
    sys.stdout.write(report.text)
    sys.stdout.write("\n")
    if args.save_history:
        if repo is None:
            parser.error("--save-history requires a resolved repo")
        path = append_outcome_history(repo, report)
        sys.stdout.write(f"Outcome history saved: {path}\n")
    return 0


def _run_outcome_history(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph outcome-history`."""
    repo = _resolve_history_repo(explicit_repo=args.repo, positional_repo=args.repo_or_alias, parser=parser)
    if args.limit < 1:
        parser.error("--limit must be greater than 0")
    sys.stdout.write(render_outcome_history(repo, limit=args.limit))
    sys.stdout.write("\n")
    return 0


def _run_outcome_learning(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph outcome-learning`."""
    repo, task = _resolve_history_repo_and_optional_task(
        explicit_repo=args.repo,
        positional=[item for item in [args.repo_or_task, *args.task] if item is not None],
        parser=parser,
    )
    if args.limit < 1:
        parser.error("--limit must be greater than 0")
    sys.stdout.write(render_outcome_learning_signals(repo, task=task, limit=args.limit))
    sys.stdout.write("\n")
    return 0


def _run_plan(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the deprecated `ph plan` compatibility view."""
    repo, repo_alias, task = _resolve_repo_and_task(explicit_repo=args.repo, positional=[args.repo_or_task, *args.task], parser=parser)
    rendered = render_controlled_execution_plan(task=task, repo=repo, repo_alias=repo_alias, include_history=args.with_history)
    candidate_path: Path | None = None
    if args.candidate_json is not None:
        try:
            _write_json_output(args.candidate_json, rendered.automation_candidate, overwrite=args.overwrite)
        except HarnessUsageError as exc:
            parser.error(str(exc))
        candidate_path = args.candidate_json.resolve()
    sys.stdout.write(rendered.text)
    sys.stdout.write("\n")
    if candidate_path is not None:
        sys.stdout.write("\n")
        sys.stdout.write(_render_candidate_execution_guide(candidate_path, repo=repo, task=task, candidate=rendered.automation_candidate))
        sys.stdout.write("\n")
    return 0


def _run_candidate_dry_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph candidate-dry-run`."""
    try:
        payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    except OSError as exc:
        parser.error(f"unable to read candidate JSON: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid candidate JSON at {args.candidate_json}: {exc.msg}")
    rendered = render_automation_candidate_dry_run(payload, source=str(args.candidate_json.resolve()))
    sys.stdout.write(rendered.text)
    sys.stdout.write("\n")
    return 0 if rendered.valid else 1


def _run_candidate_outcome(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph candidate-outcome`."""
    try:
        payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    except OSError as exc:
        parser.error(f"unable to read candidate JSON: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid candidate JSON at {args.candidate_json}: {exc.msg}")
    try:
        codex_output = args.codex_output.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(f"unable to read Codex output: {exc}")
    try:
        report = render_candidate_outcome_report(
            candidate=payload,
            candidate_source=str(args.candidate_json.resolve()),
            codex_output=codex_output,
            codex_output_source=str(args.codex_output.resolve()),
        )
    except HarnessUsageError as exc:
        parser.error(str(exc))
    sys.stdout.write(report.text)
    sys.stdout.write("\n")
    if args.save_history:
        if args.repo is None:
            parser.error("--save-history requires --repo for candidate-outcome")
        repo = _resolve_positional_repo(args.repo, parser)
        path = append_outcome_history(repo, report)
        sys.stdout.write(f"Outcome history saved: {path}\n")
    return 0


def _render_candidate_execution_guide(candidate_path: Path, *, repo: Path | None, task: str, candidate: dict[str, object]) -> str:
    """Render copy-pasteable PowerShell commands for first candidate execution."""
    approval_sha = automation_candidate_approval_sha(candidate)
    output = candidate_path.with_suffix(".codex-output.txt")
    repo_text = str(repo.resolve()) if repo is not None else "<repo>"
    generate = _powershell_command(
        (
            "ph",
            "plan",
            "--repo",
            repo_text,
            "--candidate-json",
            str(candidate_path),
            task,
        )
    )
    dry_run = _powershell_command(("ph", "candidate-dry-run", str(candidate_path)))
    execute = _powershell_command(
        (
            "ph",
            "candidate-execute",
            "--approve-sha",
            approval_sha,
            "--output",
            str(output),
            "--repo",
            repo_text,
            str(candidate_path),
        )
    )
    review = _powershell_command(
        (
            "ph",
            "candidate-outcome",
            str(candidate_path),
            "--codex-output",
            str(output),
            "--repo",
            repo_text,
            "--save-history",
        )
    )
    return f"""Candidate Execution Guide (PowerShell)
Candidate JSON: {candidate_path}
Raw Codex output file: {output}
Approval SHA: {approval_sha}

1. Generate candidate:
{generate}

2. Dry-run candidate:
{dry_run}

3. Copy approval SHA:
{approval_sha}

4. Execute candidate once:
{execute}

5. Review outcome and save history:
{review}

Safety:
- Commands above do not run automatically from this guide.
- candidate-execute still requires the exact --approve-sha.
- candidate-execute runs one foreground Codex command only.
- Raw Codex output is persisted only because --output is present.
- History is saved only by candidate-outcome --save-history."""


def _run_candidate_execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph candidate-execute`."""
    try:
        payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    except OSError as exc:
        parser.error(f"unable to read candidate JSON: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid candidate JSON at {args.candidate_json}: {exc.msg}")

    source = str(args.candidate_json.resolve())
    dry_run = render_automation_candidate_dry_run(payload, source=source)
    if not dry_run.valid:
        sys.stdout.write(dry_run.text)
        sys.stdout.write("\nCandidate execution refused: invalid candidate.\n")
        return 1

    expected_sha = automation_candidate_approval_sha(payload)
    classification = automation_candidate_readiness_classification(payload)
    if classification != "automation_ready":
        sys.stdout.write(_render_candidate_execute_refusal(expected_sha, f"candidate readiness is {classification}; expected automation_ready"))
        return 1
    if args.approve_sha != expected_sha:
        reason = "approval SHA is missing" if args.approve_sha is None else "approval SHA does not match"
        sys.stdout.write(_render_candidate_execute_refusal(expected_sha, reason))
        return 1

    prompt = automation_candidate_codex_prompt(payload)
    if not prompt:
        parser.error("candidate proposed_codex_prompt is missing")
    command = _parse_codex_command(args.codex_cmd, parser)
    try:
        result = _run_codex_once(command, prompt)
    except OSError as exc:
        parser.error(f"Codex command failed to start: {exc}")

    if args.output is not None:
        try:
            _write_text_output(args.output, result.output, overwrite=args.overwrite)
        except HarnessUsageError as exc:
            parser.error(str(exc))
    sys.stdout.write("\nNext review command:\n")
    sys.stdout.write(_candidate_outcome_next_command(args.candidate_json, output=args.output, repo=args.repo))
    sys.stdout.write("\n")
    return result.returncode


@dataclass(frozen=True)
class _CodexExecutionResult:
    """Foreground Codex command result."""

    returncode: int
    output: str


def _run_codex_once(command: tuple[str, ...], prompt: str) -> _CodexExecutionResult:
    """Invoke one foreground Codex command with the prompt on stdin."""
    process = subprocess.Popen(  # noqa: S603  # Operator-supplied command is the explicit execution boundary for this pilot.
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdin is None or process.stdout is None:
        return _CodexExecutionResult(returncode=1, output="")
    process.stdin.write(prompt)
    process.stdin.close()

    chunks: list[str] = []
    for chunk in process.stdout:
        sys.stdout.write(chunk)
        chunks.append(chunk)
    return _CodexExecutionResult(returncode=process.wait(), output="".join(chunks))


def _parse_codex_command(command: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """Parse a configured Codex command without shell expansion."""
    try:
        parts = tuple(shlex.split(command, posix=sys.platform != "win32"))
    except ValueError as exc:
        parser.error(f"invalid --codex-cmd: {exc}")
    if not parts:
        parser.error("--codex-cmd must not be empty")
    return parts


def _render_candidate_execute_refusal(expected_sha: str, reason: str) -> str:
    """Render an execution refusal with the approval token."""
    return f"""Candidate execution refused: {reason}.
Candidate approval SHA: {expected_sha}
Re-run with: --approve-sha {expected_sha}
"""


def _candidate_outcome_next_command(candidate: Path, *, output: Path | None, repo: str | None) -> str:
    """Render the next explicit candidate review command."""
    output_text = str(output.resolve()) if output is not None else "<codex-output>"
    repo_text = repo if repo is not None else "<repo>"
    return f"ph candidate-outcome {candidate.resolve()} --codex-output {output_text} --repo {repo_text} --save-history"


def _powershell_command(parts: tuple[str, ...]) -> str:
    """Render a command line suitable for Windows PowerShell copy/paste."""
    return " ".join(_powershell_quote(part) for part in parts)


def _powershell_quote(value: str) -> str:
    """Quote a single PowerShell argument when needed."""
    if value and not any(char.isspace() for char in value) and not any(char in value for char in "'\"&|<>()@;"):
        return value
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _write_text_output(path: Path, text: str, *, overwrite: bool) -> None:
    """Write explicit raw output with overwrite protection."""
    target = path.resolve()
    if target.exists() and not overwrite:
        msg = f"output file already exists: {target}; pass --overwrite to replace it"
        raise HarnessUsageError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _write_json_output(path: Path, payload: dict[str, object], *, overwrite: bool) -> None:
    """Write a deterministic JSON payload with explicit overwrite protection."""
    target = path.resolve()
    if target.exists() and not overwrite:
        msg = f"output file already exists: {target}; pass --overwrite to replace it"
        raise HarnessUsageError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


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


def _run_ecc_status() -> int:
    """Run `ph ecc-status`."""
    sys.stdout.write(render_ecc_status())
    sys.stdout.write("\n")
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


def _run_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph benchmark`."""
    resolved = _resolve_positional_repo_with_alias(args.repo, parser) if args.repo is not None else None
    repo = resolved.path if resolved is not None else None
    repo_alias = resolved.alias if resolved is not None else None
    try:
        results = run_prompt_benchmarks(benchmarks_dir=args.benchmarks, repo=repo, repo_alias=repo_alias)
    except HarnessUsageError as exc:
        parser.error(str(exc))
    sys.stdout.write(render_prompt_benchmark_report(results))
    sys.stdout.write("\n")
    return 0 if all(result.passed for result in results) else 1


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912  # explicit argparse dispatch keeps command behavior readable
    """Run the `ph` CLI."""
    _configure_utf8_stdio()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "task":
        result = _run_task(args, parser)
    elif args.command == "review":
        result = _run_review(args, parser)
    elif args.command == "review-codex":
        result = _run_review_codex(args, parser)
    elif args.command == "outcome":
        result = _run_outcome(args, parser)
    elif args.command == "outcome-history":
        result = _run_outcome_history(args, parser)
    elif args.command == "outcome-learning":
        result = _run_outcome_learning(args, parser)
    elif args.command == "plan":
        result = _run_plan(args, parser)
    elif args.command == "candidate-dry-run":
        result = _run_candidate_dry_run(args, parser)
    elif args.command == "candidate-outcome":
        result = _run_candidate_outcome(args, parser)
    elif args.command == "candidate-execute":
        result = _run_candidate_execute(args, parser)
    elif args.command == "run":
        result = _run_interactive(args, parser)
    elif args.command == "status":
        result = _run_status()
    elif args.command == "ecc-status":
        result = _run_ecc_status()
    elif args.command == "repos":
        result = _run_repos(args, parser)
    elif args.command == "benchmark":
        result = _run_benchmark(args, parser)
    else:
        parser.error(f"unknown command: {args.command}")
    return result


def _configure_utf8_stdio() -> None:
    """Prefer UTF-8 for interactive text streams when Python supports it."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
