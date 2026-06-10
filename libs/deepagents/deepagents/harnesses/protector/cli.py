"""Command-line entry point for the Protector engineering harness."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

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
    from collections.abc import Callable, Sequence

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
_CODEX_EXECUTION_HEARTBEAT_SECONDS = 10.0
_CODEX_PROCESS_POLL_SECONDS = 0.1
_CODEX_READER_JOIN_SECONDS = 1.0
_CLI_SPINNER_INTERVAL_SECONDS = 0.1
_CLI_SPINNER_FRAMES = ("|", "/", "-", "\\")
_CODEX_TERMINATION_WAIT_SECONDS = 5.0
_CODEX_WINDOWS_SANDBOX_HELPER = "codex-windows-sandbox-setup.exe"
_CODEX_DIAGNOSTIC_TAIL_CHARS = 2000
_CODEX_TERMINAL_EXECUTOR_FAILURE_PATTERNS = (
    "LOCAL_EXECUTOR_UNAVAILABLE",
    f"{_CODEX_WINDOWS_SANDBOX_HELPER} program not found",
    "executor failure",
    "local executor unavailable",
)
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
    "executor-status",
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


@dataclass(frozen=True)
class _ExecutorReliabilityCheck:
    """One local executor reliability check."""

    name: str
    passed: bool
    detail: str
    suggested_fix: str | None = None


@dataclass(frozen=True)
class _ExecutorReliabilityReport:
    """Local workspace executor reliability status."""

    executor: str
    repo: Path
    executable_path: str
    version: str
    codex_home: str
    checks: tuple[_ExecutorReliabilityCheck, ...]
    sandbox_mode: str = "(unknown)"
    execution_strategy: str = "(unknown)"
    command_line: str = "(unknown)"
    npm_package_location: str = "(unknown)"
    sandbox_helper_path: str = "(unknown)"
    runtime_path_prefix: str = "(unknown)"

    @property
    def ok(self) -> bool:
        """Return whether every reliability check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failing_checks(self) -> tuple[_ExecutorReliabilityCheck, ...]:
        """Return checks that block local candidate execution."""
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True)
class _CodexRuntimeEnvironment:
    """Resolved Codex process environment shared by status and execution."""

    executable_path: Path
    env: dict[str, str]
    npm_package_location: str
    sandbox_helper_path: str
    runtime_path_prefix: str
    sandbox_helper_check: _ExecutorReliabilityCheck


class _LocalWorkspaceExecutor(Protocol):
    """Adapter boundary for local candidate execution providers."""

    name: str
    repo: Path

    def check_reliability(self, progress: Callable[[str], None] | None = None) -> _ExecutorReliabilityReport:
        """Verify the executor can read and mutate the local workspace."""

    def execute(self, prompt: str, *, timeout: float | None) -> _CodexExecutionResult:
        """Run one approved candidate prompt through the local executor."""


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


def _resolve_candidate_execution_repo(repo: str | None, parser: argparse.ArgumentParser) -> Path:
    """Resolve the workspace where local candidate execution must succeed."""
    if repo is not None:
        return _resolve_positional_repo(repo, parser)
    return Path.cwd().resolve()


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
    plan.add_argument("--refine-task", action="store_true", help="Normalize rough operator task text before candidate generation.")
    plan.add_argument("--evidence", type=Path, action="append", default=[], help="Evidence file path reference to attach; repeat for multiple files.")
    plan.add_argument(
        "--evidence-note",
        action="append",
        default=[],
        help="Human-readable evidence observation to attach; repeat for multiple notes.",
    )
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
        "--executor",
        choices=("codex_cli",),
        default="codex_cli",
        help="Local workspace executor adapter to use. Default: codex_cli.",
    )
    candidate_execute.add_argument(
        "--codex-cmd",
        default=_DEFAULT_CODEX_COMMAND,
        help=f"Codex command to invoke once. Default: {_DEFAULT_CODEX_COMMAND!r}.",
    )
    candidate_execute.add_argument("--output", type=Path, default=None, help="Optional file path for persisting raw Codex output.")
    candidate_execute.add_argument("--overwrite", action="store_true", help="Allow --output to replace an existing file.")
    candidate_execute.add_argument("--repo", default=None, help="Optional repository path or alias shown in the next candidate-outcome command.")
    candidate_execute.add_argument("--no-spinner", action="store_true", help="Disable TTY progress spinner during executor preflight.")
    candidate_execute.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for the foreground Codex execution.",
    )
    candidate_execute.add_argument("candidate_json", type=Path, help="Automation-ready candidate JSON to execute once.")

    executor_status = subparsers.add_parser(
        "executor-status",
        help="Check whether the local candidate executor can read and mutate a workspace.",
    )
    executor_status.add_argument("--repo", required=True, help="Repository path or alias to verify.")
    executor_status.add_argument(
        "--executor",
        choices=("codex_cli",),
        default="codex_cli",
        help="Local workspace executor adapter to check. Default: codex_cli.",
    )
    executor_status.add_argument(
        "--codex-cmd",
        default=_DEFAULT_CODEX_COMMAND,
        help=f"Codex command used by the codex_cli adapter. Default: {_DEFAULT_CODEX_COMMAND!r}.",
    )
    executor_status.add_argument("--no-spinner", action="store_true", help="Disable TTY progress spinner during executor checks.")

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
    rendered = render_controlled_execution_plan(
        task=task,
        repo=repo,
        repo_alias=repo_alias,
        include_history=args.with_history,
        refine_task=args.refine_task,
        evidence_paths=tuple(str(path) for path in args.evidence),
        evidence_notes=tuple(args.evidence_note),
    )
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
    sys.stdout.write(_render_candidate_dry_run_approval(payload, candidate_path=args.candidate_json.resolve(), decision=rendered.decision))
    sys.stdout.write("\n")
    return 0 if rendered.valid else 1


def _render_candidate_dry_run_approval(candidate: object, *, candidate_path: Path, decision: str) -> str:
    """Render approval SHA and the next execution command when eligible."""
    approval_sha = automation_candidate_approval_sha(candidate)
    if decision == "would execute":
        execute = _powershell_command(("ph", "candidate-execute", "--approve-sha", approval_sha, str(candidate_path)))
        return f"""Approval
- Candidate approval SHA: {approval_sha}
- Execution decision: would execute

Next candidate-execute command:
{execute}
"""
    return f"""Approval
- Candidate approval SHA: {approval_sha}
- Execution decision: {decision}
- Execution blocked/not recommended; no candidate-execute command is emitted for this candidate.
"""


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
    task_payload = candidate.get("task")
    intake = task_payload.get("intake_refinement") if isinstance(task_payload, dict) else None
    refine_args = ("--refine-task",) if isinstance(intake, dict) and intake.get("enabled") is True else ()
    evidence = task_payload.get("evidence_paths") if isinstance(task_payload, dict) else ()
    evidence_args: list[str] = []
    if isinstance(evidence, (list, tuple)):
        for path in evidence:
            if isinstance(path, str):
                evidence_args.extend(("--evidence", path))
    evidence_notes = task_payload.get("evidence_notes") if isinstance(task_payload, dict) else ()
    if isinstance(evidence_notes, (list, tuple)):
        for note in evidence_notes:
            if isinstance(note, str):
                evidence_args.extend(("--evidence-note", note))
    generate = _powershell_command(
        (
            "ph",
            "plan",
            "--repo",
            repo_text,
            *refine_args,
            *evidence_args,
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
    timeout = _candidate_execute_timeout(args.timeout, parser)

    prompt = automation_candidate_codex_prompt(payload)
    if not prompt:
        parser.error("candidate proposed_codex_prompt is missing")
    command = _parse_codex_command(args.codex_cmd, parser)
    repo = _resolve_candidate_execution_repo(args.repo, parser)
    executor = _build_local_executor(args.executor, command=command, repo=repo, parser=parser)
    reliability = _check_executor_reliability_with_spinner(executor, no_spinner=args.no_spinner)
    if not reliability.ok:
        sys.stdout.write(_render_executor_reliability_report(reliability, candidate_blocked=True))
        return 1
    try:
        result = executor.execute(prompt, timeout=timeout)
    except OSError as exc:
        parser.error(f"Codex command failed to start: {exc}")

    _write_candidate_execution_output(args, parser, result)
    if result.termination_report is not None or result.failed:
        sys.stdout.write(_render_candidate_execute_termination(args.candidate_json, output=args.output, repo=args.repo, result=result))
        return _candidate_execute_termination_returncode(result)
    sys.stdout.write("\nNext review command:\n")
    sys.stdout.write(_candidate_outcome_next_command(args.candidate_json, output=args.output, repo=args.repo))
    sys.stdout.write("\n")
    return result.returncode


def _candidate_execute_termination_returncode(result: _CodexExecutionResult) -> int:
    """Return the CLI status for a controlled candidate termination."""
    if result.timed_out:
        return 124
    if result.returncode != 0:
        return result.returncode
    if result.failed:
        return 1
    return 130


def _candidate_execute_timeout(timeout: float | None, parser: argparse.ArgumentParser) -> float | None:
    """Validate the optional candidate execution timeout."""
    if timeout is not None and timeout <= 0:
        parser.error("--timeout must be greater than 0")
    return timeout


def _write_candidate_execution_output(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    result: _CodexExecutionResult,
) -> None:
    """Persist raw candidate execution output only when explicitly requested."""
    if args.output is None:
        return
    try:
        _write_text_output(args.output, result.output, overwrite=args.overwrite)
    except HarnessUsageError as exc:
        parser.error(str(exc))


@dataclass(frozen=True)
class _CodexCliExecutor:
    """Codex CLI local workspace executor adapter."""

    command: tuple[str, ...]
    repo: Path
    name: str = "codex_cli"

    def check_reliability(self, progress: Callable[[str], None] | None = None) -> _ExecutorReliabilityReport:
        """Verify the Codex CLI can execute local workspace commands."""
        checks: list[_ExecutorReliabilityCheck] = []
        _notify_progress(progress, "resolving executable")
        executable = _resolve_executor_executable(self.command)
        if executable is None:
            command_line = _powershell_command(self.command)
            return _ExecutorReliabilityReport(
                executor=self.name,
                repo=self.repo,
                executable_path="(not found)",
                version="(unknown)",
                codex_home=_format_codex_home(_codex_home_path()),
                checks=(
                    _ExecutorReliabilityCheck(
                        name="codex_cli_available",
                        passed=False,
                        detail=f"Unable to resolve executable from command: {self.command[0]}",
                        suggested_fix="Install Codex CLI or pass --codex-cmd with the intended Codex executable on PATH.",
                    ),
                ),
                sandbox_mode=_codex_sandbox_mode(self.command),
                execution_strategy=_codex_execution_strategy(self.repo),
                command_line=command_line,
            )

        runtime = _resolve_codex_runtime_environment(executable)
        checks.append(
            _ExecutorReliabilityCheck(
                name="codex_cli_available",
                passed=True,
                detail=str(executable),
            )
        )
        checks.append(runtime.sandbox_helper_check)
        _notify_progress(progress, "checking version")
        version, version_check = _check_executor_version(executable, env=runtime.env)
        checks.append(version_check)
        _notify_progress(progress, "checking CODEX_HOME")
        codex_home = _codex_home_path()
        checks.append(_check_codex_home(codex_home))

        if all(check.passed for check in checks):
            checks.extend(_run_deterministic_local_workspace_checks(self.repo, progress=progress))
            checks.append(_codex_model_execution_available_check())

        return _ExecutorReliabilityReport(
            executor=self.name,
            repo=self.repo,
            executable_path=str(executable),
            version=version,
            codex_home=_format_codex_home(codex_home),
            checks=tuple(checks),
            sandbox_mode=_codex_sandbox_mode(self.command),
            execution_strategy=_codex_execution_strategy(self.repo),
            command_line=_powershell_command(self.command),
            npm_package_location=runtime.npm_package_location,
            sandbox_helper_path=runtime.sandbox_helper_path,
            runtime_path_prefix=runtime.runtime_path_prefix,
        )

    def execute(self, prompt: str, *, timeout: float | None) -> _CodexExecutionResult:
        """Run one candidate prompt with connector fallback explicitly disallowed."""
        runtime = _resolve_codex_runtime_environment(self._resolved_executable())
        return _run_codex_once(
            self.command,
            _local_executor_candidate_prompt(prompt),
            timeout=timeout,
            cwd=self.repo,
            env=runtime.env,
        )

    def _resolved_executable(self) -> Path:
        """Return the resolved Codex executable for execution."""
        executable = _resolve_executor_executable(self.command)
        if executable is None:
            msg = f"unable to resolve executable from command: {self.command[0]}"
            raise OSError(msg)
        return executable


def _build_local_executor(
    executor: str,
    *,
    command: tuple[str, ...],
    repo: Path,
    parser: argparse.ArgumentParser,
) -> _LocalWorkspaceExecutor:
    """Build the requested local workspace executor adapter."""
    if executor == "codex_cli":
        return _CodexCliExecutor(command=command, repo=repo)
    parser.error(f"unknown executor: {executor}")
    msg = "unreachable"
    raise AssertionError(msg)


def _run_executor_status(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run `ph executor-status`."""
    command = _parse_codex_command(args.codex_cmd, parser)
    repo = _resolve_candidate_execution_repo(args.repo, parser)
    executor = _build_local_executor(args.executor, command=command, repo=repo, parser=parser)
    report = _check_executor_reliability_with_spinner(executor, no_spinner=args.no_spinner)
    sys.stdout.write(_render_executor_reliability_report(report, candidate_blocked=False))
    return 0 if report.ok else 1


def _check_executor_reliability_with_spinner(
    executor: _LocalWorkspaceExecutor,
    *,
    no_spinner: bool,
) -> _ExecutorReliabilityReport:
    """Run executor reliability checks with optional TTY progress."""
    with _CliSpinner(enabled=_cli_spinner_enabled(no_spinner=no_spinner), phase="starting executor checks") as spinner:
        return executor.check_reliability(progress=spinner.update)


def _cli_spinner_enabled(*, no_spinner: bool) -> bool:
    """Return whether CLI progress should be visible for this output stream."""
    if no_spinner:
        return False
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty is not None and isatty())


def _notify_progress(progress: Callable[[str], None] | None, phase: str) -> None:
    """Notify a progress sink when one is active."""
    if progress is not None:
        progress(phase)


def _codex_sandbox_mode(command: tuple[str, ...]) -> str:
    """Return the effective sandbox mode declared by the Codex command."""
    for index, part in enumerate(command):
        if part == "--sandbox" and index + 1 < len(command):
            return command[index + 1]
        if part.startswith("--sandbox="):
            return part.split("=", maxsplit=1)[1]
    return "codex default"


def _codex_execution_strategy(repo: Path) -> str:
    """Return a compact description of the local execution strategy."""
    return f"foreground codex_cli process; cwd={repo}; local workspace only; connector fallback disabled"


class _CliSpinner:
    """Lightweight stderr spinner for long foreground checks."""

    def __init__(self, *, enabled: bool, phase: str) -> None:
        self._enabled = enabled
        self._phase = phase
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_width = 0

    def __enter__(self) -> Self:
        if self._enabled:
            self._thread = threading.Thread(target=self._run, name="protector-cli-spinner", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        _ = exc_type, exc, traceback
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._clear()

    def update(self, phase: str) -> None:
        """Update the visible spinner phase."""
        if not self._enabled:
            return
        with self._lock:
            self._phase = phase

    def _run(self) -> None:
        """Render spinner frames until stopped."""
        index = 0
        while not self._stop.wait(_CLI_SPINNER_INTERVAL_SECONDS):
            with self._lock:
                phase = self._phase
            frame = _CLI_SPINNER_FRAMES[index % len(_CLI_SPINNER_FRAMES)]
            index += 1
            self._write(f"\r{frame} {phase}")

    def _write(self, text: str) -> None:
        """Write one spinner frame to stderr."""
        self._last_width = max(self._last_width, len(text))
        sys.stderr.write(text.ljust(self._last_width))
        sys.stderr.flush()

    def _clear(self) -> None:
        """Clear the spinner line."""
        if self._last_width:
            sys.stderr.write("\r" + (" " * self._last_width) + "\r")
            sys.stderr.flush()


def _resolve_executor_executable(command: tuple[str, ...]) -> Path | None:
    """Resolve the active executable path for an executor command."""
    executable = Path(command[0]).expanduser()
    if executable.is_file():
        return executable.resolve()
    resolved = shutil.which(command[0])
    if resolved is None:
        return None
    return Path(resolved).resolve()


def _resolve_codex_runtime_environment(executable: Path) -> _CodexRuntimeEnvironment:
    """Resolve the effective Codex process environment used by status and execution."""
    npm_package = _resolve_codex_npm_package_location(executable)
    helper = _resolve_codex_sandbox_helper(executable, npm_package=npm_package)
    env = dict(os.environ)
    path_prefix = "(none)"
    if helper is not None:
        path_prefix = str(helper.parent)
        env["PATH"] = f"{path_prefix}{os.pathsep}{env.get('PATH', '')}"
    helper_check = _codex_sandbox_helper_check(helper)
    return _CodexRuntimeEnvironment(
        executable_path=executable,
        env=env,
        npm_package_location=str(npm_package) if npm_package is not None else "(not found)",
        sandbox_helper_path=str(helper) if helper is not None else "(not found)",
        runtime_path_prefix=path_prefix,
        sandbox_helper_check=helper_check,
    )


def _resolve_codex_npm_package_location(executable: Path) -> Path | None:
    """Resolve the npm `@openai/codex` package used by an npm shim."""
    candidates = (
        executable.parent / "node_modules" / "@openai" / "codex",
        executable.parent.parent / "node_modules" / "@openai" / "codex",
    )
    for candidate in candidates:
        if (candidate / "package.json").is_file():
            return candidate.resolve()
    return None


def _resolve_codex_sandbox_helper(executable: Path, *, npm_package: Path | None) -> Path | None:
    """Resolve the Windows sandbox helper for the active Codex installation."""
    if sys.platform != "win32":
        return None
    candidates: list[Path] = []
    if npm_package is not None:
        vendor = Path("vendor") / "x86_64-pc-windows-msvc" / "codex-resources" / _CODEX_WINDOWS_SANDBOX_HELPER
        candidates.extend(
            (
                npm_package.parent / "codex-win32-x64" / vendor,
                npm_package / "node_modules" / "@openai" / "codex-win32-x64" / vendor,
            )
        )
    candidates.extend(
        (
            executable.parent / "codex-resources" / _CODEX_WINDOWS_SANDBOX_HELPER,
            executable.parent.parent / "codex-resources" / _CODEX_WINDOWS_SANDBOX_HELPER,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    path_match = shutil.which(_CODEX_WINDOWS_SANDBOX_HELPER)
    if path_match is not None and Path(path_match).is_file():
        return Path(path_match).resolve()
    return None


def _codex_sandbox_helper_check(helper: Path | None) -> _ExecutorReliabilityCheck:
    """Return a preflight check for the Codex Windows sandbox helper."""
    if sys.platform != "win32":
        return _ExecutorReliabilityCheck(
            name="codex_sandbox_helper_available",
            passed=True,
            detail="Windows sandbox helper is not required on this platform.",
        )
    if helper is not None:
        return _ExecutorReliabilityCheck(
            name="codex_sandbox_helper_available",
            passed=True,
            detail=str(helper),
        )
    return _ExecutorReliabilityCheck(
        name="codex_sandbox_helper_available",
        passed=False,
        detail=f"{_CODEX_WINDOWS_SANDBOX_HELPER} could not be resolved for the active Codex executable.",
        suggested_fix="Use the Codex install whose platform package includes codex-resources, or reinstall/update the active Codex CLI.",
    )


def _check_executor_version(executable: Path, *, env: dict[str, str]) -> tuple[str, _ExecutorReliabilityCheck]:
    """Run the executor version command."""
    try:
        completed = subprocess.run(  # noqa: S603  # The resolved executor path is the inspected local execution boundary.
            (str(executable), "--version"),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001  # Status diagnostics must not crash when tooling is broken.
        detail = str(exc) or type(exc).__name__
        return (
            "(unknown)",
            _ExecutorReliabilityCheck(
                name="version",
                passed=False,
                detail=detail,
                suggested_fix="Repair or reinstall the active Codex CLI executable so it can report its version.",
            ),
        )

    version = (completed.stdout or completed.stderr).strip() or "(empty version output)"
    if completed.returncode != 0:
        return (
            version,
            _ExecutorReliabilityCheck(
                name="version",
                passed=False,
                detail=f"Version command exited {completed.returncode}: {version}",
                suggested_fix="Repair or reinstall the active Codex CLI executable before running local candidates.",
            ),
        )
    return version, _ExecutorReliabilityCheck(name="version", passed=True, detail=version)


def _codex_home_path() -> Path:
    """Return the active Codex home path from environment or the default."""
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def _format_codex_home(path: Path) -> str:
    """Render CODEX_HOME with enough context for diagnostics."""
    configured = os.environ.get("CODEX_HOME")
    source = "env" if configured else "default"
    return f"{path} ({source})"


def _check_codex_home(path: Path) -> _ExecutorReliabilityCheck:
    """Verify the active Codex home path is usable."""
    if path.exists() and path.is_dir():
        return _ExecutorReliabilityCheck(name="codex_home", passed=True, detail=str(path))
    return _ExecutorReliabilityCheck(
        name="codex_home",
        passed=False,
        detail=f"Codex home does not exist or is not a directory: {path}",
        suggested_fix="Set CODEX_HOME to an existing Codex home with auth/config, or unset CODEX_HOME to use the default.",
    )


def _run_deterministic_local_workspace_checks(
    repo: Path,
    *,
    progress: Callable[[str], None] | None,
) -> tuple[_ExecutorReliabilityCheck, ...]:
    """Verify local workspace read/write/delete without asking a model to run commands."""
    _notify_progress(progress, "checking workspace command")
    command = _ExecutorReliabilityCheck(
        name="ph_local_workspace_check",
        passed=True,
        detail=f"ph direct local filesystem check running in {repo}",
    )
    _notify_progress(progress, "checking workspace read")
    read = _local_workspace_read_check(repo)
    write_delete = _ExecutorReliabilityCheck(
        name="workspace_write_delete",
        passed=False,
        detail="Skipped because workspace read failed.",
        suggested_fix="Fix local repository read access before checking write/delete.",
    )
    if read.passed:
        _notify_progress(progress, "checking workspace write/delete")
        write_delete = _local_workspace_write_delete_check(repo)
    return (command, read, write_delete)


def _local_workspace_read_check(repo: Path) -> _ExecutorReliabilityCheck:
    """Verify the target repo can be read directly by `ph`."""
    target = repo / "AGENTS.md"
    try:
        target.read_text(encoding="utf-8")
    except OSError as exc:
        return _ExecutorReliabilityCheck(
            name="workspace_read",
            passed=False,
            detail=f"ph could not read {target}: {exc}",
            suggested_fix="Fix local repo path or filesystem permissions before candidate execution.",
        )
    return _ExecutorReliabilityCheck(
        name="workspace_read",
        passed=True,
        detail=f"ph read {target}",
    )


def _local_workspace_write_delete_check(repo: Path) -> _ExecutorReliabilityCheck:
    """Verify the target repo can be written and cleaned up directly by `ph`."""
    proof = repo / ".protector-harness" / "executor-preflight.tmp"
    token = f"protector-executor-preflight-{os.getpid()}-{time.monotonic_ns()}"
    try:
        proof.parent.mkdir(parents=True, exist_ok=True)
        proof.write_text(token, encoding="utf-8")
        actual = proof.read_text(encoding="utf-8")
        if actual != token:
            return _ExecutorReliabilityCheck(
                name="workspace_write_delete",
                passed=False,
                detail=f"ph read back unexpected content from {proof}",
                suggested_fix="Fix local filesystem consistency before candidate execution.",
            )
        proof.unlink()
        if proof.exists():
            return _ExecutorReliabilityCheck(
                name="workspace_write_delete",
                passed=False,
                detail=f"ph deleted {proof}, but it still exists",
                suggested_fix="Fix local filesystem delete behavior before candidate execution.",
            )
    except OSError as exc:
        with contextlib.suppress(OSError):
            if proof.exists():
                proof.unlink()
        return _ExecutorReliabilityCheck(
            name="workspace_write_delete",
            passed=False,
            detail=f"ph could not write/read/delete {proof}: {exc}",
            suggested_fix="Fix local repo write/delete permissions before candidate execution.",
        )
    return _ExecutorReliabilityCheck(
        name="workspace_write_delete",
        passed=True,
        detail=f"ph wrote, read, and deleted {proof}",
    )


def _codex_model_execution_available_check() -> _ExecutorReliabilityCheck:
    """Report that model execution is intentionally not part of reliability gating."""
    return _ExecutorReliabilityCheck(
        name="codex_model_execution_available",
        passed=True,
        detail="not checked by deterministic preflight; candidate-execute is the first model execution",
    )


def _local_executor_candidate_prompt(prompt: str) -> str:
    """Add local-executor policy without changing the approved task body."""
    return f"""{prompt}

Local executor policy:
- Use only local workspace shell/file execution for repository inspection and edits.
- Connector-only or remote repository fallback is explicitly disallowed for candidate-execute.
- If local workspace execution becomes unavailable, stop immediately and report LOCAL_EXECUTOR_UNAVAILABLE.
"""


def _render_executor_reliability_report(report: _ExecutorReliabilityReport, *, candidate_blocked: bool) -> str:
    """Render local executor reliability diagnostics."""
    status = "PASS" if report.ok else "FAIL"
    heading = "Candidate execution refused: local executor reliability preflight failed.\n" if candidate_blocked and not report.ok else ""
    failing = "\n".join(f"- {check.name}: {check.detail}" for check in report.failing_checks) or "- (none)"
    fixes = "\n".join(f"- {check.name}: {check.suggested_fix}" for check in report.failing_checks if check.suggested_fix is not None)
    if not fixes:
        fixes = "- (none)"
    checks = "\n".join(f"- {check.name}: {'PASS' if check.passed else 'FAIL'} - {check.detail}" for check in report.checks)
    return f"""{heading}Local Executor Status: {status}
executor: {report.executor}
repo: {report.repo}
active_executable_path: {report.executable_path}
effective_command: {report.command_line}
sandbox_mode: {report.sandbox_mode}
execution_strategy: {report.execution_strategy}
npm_package_location: {report.npm_package_location}
sandbox_helper_path: {report.sandbox_helper_path}
runtime_PATH_prefix: {report.runtime_path_prefix}
version: {report.version}
CODEX_HOME: {report.codex_home}

Checks:
{checks}

Failing checks:
{failing}

Suggested fix:
{fixes}
"""


@dataclass(frozen=True)
class _CodexExecutionResult:
    """Foreground Codex command result."""

    returncode: int
    output: str
    interrupted: bool = False
    timed_out: bool = False
    failed: bool = False
    failure_message: str | None = None
    termination_report: _ProcessTerminationReport | None = None
    failure_kind: str | None = None
    root_exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True)
class _ProcessTerminationReport:
    """Process-tree termination diagnostics."""

    root_pid: int | None
    tracked_pids: tuple[int, ...]
    method: str
    terminated_pids: tuple[int, ...]
    resisted_pids: tuple[int, ...]
    diagnostics_error: str | None = None


@dataclass
class _OwnedCodexProcess:
    """Foreground process plus platform ownership handle."""

    process: subprocess.Popen[str]
    windows_job_handle: int | None = None
    ownership_error: str | None = None


def _run_codex_once(
    command: tuple[str, ...],
    prompt: str,
    *,
    timeout: float | None = None,
    heartbeat_interval: float = _CODEX_EXECUTION_HEARTBEAT_SECONDS,
    poll_interval: float = _CODEX_PROCESS_POLL_SECONDS,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    output_mode: str = "echo",
) -> _CodexExecutionResult:
    """Invoke one foreground Codex command with the prompt on stdin."""
    process = _launch_codex_process(command, cwd=cwd, env=env)
    if process.stdin is None or process.stdout is None:
        return _CodexExecutionResult(returncode=1, output="")
    owned = _own_codex_process(process)
    return _run_owned_codex_process(
        owned,
        prompt,
        timeout=timeout,
        heartbeat_interval=heartbeat_interval,
        poll_interval=poll_interval,
        output_mode=output_mode,
    )


def _launch_codex_process(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Launch the foreground Codex process."""
    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if env is not None:
        kwargs["env"] = env
    return subprocess.Popen(  # noqa: S603  # Operator-supplied command is the explicit execution boundary for this pilot.
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


def _run_owned_codex_process(
    owned: _OwnedCodexProcess,
    prompt: str,
    *,
    timeout: float | None,
    heartbeat_interval: float,
    poll_interval: float,
    output_mode: str,
) -> _CodexExecutionResult:
    """Run an owned Codex process while the main thread keeps cancellation control."""
    process = owned.process
    output = _ProcessOutputBuffer()
    errors: queue.SimpleQueue[str] = queue.SimpleQueue()
    terminal_failure = threading.Event()
    readers = _start_process_readers(
        process,
        output=output,
        errors=errors,
        output_mode=output_mode,
        terminal_failure=terminal_failure,
    )
    write_result = _write_prompt_or_terminate(owned, prompt=prompt, readers=tuple(readers), output=output)
    if write_result is not None:
        return write_result

    started = time.monotonic()
    next_heartbeat = started + heartbeat_interval
    try:
        while True:
            if terminal_failure.is_set():
                root_exit_code = _process_exit_code(process)
                report = _terminate_process_tree(owned, method="executor_failure")
                _join_readers(tuple(readers))
                return _CodexExecutionResult(
                    returncode=1,
                    output=output.text(),
                    failed=True,
                    failure_message=output.terminal_failure_reason() or "terminal executor failure",
                    termination_report=report,
                    failure_kind="harness_killed_process_tree_after_terminal_executor_failure",
                    root_exit_code=root_exit_code,
                    stdout_tail=output.stdout_tail(),
                    stderr_tail=output.stderr_tail(),
                )
            returncode = _wait_for_process(process, timeout=poll_interval)
            if returncode is not None:
                _join_readers(tuple(readers))
                _close_owned_process(owned)
                return _completed_codex_result(returncode, output=output, errors=errors)

            now = time.monotonic()
            if timeout is not None and now - started >= timeout:
                report = _terminate_process_tree(owned, method="timeout")
                _join_readers(tuple(readers))
                return _CodexExecutionResult(
                    returncode=124,
                    output=output.text(),
                    timed_out=True,
                    termination_report=report,
                    failure_kind="harness_timeout",
                    root_exit_code=_process_exit_code(process),
                    stdout_tail=output.stdout_tail(),
                    stderr_tail=output.stderr_tail(),
                )
            if heartbeat_interval > 0 and now >= next_heartbeat:
                _write_candidate_execute_heartbeat(process, elapsed_seconds=now - started)
                next_heartbeat = now + heartbeat_interval
    except KeyboardInterrupt:
        report = _terminate_process_tree(owned, method="keyboard_interrupt")
        _join_readers(tuple(readers))
        return _CodexExecutionResult(
            returncode=130,
            output=output.text(),
            interrupted=True,
            termination_report=report,
            failure_kind="harness_cancelled_by_operator",
            root_exit_code=_process_exit_code(process),
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )
    except Exception as exc:  # noqa: BLE001  # Failure cleanup must run before surfacing a deterministic result.
        root_exit_code = _process_exit_code(process)
        report = _terminate_process_tree(owned, method="failure")
        _join_readers(tuple(readers))
        return _CodexExecutionResult(
            returncode=1,
            output=output.text(),
            failed=True,
            failure_message=str(exc),
            termination_report=report,
            failure_kind="harness_killed_process_tree_after_executor_exception",
            root_exit_code=root_exit_code,
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )


def _write_prompt_or_terminate(
    owned: _OwnedCodexProcess,
    *,
    prompt: str,
    readers: tuple[threading.Thread, ...],
    output: _ProcessOutputBuffer,
) -> _CodexExecutionResult | None:
    """Write the prompt, terminating the process tree if stdin setup fails."""
    try:
        owned.process.stdin.write(prompt)
        owned.process.stdin.close()
    except KeyboardInterrupt:
        report = _terminate_process_tree(owned, method="keyboard_interrupt")
        _join_readers(readers)
        return _CodexExecutionResult(
            returncode=130,
            output=output.text(),
            interrupted=True,
            termination_report=report,
            failure_kind="harness_cancelled_by_operator",
            root_exit_code=_process_exit_code(owned.process),
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )
    except Exception as exc:  # noqa: BLE001  # Failure cleanup must run before surfacing a deterministic result.
        root_exit_code = _process_exit_code(owned.process)
        report = _terminate_process_tree(owned, method="failure")
        _join_readers(readers)
        return _CodexExecutionResult(
            returncode=1,
            output=output.text(),
            failed=True,
            failure_message=str(exc),
            termination_report=report,
            failure_kind="stdin_pipe_or_eof_issue",
            root_exit_code=root_exit_code,
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )
    return None


def _start_process_readers(
    process: subprocess.Popen[str],
    *,
    output: _ProcessOutputBuffer,
    errors: queue.SimpleQueue[str],
    output_mode: str,
    terminal_failure: threading.Event,
) -> tuple[threading.Thread, ...]:
    """Start stdout/stderr reader threads for a Codex process."""
    readers = [
        threading.Thread(
            target=_read_process_stream,
            args=(process.stdout, output, errors, output_mode, terminal_failure, "stdout"),
            name="candidate-execute-stdout",
            daemon=True,
        )
    ]
    stderr_stream = getattr(process, "stderr", None)
    if stderr_stream is not None:
        readers.append(
            threading.Thread(
                target=_read_process_stream,
                args=(stderr_stream, output, errors, output_mode, terminal_failure, "stderr"),
                name="candidate-execute-stderr",
                daemon=True,
            )
        )
    for reader in readers:
        reader.start()
    return tuple(readers)


def _completed_codex_result(
    returncode: int,
    *,
    output: _ProcessOutputBuffer,
    errors: queue.SimpleQueue[str],
) -> _CodexExecutionResult:
    """Build the result for a normally exited Codex process."""
    reader_error = _first_queue_item(errors)
    if reader_error is not None:
        return _CodexExecutionResult(
            returncode=1,
            output=output.text(),
            failed=True,
            failure_message=reader_error,
            failure_kind="stream_reader_error",
            root_exit_code=returncode,
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )
    failure_kind, failure_message = _completed_codex_failure(returncode, output.text())
    if failure_kind is not None:
        return _CodexExecutionResult(
            returncode=returncode,
            output=output.text(),
            failed=True,
            failure_message=failure_message,
            failure_kind=failure_kind,
            root_exit_code=returncode,
            stdout_tail=output.stdout_tail(),
            stderr_tail=output.stderr_tail(),
        )
    return _CodexExecutionResult(
        returncode=returncode,
        output=output.text(),
        root_exit_code=returncode,
        stdout_tail=output.stdout_tail(),
        stderr_tail=output.stderr_tail(),
    )


def _completed_codex_failure(returncode: int, output: str) -> tuple[str | None, str | None]:
    """Classify completed Codex runs that did not produce an implementation result."""
    no_assistant = _codex_transcript_without_assistant_result(output)
    if returncode == 0 and no_assistant:
        return ("codex_exited_zero_without_assistant_result", "Codex exited 0 without producing an assistant/result turn")
    if returncode != 0 and no_assistant:
        return (
            "codex_exited_nonzero_without_assistant_result",
            f"Codex exited {returncode} without producing an assistant/result turn",
        )
    if returncode != 0:
        return ("codex_exited_nonzero", f"Codex exited with non-zero status {returncode}")
    return (None, None)


class _ProcessOutputBuffer:
    """Thread-safe process output accumulator."""

    def __init__(self) -> None:
        self._chunks: list[tuple[str, str]] = []
        self._terminal_failure_reason: str | None = None
        self._lock = threading.Lock()

    def append(self, text: str, *, stream_name: str) -> None:
        """Append one output chunk."""
        with self._lock:
            self._chunks.append((stream_name, text))

    def text(self) -> str:
        """Return accumulated output."""
        with self._lock:
            return "".join(text for _, text in self._chunks)

    def stdout_tail(self) -> str:
        """Return a bounded stdout tail for diagnostics."""
        with self._lock:
            return _tail_text("".join(text for stream, text in self._chunks if stream == "stdout"))

    def stderr_tail(self) -> str:
        """Return a bounded stderr tail for diagnostics."""
        with self._lock:
            return _tail_text("".join(text for stream, text in self._chunks if stream == "stderr"))

    def mark_terminal_failure(self, reason: str) -> None:
        """Remember the first terminal executor failure reason."""
        with self._lock:
            if self._terminal_failure_reason is None:
                self._terminal_failure_reason = reason

    def terminal_failure_reason(self) -> str | None:
        """Return the first terminal executor failure reason."""
        with self._lock:
            return self._terminal_failure_reason


def _read_process_stream(
    stream: object,
    output: _ProcessOutputBuffer,
    errors: queue.SimpleQueue[str],
    output_mode: str,
    terminal_failure: threading.Event,
    stream_name: str,
) -> None:
    """Read process output without blocking the watchdog loop."""
    try:
        for chunk in stream:
            if output_mode == "echo":
                target = sys.stderr if stream_name == "stderr" else sys.stdout
                target.write(chunk)
                target.flush()
            output.append(chunk, stream_name=stream_name)
            reason = _terminal_executor_failure_reason(output.text())
            if reason is not None:
                output.mark_terminal_failure(reason)
                terminal_failure.set()
                return
    except Exception as exc:  # noqa: BLE001  # Reader failures must not stop timeout/interruption cleanup.
        errors.put(str(exc) or type(exc).__name__)


def _terminal_executor_failure_reason(output: str) -> str | None:
    """Return a terminal executor failure reason when Codex output proves local execution is unavailable."""
    diagnostic_output = _codex_result_region(output)
    if not diagnostic_output.strip():
        return None
    lower = diagnostic_output.lower()
    for pattern in _CODEX_TERMINAL_EXECUTOR_FAILURE_PATTERNS:
        if pattern.lower() in lower:
            return pattern
    return None


def _codex_transcript_without_assistant_result(output: str) -> bool:
    """Return whether Codex produced only its session/user prompt transcript."""
    if "OpenAI Codex" not in output or "Codex Prompt:" not in output:
        return False
    lines = [line.strip().lower() for line in output.splitlines()]
    return "user" in lines and "assistant" not in lines and not _codex_result_region(output).strip()


def _codex_result_region(output: str) -> str:
    """Return the assistant/result region, excluding echoed prompt contract text."""
    lines = output.splitlines()
    prompt_index = _last_line_index(lines, "Codex Prompt:")
    if prompt_index is None:
        return output
    assistant_index = _first_exact_line_index(lines, "assistant", start=prompt_index + 1)
    if assistant_index is not None:
        return "\n".join(lines[assistant_index + 1 :])
    policy_index = _last_line_index(lines, "Local executor policy:", start=prompt_index)
    if policy_index is not None:
        blank_index = _first_blank_line_after(lines, policy_index)
        if blank_index is not None:
            return "\n".join(lines[blank_index + 1 :])
    return ""


def _last_line_index(lines: list[str], prefix: str, *, start: int = 0) -> int | None:
    """Return the last line index whose stripped text starts with `prefix`."""
    index: int | None = None
    for current in range(start, len(lines)):
        if lines[current].strip().startswith(prefix):
            index = current
    return index


def _first_exact_line_index(lines: list[str], value: str, *, start: int = 0) -> int | None:
    """Return the first line index whose stripped text matches `value`."""
    for current in range(start, len(lines)):
        if lines[current].strip().lower() == value:
            return current
    return None


def _first_blank_line_after(lines: list[str], start: int) -> int | None:
    """Return the first blank line index after `start`."""
    for current in range(start + 1, len(lines)):
        if not lines[current].strip():
            return current
    return None


def _tail_text(text: str) -> str:
    """Return a bounded tail for process diagnostics."""
    if not text:
        return "(none)"
    if len(text) <= _CODEX_DIAGNOSTIC_TAIL_CHARS:
        return text
    omitted = len(text) - _CODEX_DIAGNOSTIC_TAIL_CHARS
    return f"... <truncated {omitted} chars>\n{text[-_CODEX_DIAGNOSTIC_TAIL_CHARS:]}"


def _wait_for_process(process: subprocess.Popen[str], *, timeout: float) -> int | None:
    """Wait briefly for process exit and return `None` while still running."""
    try:
        return process.wait(timeout=timeout)
    except TypeError:
        return process.wait()
    except subprocess.TimeoutExpired:
        return None


def _join_reader(reader: threading.Thread) -> None:
    """Join the output reader briefly without letting it block cancellation."""
    reader.join(timeout=_CODEX_READER_JOIN_SECONDS)


def _join_readers(readers: tuple[threading.Thread, ...]) -> None:
    """Join output readers briefly without letting them block cancellation."""
    for reader in readers:
        _join_reader(reader)


def _process_exit_code(process: object) -> int | None:
    """Return the current process exit code when available."""
    poll = getattr(process, "poll", None)
    if not callable(poll):
        return None
    try:
        value = poll()
    except Exception:  # noqa: BLE001  # Diagnostics must not crash cleanup.
        return None
    return value if isinstance(value, int) else None


def _close_owned_process(owned: _OwnedCodexProcess) -> None:
    """Release process ownership after normal exit."""
    if sys.platform == "win32":
        _close_windows_job_handle(owned)


def _first_queue_item(items: queue.SimpleQueue[str]) -> str | None:
    """Return one queued item if present."""
    try:
        return items.get_nowait()
    except queue.Empty:
        return None


def _write_candidate_execute_heartbeat(process: object, *, elapsed_seconds: float) -> None:
    """Render a visible liveness heartbeat for long candidate executions."""
    pid = getattr(process, "pid", None)
    pid_text = str(pid) if isinstance(pid, int) and pid > 0 else "(unknown)"
    sys.stdout.write(f"\n[candidate-execute heartbeat] elapsed={elapsed_seconds:.0f}s root_process_id={pid_text}\n")
    sys.stdout.flush()


def _own_codex_process(process: subprocess.Popen[str]) -> _OwnedCodexProcess:
    """Attach platform ownership to a launched Codex process."""
    if sys.platform != "win32":
        return _OwnedCodexProcess(process=process)
    try:
        return _OwnedCodexProcess(process=process, windows_job_handle=_create_windows_kill_on_close_job(process))
    except Exception as exc:  # noqa: BLE001  # Job ownership diagnostics must not prevent fallback cleanup.
        return _OwnedCodexProcess(process=process, ownership_error=str(exc) or type(exc).__name__)


def _create_windows_kill_on_close_job(process: subprocess.Popen[str]) -> int:
    """Create a Windows Job Object that kills the process tree when closed."""
    handle = _create_windows_job_handle()
    try:
        _configure_windows_job_kill_on_close(handle)
        _assign_process_to_windows_job(handle, process)
    except Exception:
        ctypes.windll.kernel32.CloseHandle(handle)
        raise
    return handle


def _create_windows_job_handle() -> int:
    """Create a Windows Job Object handle."""
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        msg = "CreateJobObjectW failed"
        raise OSError(msg)
    return int(handle)


def _configure_windows_job_kill_on_close(handle: int) -> None:
    """Set `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` on a Windows Job Object."""
    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x00002000

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_ulong),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_ulong),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_ulong),
            ("SchedulingClass", ctypes.c_ulong),
        ]

    class JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JobObjectBasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = JobObjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    kernel32 = ctypes.windll.kernel32
    kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    ok = kernel32.SetInformationJobObject(
        handle,
        job_object_extended_limit_information,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        msg = "SetInformationJobObject failed"
        raise OSError(msg)


def _assign_process_to_windows_job(handle: int, process: subprocess.Popen[str]) -> None:
    """Assign a process to a Windows Job Object."""
    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        msg = "process handle unavailable for Job Object assignment"
        raise OSError(msg)
    kernel32 = ctypes.windll.kernel32
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    ok = kernel32.AssignProcessToJobObject(handle, int(process_handle))
    if not ok:
        msg = "AssignProcessToJobObject failed"
        raise OSError(msg)


def _terminate_process_tree(target: subprocess.Popen[str] | _OwnedCodexProcess, *, method: str) -> _ProcessTerminationReport:
    """Terminate the foreground Codex process tree where the platform allows it."""
    owned = target if isinstance(target, _OwnedCodexProcess) else _OwnedCodexProcess(process=target)
    process = owned.process
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        terminated = _terminate_direct_process(process)
        return _ProcessTerminationReport(
            root_pid=None,
            tracked_pids=(),
            method=method,
            terminated_pids=() if not terminated else (0,),
            resisted_pids=(0,) if not terminated else (),
            diagnostics_error=owned.ownership_error,
        )
    diagnostics_error: str | None = None
    descendants: tuple[int, ...] = ()
    try:
        descendants = _collect_descendant_pids(pid)
    except Exception as exc:  # noqa: BLE001  # Diagnostics must not block cleanup.
        diagnostics_error = str(exc)
    diagnostics_error = _join_diagnostics(diagnostics_error, owned.ownership_error)
    tracked = _ordered_unique_pids((pid, *descendants))
    if sys.platform == "win32" and owned.windows_job_handle is not None:
        _close_windows_job_handle(owned)
    elif sys.platform == "win32":
        try:
            subprocess.run(  # noqa: S603  # Fixed Windows process-tree termination command.
                ("taskkill", "/PID", str(pid), "/T", "/F"),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            _terminate_direct_process(process)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            _terminate_direct_process(process)
    try:
        process.wait(timeout=_CODEX_TERMINATION_WAIT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        _terminate_direct_process(process)
    resisted = _running_pids(tracked)
    terminated = tuple(item for item in tracked if item not in resisted)
    return _ProcessTerminationReport(
        root_pid=pid,
        tracked_pids=tracked,
        method=method,
        terminated_pids=terminated,
        resisted_pids=resisted,
        diagnostics_error=diagnostics_error,
    )


def _close_windows_job_handle(owned: _OwnedCodexProcess) -> None:
    """Close a Windows Job Object handle, triggering kill-on-close."""
    handle = owned.windows_job_handle
    if handle is None:
        return
    try:
        ctypes.windll.kernel32.CloseHandle(handle)
    finally:
        owned.windows_job_handle = None


def _join_diagnostics(left: str | None, right: str | None) -> str | None:
    """Join optional diagnostics without raising."""
    parts = tuple(part for part in (left, right) if part)
    if not parts:
        return None
    return "; ".join(parts)


def _collect_descendant_pids(root_pid: int) -> tuple[int, ...]:
    """Collect descendants for a root process from the platform process table."""
    parent_map = _windows_process_parent_map() if sys.platform == "win32" else _posix_process_parent_map()
    descendants: list[int] = []
    pending = [root_pid]
    while pending:
        parent = pending.pop(0)
        children = sorted(pid for pid, ppid in parent_map.items() if ppid == parent)
        descendants.extend(children)
        pending.extend(children)
    return tuple(descendants)


def _windows_process_parent_map() -> dict[int, int]:
    """Return Windows process parent relationships using a Toolhelp snapshot."""
    max_path = 260
    th32cs_snapprocess = 0x00000002
    invalid_handle_value = ctypes.c_void_p(-1).value

    class ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * max_path),
        ]

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    if snapshot == invalid_handle_value:
        msg = "CreateToolhelp32Snapshot failed"
        raise OSError(msg)
    try:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(ProcessEntry32)
        parent_map: dict[int, int] = {}
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            msg = "Process32FirstW failed"
            raise OSError(msg)
        while True:
            parent_map[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
        return parent_map
    finally:
        kernel32.CloseHandle(snapshot)


def _posix_process_parent_map() -> dict[int, int]:
    """Return POSIX process parent relationships from `ps` output."""
    completed = subprocess.run(
        ("ps", "-eo", "pid=,ppid="),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        msg = completed.stderr.strip() or "ps process table query failed"
        raise OSError(msg)
    parent_map: dict[int, int] = {}
    expected_columns = 2
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != expected_columns:
            continue
        try:
            child, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        parent_map[child] = parent
    return parent_map


def _ordered_unique_pids(pids: tuple[int, ...]) -> tuple[int, ...]:
    """Return positive PIDs in first-seen order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for pid in pids:
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
    return tuple(ordered)


def _running_pids(pids: tuple[int, ...]) -> tuple[int, ...]:
    """Return PIDs that still appear alive."""
    return tuple(pid for pid in pids if _is_pid_running(pid))


def _is_pid_running(pid: int) -> bool:
    """Return whether a PID appears to still be running."""
    if sys.platform == "win32":
        return _is_windows_pid_running(pid)
    proc_state_index = 2
    proc_stat = Path("/proc") / str(pid) / "stat"
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding="utf-8", errors="replace").split()
        except OSError:
            fields = []
        if len(fields) > proc_state_index and fields[proc_state_index] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_windows_pid_running(pid: int) -> bool:
    """Return whether a Windows PID can be opened for synchronization."""
    process_query_limited_information = 0x00001000
    still_active = 259
    inherit_handle = False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, inherit_handle, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _terminate_direct_process(process: object) -> bool:
    """Best-effort direct-process termination fallback."""
    for method_name in ("terminate", "kill"):
        method = getattr(process, method_name, None)
        if method is None:
            continue
        try:
            method()
        except OSError:
            continue
        return True
    return False


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


def _render_candidate_execute_termination(
    candidate: Path,
    *,
    output: Path | None,
    repo: str | None,
    result: _CodexExecutionResult,
) -> str:
    """Render deterministic termination status for `candidate-execute`."""
    report = result.termination_report
    status = "interrupted_by_operator"
    if result.timed_out:
        status = "timeout"
    elif result.failed:
        status = "failure"
    termination = "process_tree_terminated"
    if report is not None and report.resisted_pids:
        termination = "termination_incomplete"
    output_row = f"raw_output_saved: {output.resolve()}" if output is not None else "raw_output_saved: (not requested)"
    save_hint = ""
    if output is None:
        save_hint_path = candidate.with_suffix(".codex-output.txt").resolve()
        save_hint = f"rerun_to_save_raw_output: add --output {save_hint_path}\n"
    diagnostics = _render_process_termination_report(report)
    failure = f"Failure: {result.failure_message}\n" if result.failure_message else ""
    failure_details = _render_candidate_execute_failure_diagnostics(result)
    next_command = ""
    if output is not None:
        next_command = f"\nNext recommended command:\n{_candidate_outcome_next_command(candidate, output=output, repo=repo)}\n"
    return f"""
Candidate execution status: {status}
Process tree status: {termination}
{failure}{failure_details}{diagnostics}
{output_row}
{save_hint}
{next_command}"""


def _render_candidate_execute_failure_diagnostics(result: _CodexExecutionResult) -> str:
    """Render Codex launcher/stream diagnostics for failed candidate execution."""
    if not result.failed and not result.timed_out and not result.interrupted:
        return ""
    rows = [
        "Candidate execution diagnostics:",
        f"- failure_kind: {result.failure_kind or '(unknown)'}",
        f"- root_exit_code: {result.root_exit_code if result.root_exit_code is not None else '(unknown)'}",
        f"- no_assistant_result: {_yes_no(value=_codex_transcript_without_assistant_result(result.output))}",
        "- stdout_tail:",
        _indent_block(result.stdout_tail or _tail_text(result.output)),
        "- stderr_tail:",
        _indent_block(result.stderr_tail or "(none)"),
    ]
    return "\n".join(rows) + "\n"


def _yes_no(*, value: bool) -> str:
    """Render a boolean diagnostic as yes/no."""
    return "yes" if value else "no"


def _indent_block(text: str) -> str:
    """Indent a multiline diagnostic block."""
    return "\n".join(f"  {line}" for line in (text or "(none)").splitlines())


def _render_process_termination_report(report: _ProcessTerminationReport | None) -> str:
    """Render process-tree termination diagnostics."""
    if report is None:
        return "Process termination diagnostics: unavailable\n"
    rows = [
        "Process termination diagnostics:",
        f"- root_process_id: {report.root_pid if report.root_pid is not None else '(unknown)'}",
        f"- tracked_process_count: {len(report.tracked_pids)}",
        f"- termination_method: {report.method}",
        f"- terminated_processes: {_format_pid_tuple(report.terminated_pids)}",
        f"- resisted_processes: {_format_pid_tuple(report.resisted_pids)}",
    ]
    if report.diagnostics_error:
        rows.append(f"- diagnostics_error: {report.diagnostics_error}")
    return "\n".join(rows) + "\n"


def _format_pid_tuple(pids: tuple[int, ...]) -> str:
    """Render a PID tuple for diagnostics."""
    if not pids:
        return "(none)"
    return ", ".join(str(pid) for pid in pids)


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
    elif args.command == "executor-status":
        result = _run_executor_status(args, parser)
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
