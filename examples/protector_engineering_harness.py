"""Smoke example for the `protector:engineering-harness` profile.

This example constructs a DeepAgent with a local dry-run chat model so profile
bootstrap and harness lookup can be verified without API credentials. It does
not invoke the agent or execute Codex.
"""
# ruff: noqa: INP001,E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING


def _bootstrap_source_checkout() -> None:
    """Make the example runnable from a source checkout with plain `python`."""
    project_dir = Path(__file__).resolve().parents[1] / "libs" / "deepagents"
    venv_dir = project_dir / ".venv"
    candidates = [
        venv_dir / "Lib" / "site-packages",
        *venv_dir.glob("lib/python*/site-packages"),
        project_dir,
    ]
    for candidate in reversed(candidates):
        if candidate.exists():
            sys.path.insert(0, str(candidate))


_bootstrap_source_checkout()

from deepagents.harnesses.protector._engineering import (
    HARNESS_PROFILE,
    HARNESS_PROFILE_ENV_VAR,
    HarnessUsageError,
    build_read_only_agent,
    render_output,
    render_review_findings,
    resolve_harness_profile,
    validate_real_model,
    write_prompt_output,
)
from deepagents.profiles.harness.harness_profiles import _get_harness_profile

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Construct a read-only Protector engineering harness smoke agent and print a Codex-ready payload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  python examples/protector_engineering_harness.py --harness-profile {HARNESS_PROFILE} --repo . "Review the planned payment-flow fix"
  {HARNESS_PROFILE_ENV_VAR}={HARNESS_PROFILE} python examples/protector_engineering_harness.py "Plan a bounded bug investigation"
""",
    )
    parser.add_argument(
        "--harness-profile",
        default=os.environ.get(HARNESS_PROFILE_ENV_VAR, HARNESS_PROFILE),
        help=f"Harness profile key to resolve in dry-run. Defaults to {HARNESS_PROFILE}; env: {HARNESS_PROFILE_ENV_VAR}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional real provider model spec for a future real-invocation path. It is recorded only and never called in this dry run.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Optional target repository path to include in the printed handoff. The script does not read or edit it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path for writing only the generated Codex Prompt section.",
    )
    parser.add_argument(
        "--codex-output",
        type=Path,
        default=None,
        help="Review a pasted Codex output text file instead of generating a prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --output to replace an existing file.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "planner", "reviewer"),
        default="auto",
        help="Instruction payload mode. Defaults to inferring the mode from the task.",
    )
    parser.add_argument("task", help="Task string to turn into a compact Codex handoff.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke example."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.codex_output is not None:
        if args.output is not None:
            parser.error("--output is only for prompt generation; reviewer mode prints findings to stdout")
        text = args.codex_output.read_text(encoding="utf-8")
        sys.stdout.write(render_review_findings(args.codex_output, args.task, text))
        sys.stdout.write("\n")
        return 0

    try:
        harness_profile = resolve_harness_profile(args.harness_profile)
        validate_real_model(args.model, harness_profile)
    except HarnessUsageError as exc:
        parser.error(str(exc))

    profile = _get_harness_profile(harness_profile)
    if profile is None:
        parser.error(f"built-in harness profile {harness_profile!r} is not registered")

    agent = build_read_only_agent(harness_profile)
    rendered = render_output(
        task=args.task,
        repo=args.repo,
        mode=args.mode,
        harness_profile=harness_profile,
        real_model=args.model,
        agent_type=type(agent).__name__,
        output=args.output,
    )
    if args.output is not None:
        try:
            write_prompt_output(args.output, rendered.codex_prompt, overwrite=args.overwrite)
        except HarnessUsageError as exc:
            parser.error(str(exc))
    sys.stdout.write(rendered.payload)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
