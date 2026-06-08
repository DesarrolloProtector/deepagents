import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from deepagents.harnesses.protector import _engineering as engineering, cli


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_repo(root: Path) -> Path:
    _write(root / "AGENTS.md")
    _write(root / "MEMORY.md")
    _write(root / ".codex" / "skills" / "payment-alpha" / "SKILL.md")
    _write(root / ".codex" / "skills" / "payment-beta" / "SKILL.md")
    _write(root / ".codex" / "skills" / "payment-gamma" / "SKILL.md")
    _write(root / ".codex" / "skills" / "payment-delta" / "SKILL.md")
    _write(root / ".codex" / "skills" / "unrelated" / "SKILL.md")
    _write(root / ".codex" / "skills" / "nested-only" / "child" / "SKILL.md")
    _write(root / "Docs" / "flows" / "payment-a.md")
    _write(root / "Docs" / "flows" / "payment-b.md")
    _write(root / "Docs" / "flows" / "payment-c.md")
    _write(root / "Docs" / "flows" / "nested" / "payment-nested.md")
    _write(root / ".codex" / "agent-workflow" / "feature-contract-template.md")
    return root


def _write_repo_config(monkeypatch, path: Path, aliases: dict[str, Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({name: str(repo) for name, repo in aliases.items()}), encoding="utf-8")
    monkeypatch.setenv(cli.REPOS_CONFIG_ENV_VAR, str(path))
    return path


def _render(repo: Path, task: str) -> engineering.RenderedOutput:
    return engineering.render_output(
        task=task,
        repo=repo,
        mode="auto",
        harness_profile=engineering.HARNESS_PROFILE,
        real_model=None,
        agent_type="CompiledStateGraph",
        output=None,
    )


STRUCTURED_SIGNATURE_REGRESSION_TASK = """Title: Restore missing platform-company signature action

Current regression:

After the successful Lleida.net integration fix:

* ProviderStatus = Success
* ProviderCorrelationId exists
* Signature status remains pending
* Company activation remains pending
* "Ver estado firma" is visible
* "Enviar a firmar" disappeared

This is a workflow dead-end.

Expected behavior:

Provider dispatch success means only that Lleida.net accepted the request.

It does NOT mean:

* signature completed;
* contract activated;
* operator actions finished.

The operator must still have the appropriate send/retry signature action while the contract remains pending.

Objective:

Find the exact action-eligibility condition that hides the signature action after ProviderStatus=Success and restore the correct behavior.

Restrictions:

* Do not modify ConfigId handling.
* Do not modify SET_CONFIG logic.
* Do not modify PDF generation.
* Do not modify START_SIGNATURE payloads.
* Do not modify onboarding, preview, approval semantics or signature lifecycle.
* Do not redesign the workflow.
* Fix only the regression.

Validation:

* Build.
* Prove which condition currently hides the action.
* Verify a contract with:

  * ProviderStatus=Success
  * CorrelationId present
  * Signature still pending
    renders the signature action again.
* Verify "Ver estado firma" still works.

PASS only if the action is restored and the successful Lleida dispatch flow remains intact."""


def test_inspects_only_bounded_repo_artifacts(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)

    context = engineering._inspect_repo_context(repo)

    assert tuple(path.name for path in context.mandatory) == ("AGENTS.md", "MEMORY.md")
    assert tuple(name for name, _ in context.skills) == (
        "payment-alpha",
        "payment-beta",
        "payment-delta",
        "payment-gamma",
        "unrelated",
    )
    assert tuple(path.name for path in context.flows) == ("payment-a.md", "payment-b.md", "payment-c.md")
    assert context.feature_contract == repo / ".codex" / "agent-workflow" / "feature-contract-template.md"
    assert context.warnings == ()


def test_context_selection_limits_skills_and_flows(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Plan payment workflow behavior change")

    assert "- AGENTS.md" in rendered.payload
    assert "- MEMORY.md" in rendered.payload
    assert "- .codex/skills/payment-alpha/SKILL.md" in rendered.payload
    assert "- .codex/skills/payment-beta/SKILL.md" in rendered.payload
    assert "- .codex/skills/payment-delta/SKILL.md" in rendered.payload
    assert "- Docs/flows/payment-a.md" in rendered.payload
    assert "- Docs/flows/payment-b.md" in rendered.payload
    assert "- .codex/skills/payment-gamma/SKILL.md" not in rendered.codex_prompt
    assert "- Docs/flows/payment-c.md" not in rendered.codex_prompt
    assert rendered.codex_prompt.count(".codex/skills/") == 3
    assert rendered.codex_prompt.count("Docs/flows/") == 2
    assert ".codex/skills/payment-gamma/SKILL.md" in rendered.payload


def test_feature_contract_selected_only_for_feature_or_contract_tasks(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)

    feature_rendered = _render(repo, "Change public payment workflow contract behavior")
    review_rendered = _render(repo, "Review payment implementation notes")

    assert "- .codex/agent-workflow/feature-contract-template.md" in feature_rendered.codex_prompt
    assert "- .codex/agent-workflow/feature-contract-template.md" not in review_rendered.codex_prompt
    assert ".codex/agent-workflow/feature-contract-template.md (task does not suggest feature/contract behavior)" in review_rendered.payload


def test_codex_prompt_uses_selected_context_only(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Plan payment workflow behavior change")

    assert "Selected context paths:" in rendered.codex_prompt
    assert ".codex/skills/payment-gamma/SKILL.md" not in rendered.codex_prompt
    assert "Docs/flows/payment-c.md" not in rendered.codex_prompt
    assert "(no task keyword match)" not in rendered.codex_prompt
    assert "Not Selected:" not in rendered.codex_prompt


def test_fix_task_generates_surgical_implementation_prompt(tmp_path: Path) -> None:
    rendered = _render(
        _build_repo(tmp_path),
        "Fix regression: missing resend signature action after successful workflow state",
    )

    assert "Task mode: implementation_fix" in rendered.codex_prompt
    assert "Do not edit files unless the caller explicitly grants editing" not in rendered.codex_prompt
    assert "start with read-only inspection" not in rendered.codex_prompt
    assert "Inspect only enough code to locate the faulty condition, then fix surgically." in rendered.codex_prompt
    assert "Scoped edits are allowed when needed to fix the requested issue." in rendered.codex_prompt
    assert "Do not redesign unrelated workflows." in rendered.codex_prompt


def test_review_task_remains_read_only(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Review payment workflow behavior")

    assert "Task mode: review_only" in rendered.codex_prompt
    assert "Do not edit files unless explicitly requested by this task." in rendered.codex_prompt
    assert "start with read-only inspection of the selected context" in rendered.codex_prompt
    assert "Scoped edits are allowed when needed to fix the requested issue." not in rendered.codex_prompt


def test_planning_task_forbids_implementation(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Plan arquitectura roadmap for payment workflow")

    assert "Task mode: planning_only" in rendered.codex_prompt
    assert "Do not implement changes; produce planning/design output only." in rendered.codex_prompt
    assert "Do not turn the plan into code changes." in rendered.codex_prompt
    assert "Scoped edits are allowed" not in rendered.codex_prompt


def test_diagnostic_bootstrap_allows_bounded_execution_only(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Diagnostic bootstrap: smoke real provider and configure SET_CONFIG")

    assert "Task mode: diagnostic_bootstrap" in rendered.codex_prompt
    assert "Bounded diagnostic execution or configuration changes are allowed only when the task explicitly asks for them." in rendered.codex_prompt
    assert "Do not modify provider/runtime state beyond the requested diagnostic or bootstrap scope." in rendered.codex_prompt


def test_continuation_followup_preserves_previous_fixes(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Continuation follow-up: preserve previous fix and repair regression after fix")

    assert "Task mode: continuation_followup" in rendered.codex_prompt
    assert "Preserve validated previous fixes" in rendered.codex_prompt
    assert "Restrictions/PASS boundaries as must-not-touch items" in rendered.codex_prompt


def test_ui_regression_preserves_state_and_requires_condition_proof(tmp_path: Path) -> None:
    task = """Title: Restore hidden modal button

Current regression:
Observed UI state: spinner remains visible and resend button is hidden.

Expected behavior:
Expected UI state: resend button renders after the modal closes.

Objective:
Fix the UI regression in the Razor/JS DOM condition."""
    rendered = _render(_build_repo(tmp_path), task)

    assert "Task mode: ui_runtime_bug" in rendered.codex_prompt
    assert "Observed UI state: spinner remains visible and resend button is hidden." in rendered.codex_prompt
    assert "Expected UI state: resend button renders after the modal closes." in rendered.codex_prompt
    assert "Preserve observed UI state and expected UI state" in rendered.codex_prompt
    assert "Prove the exact condition that hides or renders" in rendered.codex_prompt


def test_provider_api_regression_protects_unrelated_working_provider_pieces(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Fix provider API dispatch regression without touching config_id or START_SIGNATURE payload")

    assert "Task mode: provider_api_bug" in rendered.codex_prompt
    assert "Preserve working provider, config, payload, and dispatch behavior unless the task specifically targets it." in rendered.codex_prompt
    provider_guardrail = "Do not modify ConfigId, SET_CONFIG, START_SIGNATURE, payload, or dispatch behavior unless the task specifically targets it."
    assert provider_guardrail in rendered.codex_prompt


def test_structured_regression_task_preserves_details(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), STRUCTURED_SIGNATURE_REGRESSION_TASK)

    assert "Title: Restore missing platform-company signature action" in rendered.codex_prompt
    assert "Task mode: continuation_followup" in rendered.codex_prompt
    expected_objective = (
        "Objective: Find the exact action-eligibility condition that hides the signature action after ProviderStatus=Success "
        "and restore the correct behavior."
    )
    assert expected_objective in rendered.codex_prompt
    assert "Current regression:\nAfter the successful Lleida.net integration fix:" in rendered.codex_prompt
    assert '* "Enviar a firmar" disappeared' in rendered.codex_prompt
    assert "Expected behavior:\nProvider dispatch success means only that Lleida.net accepted the request." in rendered.codex_prompt
    assert "Restrictions:\n* Do not modify ConfigId handling." in rendered.codex_prompt
    assert "* Fix only the regression." in rendered.codex_prompt
    assert "Validation:\n* Build." in rendered.codex_prompt
    assert "renders the signature action again." in rendered.codex_prompt
    assert "Objective: Objective:" not in rendered.codex_prompt


def test_write_prompt_output_refuses_overwrite_unless_allowed(tmp_path: Path) -> None:
    output = tmp_path / "prompt.md"

    engineering.write_prompt_output(output, "Codex Prompt:\nfirst", overwrite=False)
    with pytest.raises(engineering.HarnessUsageError, match="pass --overwrite"):
        engineering.write_prompt_output(output, "Codex Prompt:\nsecond", overwrite=False)

    engineering.write_prompt_output(output, "Codex Prompt:\nsecond", overwrite=True)

    assert output.read_text(encoding="utf-8") == "Codex Prompt:\nsecond\n"


def test_reviewer_findings_pass_for_good_output(tmp_path: Path) -> None:
    output = tmp_path / "codex-good.txt"
    good = """Files read
- AGENTS.md

Files changed
- (none)

Summary
- Reviewed bounded context.

Validation
- ruff check passed.

PASS
"""

    findings = engineering.render_review_findings(output, "Review payment workflow", good)

    assert "Status: PASS" in findings
    assert "Missing sections:\n- (none)" in findings
    assert "Drift warnings:\n- (none)" in findings
    assert "Validation warnings:\n- (none)" in findings


def test_reviewer_findings_review_needed_for_missing_drift_and_validation_gap(tmp_path: Path) -> None:
    output = tmp_path / "codex-bad.txt"
    bad = """Summary
- Added dashboard memory graph MCP governance documentation changes and archived the proposal.

Validation
- not run

PASS
"""

    findings = engineering.render_review_findings(output, "Review payment workflow", bad)

    assert "Status: REVIEW_NEEDED" in findings
    assert "- Files read" in findings
    assert "- Files changed" in findings
    assert "- dashboard mentioned without task scope" in findings
    assert "- mcp mentioned without task scope" in findings
    assert "- PASS claimed without validation evidence" in findings
    assert "Suggested follow-up prompt:\n- Revise the Codex output" in findings


def test_cli_status_reports_profile_registration(capsys) -> None:
    assert cli.main(["status"]) == 0

    output = capsys.readouterr().out
    assert "Protector Harness Status" in output
    assert "deepagents version:" in output
    assert "profile protector:engineering-harness registered: yes" in output
    assert "cwd:" in output


def test_cli_repos_prints_known_aliases(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(cli.REPOS_CONFIG_ENV_VAR, str(tmp_path / "missing" / "repos.json"))

    assert cli.main(["repos"]) == 0

    output = capsys.readouterr().out
    assert "Protector Harness Repos" in output
    assert "config:" in output
    assert "(missing)" in output
    assert "- FinanciacionCore:" in output
    assert "- deepagents:" in output


def test_cli_repos_init_creates_config_and_refuses_overwrite(tmp_path: Path, monkeypatch, capsys) -> None:
    config = tmp_path / "repos.json"
    monkeypatch.setenv(cli.REPOS_CONFIG_ENV_VAR, str(config))

    assert cli.main(["repos", "--init"]) == 0
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["FinanciacionCore"] == r"C:\Users\DesarrolladorProtect\source\repos\FinanciacionCore"
    assert data["deepagents"] == r"C:\Users\DesarrolladorProtect\source\repos\deepagents"

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["repos", "--init"])

    assert exc_info.value.code == 2
    assert "repo alias config already exists" in capsys.readouterr().err

    assert cli.main(["repos", "--init", "--overwrite"]) == 0


def test_cli_task_copies_codex_prompt_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    copied: list[str] = []

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["task", "--repo", str(repo), "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert f"Repo: {repo}" in stdout
    assert "Selected context count: 8" in stdout
    assert "Codex prompt copied to clipboard" in stdout
    assert "Codex-ready engineering harness prompt" not in stdout
    assert copied == [_render(repo, "small payment workflow task").codex_prompt]
    assert copied[0].startswith("Codex Prompt:\n")
    assert "Selected Context:" not in copied[0]


def test_cli_task_no_copy_prints_prompt_without_clipboard(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used with --no-copy"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["task", "--repo", str(repo), "--no-copy", "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Codex-ready engineering harness prompt" in stdout
    assert "Task: small payment workflow task" in stdout
    assert "Codex Prompt:" in stdout


def test_cli_task_writes_output_and_copies_prompt(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    output = tmp_path / "prompt.md"
    copied: list[str] = []

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["task", "--repo", str(repo), "--output", str(output), "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Codex prompt copied to clipboard" in stdout
    assert output.read_text(encoding="utf-8") == f"{copied[0]}\n"


def test_cli_task_prints_prompt_when_clipboard_unavailable(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        assert text.startswith("Codex Prompt:\n")
        return cli._ClipboardCopyResult(copied=False, error="no clipboard backend")

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["task", "--repo", str(repo), "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Codex prompt was not copied" in stdout
    assert "Clipboard unavailable: no clipboard backend" in stdout
    assert "Codex-ready engineering harness prompt" in stdout
    assert "Codex Prompt:" in stdout


def test_cli_task_accepts_config_repo_alias(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    _write_repo_config(monkeypatch, tmp_path / "repos.json", {"TestRepo": repo})

    assert cli.main(["task", "TestRepo", "--no-copy", "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert f"Repo: {repo}" in stdout
    assert "Task: small payment workflow task" in stdout


def test_cli_task_accepts_positional_repo_path(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")

    assert cli.main(["task", str(repo), "--no-copy", "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert f"Repo: {repo}" in stdout
    assert "Task: small payment workflow task" in stdout


def test_cli_task_uses_builtin_fallback_alias(capsys) -> None:
    assert cli.main(["task", "deepagents", "--no-copy", "small", "test", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Repo: C:\\Users\\DesarrolladorProtect\\source\\repos\\deepagents" in stdout
    assert "Task: small test task" in stdout


def test_cli_run_passes_with_synthetic_good_output(monkeypatch, capsys) -> None:
    copied: list[str] = []
    stdin = StringIO(
        """small test task
END
Files read
- AGENTS.md

Files changed
- (none)

Summary
- Reviewed bounded context.

Validation
- ruff check passed.

PASS
END
""",
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["run", "deepagents"]) == 0

    stdout = capsys.readouterr().out
    assert "Paste or type the task. End input with a line containing only END." in stdout
    assert "Codex prompt copied to clipboard" in stdout
    assert "Paste this into your already-open Codex session." in stdout
    assert "Paste Codex output. End input with a line containing only END." in stdout
    assert "Status: PASS" in stdout
    assert stdout.rstrip().endswith("PASS")
    assert len(copied) == 1
    assert copied[0].startswith("Codex Prompt:\n")


def test_cli_run_review_needed_copies_and_prints_follow_up(monkeypatch, capsys) -> None:
    copied: list[str] = []
    stdin = StringIO(
        """small payment workflow task
END
Summary
- Added dashboard memory graph MCP governance documentation changes.

Validation
- not run

PASS
END
""",
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["run", "deepagents"]) == 0

    stdout = capsys.readouterr().out
    assert "Status: REVIEW_NEEDED" in stdout
    assert "Suggested follow-up prompt:" in stdout
    assert "Follow-up prompt copied to clipboard." in stdout
    assert "Follow-up prompt:\nRevise the Codex output" in stdout
    assert len(copied) == 2
    assert copied[0].startswith("Codex Prompt:\n")
    assert copied[1].startswith("Revise the Codex output")


def test_cli_task_unknown_alias_fails(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["task", "UnknownRepoAlias", "small", "task"])

    assert exc_info.value.code == 2
    assert "unknown repo alias or missing repo path: UnknownRepoAlias" in capsys.readouterr().err


def test_cli_review_uses_deterministic_reviewer(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text(
        """Summary
- Added dashboard memory graph MCP governance documentation changes.

Validation
- not run

PASS
""",
        encoding="utf-8",
    )

    assert cli.main(["review", "--repo", str(repo), "--codex-output", str(codex_output), "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Reviewer Findings:" in stdout
    assert "Status: REVIEW_NEEDED" in stdout
    assert "- Files read" in stdout
    assert "- dashboard mentioned without task scope" in stdout


def test_cli_review_accepts_config_repo_alias(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    _write_repo_config(monkeypatch, tmp_path / "repos.json", {"TestRepo": repo})
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text(
        """Files read
- AGENTS.md

Files changed
- (none)

Summary
- Reviewed bounded context.

Validation
- smoke check passed.

PASS
""",
        encoding="utf-8",
    )

    assert cli.main(["review", "TestRepo", "--codex-output", str(codex_output), "small", "payment", "workflow", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Reviewer Findings:" in stdout
    assert "Status: PASS" in stdout
