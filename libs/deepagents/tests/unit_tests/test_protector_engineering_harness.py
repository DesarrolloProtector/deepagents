import json
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
