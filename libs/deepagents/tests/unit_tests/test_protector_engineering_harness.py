import json
import sys
from pathlib import Path

import pytest

from deepagents.harnesses.protector import _agentic as agentic, _ecc as ecc, _engineering as engineering, cli


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


def _build_ecc_repo(root: Path) -> Path:
    _write(root / "agents" / "code-reviewer.md")
    _write(root / "agents" / "planner.md")
    _write(root / "agents" / "unrelated.md")
    _write(root / "skills" / "prompt-optimizer" / "SKILL.md")
    _write(root / "skills" / "verification-loop" / "SKILL.md")
    _write(root / "skills" / "unrelated" / "SKILL.md")
    _write(
        root / "manifests" / "install-profiles.json",
        json.dumps(
            {
                "profiles": {
                    "developer": {
                        "description": "Default workflow profile with agents and quality gates.",
                        "modules": ["agents-core", "workflow-quality"],
                    },
                    "archive": {
                        "description": "Static archive export only.",
                        "modules": ["archive"],
                    },
                }
            }
        ),
    )
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


def _prompt_benchmarks_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "tests" / "prompt_benchmarks"


def _protector_pack_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "packs" / "protector-financiacioncore"


def _write_prompt_benchmark(root: Path, name: str, task: str, expected: str) -> Path:
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    (case / "task.txt").write_text(task, encoding="utf-8")
    (case / "expected_characteristics.md").write_text(expected, encoding="utf-8")
    return case


def _append_outcome_summary(
    repo: Path,
    *,
    status: str = "needs_review",
    deviations: tuple[str, ...] = (),
    validations: tuple[str, ...] = (),
    follow_up: str | None = None,
    benchmark_additions: tuple[str, ...] = (),
) -> Path:
    summary = engineering.OutcomeSummary(
        timestamp="2026-06-09T10:00:00Z",
        task="Fix legacy onboarding path convergence for operator UI views",
        selected_pack="protector-financiacioncore",
        selected_skills=(
            engineering.OutcomeSkillSummary(name="base_prompt_quality", source="runtime_generic"),
            engineering.OutcomeSkillSummary(name="implementation_fix", source="runtime_generic"),
            engineering.OutcomeSkillSummary(name="navigation_surface_convergence", source="pack"),
        ),
        status=status,
        changed_files=("Views/Unexpected.cshtml",),
        executed_validations=validations,
        deviations=deviations,
        follow_up_prompt=follow_up,
        suggested_benchmark_additions=benchmark_additions,
    )
    report = engineering.OutcomeReport(text="", status=status, follow_up=follow_up, summary=summary)
    return engineering.append_outcome_history(repo, report)


def _pack_prompt_skill_metadata_item(name: str, cases: list[str]) -> dict[str, object]:
    return {
        "name": name,
        "trigger_keywords": ["fixture"],
        "task_modes": ["implementation_fix"],
        "observed_state": "Observed state.",
        "expected_behavior": "Expected behavior.",
        "scope_rules": ["Scope rule."],
        "restriction_rules": ["Restriction rule."],
        "validation_expectations": ["Validation expectation."],
        "forbidden_generic_wording": [],
        "benchmark_cases": cases,
    }


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


def test_financiacioncore_short_task_uses_compact_repo_knowledge(tmp_path: Path) -> None:
    rendered = engineering.render_output(
        task="Fix legacy onboarding path convergence",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        mode="auto",
        harness_profile=engineering.HARNESS_PROFILE,
        real_model=None,
        agent_type="CompiledStateGraph",
        output=None,
    )

    assert "Repo knowledge:" in rendered.codex_prompt
    assert "protector-financiacioncore/knowledge/FinanciacionCore.md" in rendered.codex_prompt.replace("\\", "/")
    assert "Preserve contract-first company and financer onboarding as the promoted workflow." in rendered.codex_prompt
    assert "Legacy direct routes may stay backend-compatible, but should not be promoted in normal UI." in rendered.codex_prompt
    assert "Avoid broad audits unless the task explicitly requests one." in rendered.codex_prompt
    assert "Do not touch Contabilidad, telemetry, or resilience unless targeted." in rendered.codex_prompt
    assert "## Current phase" not in rendered.codex_prompt
    assert "## Protected decisions" not in rendered.codex_prompt
    knowledge_block = rendered.codex_prompt.split("Repo knowledge:\n", maxsplit=1)[1].split("\nScope boundaries:", maxsplit=1)[0]
    knowledge_lines = [line for line in knowledge_block.splitlines() if line.startswith("- ")]
    assert len(knowledge_lines) <= 6


def test_pack_knowledge_warns_when_legacy_duplicate_differs(tmp_path: Path, monkeypatch) -> None:
    pack = tmp_path / "pack"
    legacy = tmp_path / "legacy"
    pack_knowledge = pack / "knowledge" / "FinanciacionCore.md"
    legacy_knowledge = legacy / "FinanciacionCore.md"
    pack_knowledge.parent.mkdir(parents=True)
    legacy_knowledge.parent.mkdir(parents=True)
    pack_knowledge.write_text(
        """# FinanciacionCore

## Current phase

- MVP convergence toward a usable SaaS as soon as possible.
""",
        encoding="utf-8",
    )
    legacy_knowledge.write_text(
        """# FinanciacionCore

## Current phase

- Legacy-only divergent content.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        engineering,
        "discover_protector_pack",
        lambda: ecc.ProtectorPackDiscovery(
            path=pack,
            found=True,
            name="protector-financiacioncore",
            skills_count=5,
            knowledge_count=1,
            capabilities=(
                "workflow_convergence",
                "provider_diagnostics",
                "onboarding_convergence",
                "localization_completion",
                "navigation_convergence",
                "form_security",
            ),
            capabilities_validation_status="valid",
            validation_status="valid",
            prompt_skills_count=9,
            prompt_skills_validation_status="valid",
            benchmark_sets_count=1,
            benchmark_cases_count=8,
            benchmark_runnable=True,
            benchmark_validation_status="not_checked",
            warnings=(),
            capability_warnings=(),
            prompt_skill_warnings=(),
            benchmark_warnings=(),
        ),
    )
    monkeypatch.setattr(engineering, "_knowledge_directories", lambda: (legacy,))

    rendered = engineering.render_output(
        task="Fix legacy onboarding path convergence",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        mode="auto",
        harness_profile=engineering.HARNESS_PROFILE,
        real_model=None,
        agent_type="CompiledStateGraph",
        output=None,
    )

    assert str(pack_knowledge.resolve()) in rendered.payload
    assert "pack knowledge differs from legacy duplicate for FinanciacionCore" in rendered.payload
    assert "Legacy-only divergent content" not in rendered.codex_prompt


def test_agentic_registries_define_initial_infrastructure() -> None:
    assert set(agentic.AGENT_REGISTRY) == {
        "planner",
        "context-loader",
        "implementer",
        "reviewer",
        "validation-runner",
        "drift-guard",
    }
    assert set(agentic.EXECUTION_PROFILE_REGISTRY) == {
        "prompt_only",
        "review_only",
        "supervised_implementation",
        "bounded_loop_candidate",
    }
    assert "navigation_surface_convergence" not in agentic.SKILL_REGISTRY
    assert "provider_bootstrap_diagnostic" not in agentic.SKILL_REGISTRY
    assert "navigation_surface_convergence" in agentic.PACK_OWNED_PROMPT_SKILL_NAMES
    assert "provider_bootstrap_diagnostic" in agentic.PACK_OWNED_PROMPT_SKILL_NAMES
    assert not any(agent.may_execute_codex for agent in agentic.AGENT_REGISTRY.values())


def test_ecc_discovery_uses_env_path_without_importing_registries(tmp_path: Path, monkeypatch) -> None:
    ecc_repo = _build_ecc_repo(tmp_path / "ECC")
    monkeypatch.setenv(ecc.ECC_REPO_ENV_VAR, str(ecc_repo))

    discovery = ecc.discover_ecc()

    assert discovery.found
    assert discovery.path == ecc_repo.resolve()
    assert discovery.source == f"env:{ecc.ECC_REPO_ENV_VAR}"
    assert tuple(agent.name for agent in discovery.agents) == ("code-reviewer", "planner")
    assert tuple(skill.name for skill in discovery.skills) == ("prompt-optimizer", "verification-loop")
    assert tuple(profile.name for profile in discovery.profiles) == ("developer",)


def test_ecc_discovery_uses_config_path(tmp_path: Path, monkeypatch) -> None:
    ecc_repo = _build_ecc_repo(tmp_path / "ECC")
    config = tmp_path / "ecc.json"
    config.write_text(json.dumps({"path": str(ecc_repo)}), encoding="utf-8")
    monkeypatch.delenv(ecc.ECC_REPO_ENV_VAR, raising=False)
    monkeypatch.setenv(ecc.ECC_CONFIG_ENV_VAR, str(config))

    discovery = ecc.discover_ecc()

    assert discovery.found
    assert discovery.path == ecc_repo.resolve()
    assert discovery.source == f"config:{config}"


def test_cli_ecc_status_reports_discovery_counts(tmp_path: Path, monkeypatch, capsys) -> None:
    ecc_repo = _build_ecc_repo(tmp_path / "ECC")
    monkeypatch.setenv(ecc.ECC_REPO_ENV_VAR, str(ecc_repo))

    assert cli.main(["ecc-status"]) == 0

    output = capsys.readouterr().out
    assert "Protector ECC Status" in output
    assert f"ECC path: {ecc_repo.resolve()}" in output
    assert "ECC discovered: yes" in output
    assert "Relevant agents: 2" in output
    assert "Relevant skills: 2" in output
    assert "Relevant profiles: 1" in output
    assert "Codex execution: disabled" in output
    assert "Protector role: ECC-backed specialization layer" in output
    assert "Protector pack:" in output
    assert "- Discovered: yes" in output
    assert "- Name: protector-financiacioncore" in output
    assert "- Skills: 5" in output
    assert "- Knowledge: 1" in output
    assert "- Validation: valid" in output
    assert "Protector pack capabilities:" in output
    assert "- Declared: 6" in output
    assert "Protector pack capability names:" in output
    assert "- workflow_convergence" in output
    assert "- provider_diagnostics" in output
    assert "- localization_completion" in output
    assert "Protector pack capability warnings:" in output
    assert "Protector pack benchmarks:" in output
    assert "Protector pack prompt skills:" in output
    assert "- Declared: 9" in output
    assert "- Validation: valid" in output
    assert "Protector prompt selection sources:" in output
    assert "- Pack-owned specialization: 9" in output
    assert "- Runtime generic fallback: 5" in output
    assert "- Declared sets: 1" in output
    assert "- Cases: 8" in output
    assert "- Runnable: yes" in output
    assert "- Validation: passing" in output


def test_protector_platform_boundaries_are_explicit() -> None:
    assert agentic.PLATFORM_COMPATIBILITY_STATUS == "deprecated_compatibility_layer"
    assert cli.STABLE_ECC_PACK_COMMANDS == (
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
    assert cli.DEPRECATED_PLATFORM_COMPATIBILITY_COMMANDS == ("plan",)
    assert any("ECC owns reusable agents" in boundary for boundary in agentic.PLATFORM_COMPATIBILITY_BOUNDARIES)
    assert any("Discovery is read-only" in boundary for boundary in ecc.ECC_DISCOVERY_BOUNDARY)


def test_protector_ecc_pack_files_are_discoverable() -> None:
    pack = _protector_pack_dir()
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "protector-financiacioncore"
    assert manifest["runtime_behavior"] == {
        "loaded_by_ph": True,
        "changes_prompt_output": False,
        "codex_execution": False,
        "model_calls": False,
        "autonomous_loops": False,
    }
    assert (pack / "README.md").is_file()
    assert (pack / "knowledge" / "FinanciacionCore.md").read_text(encoding="utf-8") == (
        Path(__file__).resolve().parents[4] / ".protector-harness" / "knowledge" / "FinanciacionCore.md"
    ).read_text(encoding="utf-8")
    assert tuple(capability["name"] for capability in manifest["capabilities"]) == (
        "workflow_convergence",
        "provider_diagnostics",
        "onboarding_convergence",
        "localization_completion",
        "navigation_convergence",
        "form_security",
    )

    skill_paths = tuple(skill["path"] for skill in manifest["skills"])
    benchmark_paths = tuple(benchmark["path"] for benchmark in manifest["verification_assets"]["benchmarks"])
    prompt_skill_path = manifest["verification_assets"]["prompt_skills"]["path"]
    assert skill_paths == (
        "skills/protector-prompt-quality/SKILL.md",
        "skills/protector-codex-handoff/SKILL.md",
        "skills/protector-codex-review/SKILL.md",
        "skills/protector-anti-drift/SKILL.md",
        "skills/financiacioncore-method/SKILL.md",
    )
    assert benchmark_paths == ("../../tests/prompt_benchmarks",)
    assert prompt_skill_path == "verification/prompt-skills.json"
    prompt_skill_metadata = json.loads((pack / prompt_skill_path).read_text(encoding="utf-8"))
    assert prompt_skill_metadata["selection_behavior"] == "pack_owned_runtime_loaded"
    assert tuple(skill["name"] for skill in prompt_skill_metadata["prompt_skills"]) == (
        "form_security_autofill_bug",
        "navigation_surface_convergence",
        "provider_bootstrap_diagnostic",
        "provider_api_bug",
        "operational_workflow_convergence",
        "global_pattern_change",
        "localization_completion",
        "mvp_surface_completion",
        "spanish_implementation_task_preservation",
    )
    for relative in skill_paths:
        text = (pack / relative).read_text(encoding="utf-8")
        assert "origin: Protector ECC pack" in text
        assert text.startswith("---\n")


def test_protector_pack_discovery_validates_static_pack() -> None:
    discovery = ecc.discover_protector_pack()

    assert discovery.found
    assert discovery.name == "protector-financiacioncore"
    assert discovery.skills_count == 5
    assert discovery.knowledge_count == 1
    assert discovery.capabilities == (
        "workflow_convergence",
        "provider_diagnostics",
        "onboarding_convergence",
        "localization_completion",
        "navigation_convergence",
        "form_security",
    )
    assert discovery.capabilities_validation_status == "valid"
    assert discovery.validation_status == "valid"
    assert discovery.prompt_skills_count == 9
    assert discovery.prompt_skills_validation_status == "valid"
    assert discovery.benchmark_sets_count == 1
    assert discovery.benchmark_cases_count == 8
    assert discovery.benchmark_runnable
    assert discovery.benchmark_validation_status == "not_checked"
    assert discovery.warnings == ()
    assert discovery.capability_warnings == ()
    assert discovery.prompt_skill_warnings == ()
    assert discovery.benchmark_warnings == ()


def test_protector_pack_discovery_runs_declared_benchmarks() -> None:
    discovery = ecc.discover_protector_pack(include_benchmarks=True)

    assert discovery.validation_status == "valid"
    assert discovery.benchmark_sets_count == 1
    assert discovery.benchmark_cases_count == 8
    assert discovery.benchmark_runnable
    assert discovery.benchmark_validation_status == "passing"
    assert discovery.benchmark_warnings == ()


def test_protector_pack_manifest_requires_benchmark_declarations() -> None:
    pack = _protector_pack_dir()
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    manifest.pop("verification_assets")

    warnings = ecc._validate_protector_pack_manifest(pack, manifest)

    assert "Protector pack benchmark declaration missing: verification_assets.benchmarks" in warnings


def test_protector_pack_manifest_requires_prompt_skill_metadata() -> None:
    pack = _protector_pack_dir()
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    manifest["verification_assets"].pop("prompt_skills")

    warnings = ecc._validate_protector_pack_manifest(pack, manifest)

    assert "Protector pack prompt skill metadata declaration missing: verification_assets.prompt_skills.path" in warnings


def test_protector_pack_prompt_skill_metadata_requires_runtime_skill_and_benchmark_coverage(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    benchmarks = tmp_path / "benchmarks"
    metadata = root / "verification" / "prompt-skills.json"
    _write_prompt_benchmark(
        benchmarks,
        "covered-case",
        "Fix scoped issue",
        """# Expected Characteristics

## Required skills
- implementation_fix
""",
    )
    _write(
        metadata,
        json.dumps(
            {
                "schema_version": "protector-pack-prompt-skills-v1",
                "selection_behavior": "pack_owned_runtime_loaded",
                "changes_prompt_output": False,
                "prompt_skills": [
                    _pack_prompt_skill_metadata_item("not_a_runtime_skill", ["covered-case"]),
                    _pack_prompt_skill_metadata_item("form_security_autofill_bug", ["covered-case"]),
                ],
            }
        ),
    )
    manifest = {
        "verification_assets": {
            "benchmarks": [
                {
                    "name": "fixture",
                    "runner": "ph_benchmark",
                    "path": str(benchmarks),
                    "expected_cases": 1,
                }
            ],
            "prompt_skills": {"path": "verification/prompt-skills.json"},
        }
    }

    warnings = ecc._validate_prompt_skill_metadata(root, manifest)

    assert "Protector pack prompt skill is not defined by current runtime: not_a_runtime_skill" in warnings
    assert "Protector pack prompt skill is not covered by any benchmark case: form_security_autofill_bug" in warnings
    assert "Protector pack prompt skill benchmark case does not require form_security_autofill_bug: covered-case" in warnings


def test_protector_pack_capabilities_require_pack_skill_and_benchmark_coverage(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    benchmarks = tmp_path / "benchmarks"
    metadata = root / "verification" / "prompt-skills.json"
    _write_prompt_benchmark(
        benchmarks,
        "covered-case",
        "Fix scoped form security issue",
        """# Expected Characteristics

## Required skills
- form_security_autofill_bug
""",
    )
    _write(
        metadata,
        json.dumps(
            {
                "schema_version": "protector-pack-prompt-skills-v1",
                "selection_behavior": "pack_owned_runtime_loaded",
                "changes_prompt_output": False,
                "prompt_skills": [
                    _pack_prompt_skill_metadata_item("form_security_autofill_bug", ["covered-case"]),
                    _pack_prompt_skill_metadata_item("provider_api_bug", ["covered-case"]),
                ],
            }
        ),
    )
    manifest = {
        "capabilities": [
            {"name": "form_security", "prompt_skills": ["form_security_autofill_bug"]},
            {"name": "orphan_unknown", "prompt_skills": ["missing_pack_skill"]},
            {"name": "orphan_uncovered", "prompt_skills": ["provider_api_bug"]},
        ],
        "verification_assets": {
            "benchmarks": [
                {
                    "name": "fixture",
                    "runner": "ph_benchmark",
                    "path": str(benchmarks),
                    "expected_cases": 1,
                }
            ],
            "prompt_skills": {"path": "verification/prompt-skills.json"},
        },
    }

    warnings = ecc._validate_pack_capabilities(root, manifest)

    assert "Protector pack capability references unknown pack skill: orphan_unknown -> missing_pack_skill" in warnings
    assert "Protector pack capability is not backed by any pack skill: orphan_unknown" in warnings
    assert "Protector pack capability is not covered by any benchmark case: orphan_uncovered" in warnings
    assert not any("form_security" in warning for warning in warnings)


def test_sample_task_produces_controlled_execution_plan(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    )

    assert rendered.profile == "supervised_implementation"
    assert rendered.agents == (
        "planner",
        "context-loader",
        "implementer",
        "reviewer",
        "validation-runner",
        "drift-guard",
    )
    assert "navigation_surface_convergence" in rendered.skills
    assert "implementation_fix" in rendered.skills
    assert "Codex execution: disabled" in rendered.text
    assert "Autonomous code execution: disabled" in rendered.text
    assert "Reviewer chain:" in rendered.text
    assert "Drift-guard must block broad audits and unrelated architecture." in rendered.text
    assert "Knowledge gates:" in rendered.text
    assert "contract-first company and financer onboarding" in rendered.text
    assert "ECC Supervised Automation Pilot:" in rendered.text
    assert "Concept owner: ECC" in rendered.text
    assert "Pilot mode: read-only planning" in rendered.text
    assert "Selected ECC pack:" in rendered.text
    assert "- Name: protector-financiacioncore" in rendered.text
    assert "Selected skills:" in rendered.text
    assert "navigation_surface_convergence [pack] - covered by legacy-onboarding-path-convergence, navigation-convergence" in rendered.text
    assert "implementation_fix [runtime_generic] - generic fallback; no pack benchmark required" in rendered.text
    assert "Knowledge used:" in rendered.text
    assert "Knowledge path:" in rendered.text
    assert "Benchmark confidence:" in rendered.text
    assert "Confidence: high: verified pack and selected pack skills are benchmark-covered" in rendered.text
    assert "Proposed Codex prompt:" in rendered.text
    assert "Codex Prompt:" in rendered.text
    assert "Review criteria:" in rendered.text
    assert "Blockers or missing coverage:" in rendered.text
    assert "None for read-only supervised planning." in rendered.text
    assert "ECC Review Contract:" in rendered.text
    assert "Contract owner: ECC" in rendered.text
    assert "Review execution: manual/supervised only" in rendered.text
    assert "Git diff inspection: not automatic" in rendered.text
    assert "Proposed Codex task:" in rendered.text
    assert "Expected implementation areas:" in rendered.text
    assert "Route, menu, index, dashboard, or promoted operator-view entry points." in rendered.text
    assert "Expected files likely to change:" in rendered.text
    assert "Expected validation scope:" in rendered.text
    assert "Validation Decision Record:" in rendered.text
    assert "Global Validation Law: Choose the cheapest credible falsifier first" in rendered.text
    assert "VDR uncertainty:" in rendered.text
    assert "VDR cheapest_falsifier:" in rendered.text
    assert "VDR escalation_reason:" in rendered.text
    assert "Benchmark relevance:" in rendered.text
    assert "navigation_surface_convergence: legacy-onboarding-path-convergence, navigation-convergence" in rendered.text
    assert "Review risks:" in rendered.text
    assert "Generic Protector fallback is present; verify it did not overpower pack-specific specialization." in rendered.text
    assert "Anti-drift checks:" in rendered.text
    assert "Reject dashboard/session/workflow-engine/agent-loop additions unless the task explicitly requested them." in rendered.text
    assert "PASS/FAIL criteria:" in rendered.text
    assert "PASS only if the Codex output addresses the proposed task and respects every selected pack-skill restriction." in rendered.text


def test_expected_files_do_not_use_selected_guidance_context(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix public payment workflow contract button Razor CSS visual alignment",
        repo=_build_repo(tmp_path / "repo"),
    )
    candidate = rendered.automation_candidate
    review_contract = candidate["review_contract"]
    selected_context_paths = candidate["selected_context_paths"]

    assert ".codex/agent-workflow/feature-contract-template.md" in selected_context_paths
    assert "Selected context paths:" in rendered.text
    assert "- .codex/agent-workflow/feature-contract-template.md" in rendered.text
    assert review_contract["expected_files_likely_to_change"] == ("Unknown until code inspection",)
    assert "Expected files likely to change:\n- Unknown until code inspection" in rendered.text
    expected_files_section = rendered.text.split("Expected files likely to change:", maxsplit=1)[1].split("Expected validation scope:", maxsplit=1)[0]
    assert ".codex/agent-workflow/feature-contract-template.md" not in expected_files_section


def test_task_refinement_normalizes_rough_ui_task_for_candidate_generation(tmp_path: Path) -> None:
    raw_task = "Change the delete action icon on Clients/New Client table of receipts, all other views use taildwind"

    rendered = engineering.render_controlled_execution_plan(
        task=raw_task,
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
    )
    candidate = rendered.automation_candidate
    intake = candidate["task"]["intake_refinement"]

    assert "Task Intake Refinement:" in rendered.text
    assert "Status: ready" in rendered.text
    assert intake["status"] == "ready"
    assert intake["target_surface"] == "Clients/NewClient receipts table"
    assert intake["requested_change"] == "Update only the delete action icon/button styling to match the existing Tailwind-style delete actions."
    assert candidate["task"]["raw_operator_task"] == raw_task
    assert candidate["task"]["summary"].startswith("Update only the delete action icon/button styling")
    assert "Update only the delete action icon/button styling" in candidate["proposed_codex_prompt"]
    assert "Preserve routes, handlers, forms, table data, delete behavior, and all non-delete UI." not in candidate["proposed_codex_prompt"]
    assert "Do not change delete behavior." in candidate["proposed_codex_prompt"]
    assert "Unknown until code inspection" in candidate["review_contract"]["expected_files_likely_to_change"]


def test_task_refinement_ambiguous_task_needs_clarification(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix it",
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
    )
    readiness = rendered.automation_candidate["automation_readiness"]
    intake = rendered.automation_candidate["task"]["intake_refinement"]

    assert intake["status"] == "needs_clarification"
    assert "Which target surface, route, view, table, or component should change?" in intake["missing_details"]
    assert readiness["classification"] == "supervised_only"
    assert "Task intake needs clarification:" in rendered.text
    assert "Which target surface, route, view, table, or component should change?" in rendered.text


def test_cli_plan_refine_task_preserves_evidence_references(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    evidence = tmp_path / "screenshot.png"
    evidence.write_text("IMAGE CONTENT SHOULD NOT BE READ", encoding="utf-8")
    output = tmp_path / "candidate.json"

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--refine-task",
                "--evidence",
                str(evidence),
                "--candidate-json",
                str(output),
                "Change",
                "the",
                "delete",
                "action",
                "icon",
                "on",
                "Clients/New",
                "Client",
                "table",
                "of",
                "receipts",
                "all",
                "other",
                "views",
                "use",
                "taildwind",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    evidence_path = str(evidence)
    intake = payload["task"]["intake_refinement"]
    contract = payload["review_contract"]

    assert evidence_path in intake["evidence_paths"]
    assert evidence_path in payload["task"]["evidence_paths"]
    assert evidence_path in contract["evidence_references"]
    assert "Evidence references:" in stdout
    assert evidence_path in stdout
    assert "Evidence paths are references only; no OCR, image analysis, or file parsing was performed." in stdout
    assert "Codex/reviewer must consider these references when validating observed UI or file state." in contract["evidence_handling"]
    assert "IMAGE CONTENT SHOULD NOT BE READ" not in stdout
    assert "IMAGE CONTENT SHOULD NOT BE READ" not in output.read_text(encoding="utf-8")
    assert "Evidence references are available but not analyzed" in payload["proposed_codex_prompt"]
    assert payload["task"]["evidence_notes"] == []
    assert "Evidence paths are attached but not interpreted." in contract["evidence_handling"]


def test_evidence_note_improves_normalized_task_and_candidate(tmp_path: Path) -> None:
    note = (
        "Screenshot shows Clients/NewClient receipts table. Delete action is still a red rounded legacy button; "
        "expected style is the Tailwind trash icon used elsewhere."
    )
    rendered = engineering.render_controlled_execution_plan(
        task="Change delete icon, other views use tailwind",
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
        evidence_notes=(note,),
    )
    payload = rendered.automation_candidate
    intake = payload["task"]["intake_refinement"]
    contract = payload["review_contract"]

    assert intake["status"] == "ready"
    assert intake["target_surface"] == "Clients/NewClient receipts table"
    assert "Tailwind trash/delete action used by other promoted views" in intake["requested_change"]
    assert "Delete action is still a red rounded legacy button" in intake["observed_state"]
    assert "expected style is the Tailwind trash icon used elsewhere" in intake["expected_state"]
    assert "Clients/NewClient receipts table" in intake["normalized_task"]
    assert payload["task"]["evidence_notes"] == (note,)
    assert contract["operator_evidence_notes"] == (note,)
    assert "Treat evidence notes as operator-provided observations, not inferred facts." in contract["evidence_handling"]
    assert "Operator evidence notes:" in rendered.text
    assert "Operator evidence note observation:" in payload["proposed_codex_prompt"]


def test_cli_plan_preserves_evidence_notes_in_candidate_and_guide(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    output = tmp_path / "candidate.json"
    note = (
        "Screenshot shows Clients/NewClient receipts table. Delete action is still a red rounded legacy button; "
        "expected style is the Tailwind trash icon used elsewhere."
    )

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--refine-task",
                "--evidence-note",
                note,
                "--candidate-json",
                str(output),
                "Change",
                "delete",
                "icon,",
                "other",
                "views",
                "use",
                "tailwind",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["task"]["evidence_notes"] == [note]
    assert payload["review_contract"]["operator_evidence_notes"] == [note]
    assert "--evidence-note" in stdout
    assert "Operator evidence notes:" in stdout
    assert note in stdout


def test_candidate_dry_run_renders_evidence_references(tmp_path: Path) -> None:
    evidence = tmp_path / "screenshot.png"
    evidence.write_text("DO NOT INVENT THIS CONTENT", encoding="utf-8")
    candidate = engineering.render_controlled_execution_plan(
        task="Change the delete action icon on Clients/New Client table of receipts, all other views use taildwind",
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
        evidence_paths=(str(evidence),),
    ).automation_candidate

    rendered = engineering.render_automation_candidate_dry_run(candidate, source="candidate.json")

    assert rendered.valid
    assert "Evidence references:" in rendered.text
    assert str(evidence) in rendered.text
    assert "Operator evidence notes:\n- (none)" in rendered.text
    assert "Evidence handling: Treat evidence paths as operator-provided references only." in rendered.text
    assert "Evidence paths are attached but not interpreted." in rendered.text
    assert "DO NOT INVENT THIS CONTENT" not in rendered.text


def test_candidate_dry_run_renders_operator_evidence_notes(tmp_path: Path) -> None:
    note = "Screenshot shows Clients/NewClient receipts table; expected style is the Tailwind trash icon used elsewhere."
    candidate = engineering.render_controlled_execution_plan(
        task="Change delete icon, other views use tailwind",
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
        evidence_notes=(note,),
    ).automation_candidate

    rendered = engineering.render_automation_candidate_dry_run(candidate, source="candidate.json")

    assert rendered.valid
    assert "Operator evidence notes:" in rendered.text
    assert note in rendered.text
    assert "Operator evidence notes: " in rendered.text
    assert "Treat evidence notes as operator-provided observations, not inferred facts." in rendered.text


def test_candidate_outcome_includes_evidence_references(tmp_path: Path) -> None:
    evidence = tmp_path / "screenshot.png"
    evidence.write_text("UNREAD IMAGE CONTENT", encoding="utf-8")
    note = "Screenshot shows Clients/NewClient receipts table; expected style is the Tailwind trash icon used elsewhere."
    candidate = engineering.render_controlled_execution_plan(
        task="Change the delete action icon on Clients/New Client table of receipts, all other views use taildwind",
        repo=_build_repo(tmp_path / "repo"),
        refine_task=True,
        evidence_paths=(str(evidence),),
        evidence_notes=(note,),
    ).automation_candidate

    report = engineering.render_candidate_outcome_report(
        candidate=candidate,
        candidate_source="candidate.json",
        codex_output="""Files read
- AGENTS.md

Files changed
- Views/Clients/NewClient.cshtml

Summary
- Updated delete action icon styling only.

Validation
- Build check passed and focused UI smoke check passed.

PASS
""",
        codex_output_source="codex-output.txt",
    )

    assert "Evidence references:\n-" in report.text
    assert str(evidence) in report.text
    assert "Operator evidence notes:\n-" in report.text
    assert note in report.text
    assert "UNREAD IMAGE CONTENT" not in report.text


def test_supervised_outcome_report_accepts_plan_consistent_codex_output(tmp_path: Path) -> None:
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        source="codex-output.txt",
    )

    assert report.status == "accepted"
    assert report.follow_up is None
    assert "ECC Supervised Outcome Report" in report.text
    assert "Status: accepted" in report.text
    assert "Selected skills:" in report.text
    assert "navigation_surface_convergence [pack] - covered by legacy-onboarding-path-convergence, navigation-convergence" in report.text
    assert "Knowledge used:" in report.text
    assert "Actual files changed:\n- Views/Operator/Index.cshtml" in report.text
    assert "Deviations from plan:\n- (none)" in report.text
    assert "Validation gaps:\n- (none)" in report.text
    assert "PASS/FAIL consistency:\n- (none)" in report.text
    assert "Follow-up prompt:\n- (none)" in report.text
    assert "Codex execution was not invoked." in report.text
    assert "Git diffs were not inspected automatically." in report.text


def test_supervised_outcome_report_flags_drift_and_validation_gaps(tmp_path: Path) -> None:
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        codex_output="""Summary
- Added dashboard memory graph governance changes.

Validation
- not run

PASS
""",
        source="codex-output.txt",
    )

    assert report.status == "needs review"
    assert report.follow_up is not None
    assert "Status: needs review" in report.text
    assert "Codex output did not report changed files." in report.text
    assert "dashboard mentioned without task scope" in report.text
    assert "memory mentioned without task scope" in report.text
    assert "Codex claimed PASS but deterministic review found unresolved gaps." in report.text
    assert "Reported validation lacks build/test/smoke/check evidence." in report.text
    assert "Follow-up prompt:\n- Revise or justify the Codex result" in report.text


def test_supervised_outcome_report_flags_validation_escalation_without_vdr(tmp_path: Path) -> None:
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build check passed, UI workflow route smoke check passed, then ran end-to-end integration validation against an external service.

PASS
""",
        source="codex-output.txt",
    )

    assert report.status == "needs review"
    assert "Validation escalated without VDR evidence: missing uncertainty, cheapest_falsifier, escalation_reason." in report.text
    assert "Codex claimed PASS but deterministic review found unresolved gaps." in report.text


def test_supervised_outcome_report_marks_reported_fail_as_failed(tmp_path: Path) -> None:
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Blocked before changing navigation route menu view operator onboarding convergence.

Validation
- Build not run because the task was blocked.

FAIL
""",
        source="codex-output.txt",
    )

    assert report.status == "failed"
    assert "Status: failed" in report.text
    assert "Codex reported FAIL; outcome cannot be accepted." in report.text
    assert "Follow-up prompt:\n- Revise or justify the Codex result" in report.text


def test_outcome_history_persists_structured_summary_without_raw_codex_output(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = """Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
"""
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        codex_output=codex_output,
        source="codex-output.txt",
    )

    path = engineering.append_outcome_history(repo, report)
    payload = json.loads(path.read_text(encoding="utf-8").strip())

    assert path == repo / ".protector-harness" / "outcome-history.jsonl"
    assert payload["selected_pack"] == "protector-financiacioncore"
    assert payload["status"] == "accepted"
    assert payload["changed_files"] == ["Views/Operator/Index.cshtml"]
    assert payload["executed_validations"] == ["Build verified and UI workflow route smoke test check passed."]
    assert payload["deviations"] == []
    assert payload["follow_up_prompt"] is None
    assert payload["suggested_benchmark_additions"] == []
    assert {"name": "navigation_surface_convergence", "source": "pack"} in payload["selected_skills"]
    assert "codex_output" not in payload
    assert "Files read" not in path.read_text(encoding="utf-8")


def test_plan_can_surface_recent_relevant_outcome_signals(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        source="codex-output.txt",
    )
    engineering.append_outcome_history(repo, report)

    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        include_history=True,
    )

    assert "Recent outcome signals:" in rendered.text
    assert "accepted | skills=base_prompt_quality, implementation_fix, navigation_surface_convergence" in rendered.text
    assert "files=Views/Operator/Index.cshtml" in rendered.text


def test_automation_readiness_marks_verified_covered_plan_ready(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    )

    assert "Automation Readiness:" in rendered.text
    assert "Classification: automation_ready" in rendered.text
    assert "Pack benchmark validation: passing." in rendered.text
    assert "Readiness Decision Record:" in rendered.text
    assert "Selected pack skills have benchmark coverage: navigation_surface_convergence." in rendered.text
    assert "Selected knowledge provides 6 compact fact(s)." in rendered.text
    assert "No negative learning signals were found." in rendered.text
    assert "Remaining uncertainty: (none)" in rendered.text
    assert "Decision: automation_ready wins because required evidence is present" in rendered.text
    assert "automation_ready means candidate-backed execution after explicit approval, not autonomous execution." in rendered.text
    assert "proceed with candidate-backed execution only after explicit human approval" in rendered.text


def test_automation_readiness_supervised_only_has_concrete_uncertainty(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Review payment implementation notes",
        repo=_build_repo(tmp_path / "repo"),
    )
    record = rendered.automation_candidate["automation_readiness"]["readiness_decision_record"]

    assert rendered.automation_candidate["automation_readiness"]["classification"] == "supervised_only"
    assert "Remaining uncertainty: Implementation intent is unresolved because the task asks for review output only." in rendered.text
    assert "Supervision rationale: Review-only task has no approved implementation intent for candidate execution." in rendered.text
    assert "Decision: supervised_only wins because concrete uncertainty remains" in rendered.text
    assert record["remaining_uncertainty"] == ("Implementation intent is unresolved because the task asks for review output only.",)
    assert record["supervision_rationale"] == ("Review-only task has no approved implementation intent for candidate execution.",)


def test_automation_readiness_allows_generic_only_narrow_visual_task(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix Razor CSS visual alignment",
        repo=_build_repo(tmp_path / "repo"),
    )
    selected_skills = rendered.automation_candidate["selected_skills"]
    record = rendered.automation_candidate["automation_readiness"]["readiness_decision_record"]

    assert all(skill["source"] == "runtime_generic" for skill in selected_skills)
    assert rendered.automation_candidate["automation_readiness"]["classification"] == "automation_ready"
    assert (
        "No pack specialization was selected; generic runtime skills are acceptable because no pack-specific evidence is required."
        in rendered.text
    )
    assert "automation_ready wins because required evidence is present" in record["decision"]


def test_execution_plan_exposes_structured_automation_candidate(tmp_path: Path) -> None:
    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
        include_history=True,
    )
    candidate = rendered.automation_candidate

    assert candidate["schema_version"] == "ecc-automation-candidate-v1"
    assert candidate["selected_pack"]["name"] == "protector-financiacioncore"
    assert candidate["selected_pack"]["validation_status"] == "valid"
    assert candidate["task"]["summary"] == "Fix legacy onboarding path convergence for operator UI views"
    assert candidate["task"]["mode"] == "ui_runtime_bug"
    assert candidate["task"]["raw_operator_task"] == "Fix legacy onboarding path convergence for operator UI views"
    assert "AGENTS.md" in candidate["selected_context_paths"]
    assert "contract-first company and financer onboarding" in " ".join(candidate["selected_knowledge"]["facts"])
    assert {"name": "navigation_surface_convergence", "source": "pack"} in candidate["selected_skills"]
    assert candidate["benchmark_coverage"]["navigation_surface_convergence"] == (
        "legacy-onboarding-path-convergence",
        "navigation-convergence",
    )
    review_contract = candidate["review_contract"]
    assert "Route, menu, index, dashboard, or promoted operator-view entry points." in review_contract["expected_implementation_areas"]
    assert "Verify the affected navigation entries land on the intended operational view." in review_contract["expected_validation_scope"]
    assert review_contract["validation_decision_record"]["global_validation_law"].startswith("Choose the cheapest credible falsifier first")
    assert set(review_contract["validation_decision_record"]) == {
        "global_validation_law",
        "uncertainty",
        "cheapest_falsifier",
        "escalation_reason",
    }
    assert candidate["automation_readiness"]["classification"] == "automation_ready"
    assert "Selected pack skills have benchmark coverage: navigation_surface_convergence." in candidate["automation_readiness"]["reasons"]
    readiness_record = candidate["automation_readiness"]["readiness_decision_record"]
    assert readiness_record["decision"].startswith("automation_ready wins because required evidence is present")
    assert readiness_record["remaining_uncertainty"] == ()
    assert "Pack, prompt-skill, capability, and benchmark validation are passing." in readiness_record["automation_rationale"]
    assert candidate["proposed_codex_prompt"].startswith("Codex Prompt:")
    assert candidate["execution_boundaries"] == {
        "codex_execution": False,
        "model_calls": False,
        "autonomous_loops": False,
        "workflow_engine": False,
    }


def test_automation_candidate_import_dry_run_validates_current_evidence(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate

    rendered = engineering.render_automation_candidate_dry_run(candidate, source="candidate.json")

    assert rendered.valid
    assert rendered.decision == "would execute"
    assert rendered.validation_errors == ()
    assert rendered.text.startswith("ECC Candidate Import Dry Run\n")
    assert "Schema validation: PASS" in rendered.text
    assert "Decision: would execute" in rendered.text
    assert "- Selected pack still valid: yes" in rendered.text
    assert "- Selected skills still available: yes" in rendered.text
    assert "- Benchmark coverage still valid: yes" in rendered.text
    assert "- Readiness classification explainable: yes" in rendered.text
    assert "Codex prompt preview:\nCodex Prompt:" in rendered.text
    assert "Review contract summary:" in rendered.text
    assert "Validation Decision Record: law=Choose the cheapest credible falsifier first" in rendered.text
    assert "Readiness Decision Record:" in rendered.text
    assert "automation_ready wins because required evidence is present" in rendered.text
    assert "Safety boundaries:" in rendered.text
    assert "- Codex execution: disabled" in rendered.text
    assert "- File edits: disabled by dry-run importer" in rendered.text


def test_automation_candidate_import_dry_run_blocks_stale_skill(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate["selected_skills"] = (*candidate["selected_skills"], {"name": "removed_pack_skill", "source": "pack"})

    rendered = engineering.render_automation_candidate_dry_run(candidate, source="candidate.json")

    assert not rendered.valid
    assert rendered.decision == "blocked"
    assert "Selected skill unavailable: removed_pack_skill [pack]" in rendered.validation_errors
    assert "Schema validation: FAIL" in rendered.text
    assert "- Selected skills still available: no" in rendered.text


def test_candidate_outcome_report_accepts_candidate_consistent_codex_output(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate

    report = engineering.render_candidate_outcome_report(
        candidate=candidate,
        candidate_source="candidate.json",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        codex_output_source="codex-output.txt",
    )

    assert report.status == "accepted"
    assert report.follow_up is None
    assert "ECC Supervised Outcome Report" in report.text
    assert "Status: accepted" in report.text
    assert "Candidate source:\n- candidate.json" in report.text
    assert "Automation readiness:\n- Classification: automation_ready" in report.text
    assert "Knowledge used:" in report.text
    assert "contract-first company and financer onboarding" in report.text
    assert "Actual files changed:\n- Views/Operator/Index.cshtml" in report.text
    assert "Deviations from plan:\n- (none)" in report.text
    assert "Validation gaps:\n- (none)" in report.text
    assert "PASS/FAIL consistency:\n- (none)" in report.text
    assert report.summary.selected_pack == "protector-financiacioncore"
    assert {"name": "navigation_surface_convergence", "source": "pack"} in [
        {"name": skill.name, "source": skill.source} for skill in report.summary.selected_skills
    ]


def test_candidate_outcome_report_rejects_invalid_candidate_before_review(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate["execution_boundaries"] = {**candidate["execution_boundaries"], "codex_execution": True}

    with pytest.raises(engineering.HarnessUsageError, match="candidate validation failed"):
        engineering.render_candidate_outcome_report(
            candidate=candidate,
            candidate_source="candidate.json",
            codex_output="PASS",
            codex_output_source="codex-output.txt",
        )


def test_candidate_outcome_report_flags_drift_and_validation_gaps(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate

    report = engineering.render_candidate_outcome_report(
        candidate=candidate,
        candidate_source="candidate.json",
        codex_output="""Summary
- Added dashboard memory graph governance changes.

Validation
- not run

PASS
""",
        codex_output_source="codex-output.txt",
    )

    assert report.status == "needs review"
    assert report.follow_up is not None
    assert "Codex output did not report changed files." in report.text
    assert "dashboard mentioned without task scope" in report.text
    assert "memory mentioned without task scope" in report.text
    assert "Codex claimed PASS but deterministic review found unresolved gaps." in report.text
    assert "Reported validation lacks build/test/smoke/check evidence." in report.text


def test_candidate_outcome_report_flags_validation_escalation_without_vdr(tmp_path: Path) -> None:
    candidate = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    ).automation_candidate

    report = engineering.render_candidate_outcome_report(
        candidate=candidate,
        candidate_source="candidate.json",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build check passed, UI workflow route smoke check passed, then ran end-to-end integration validation against an external service.

PASS
""",
        codex_output_source="codex-output.txt",
    )

    assert report.status == "needs review"
    assert "Validation escalated without VDR evidence: missing uncertainty, cheapest_falsifier, escalation_reason." in report.text
    assert "Codex claimed PASS but deterministic review found unresolved gaps." in report.text


def test_outcome_learning_signals_derive_recurring_patterns(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")
    for _ in range(2):
        _append_outcome_summary(
            repo,
            validations=("not run",),
            deviations=(
                "Changed file outside expected plan context: Views/Unexpected.cshtml",
                "dashboard mentioned without task scope",
                "PASS claimed without validation evidence",
            ),
            follow_up="Revise or justify the Codex result for: Fix legacy onboarding path convergence.",
            benchmark_additions=("Add prompt benchmark coverage for uncovered pack skill: localization_completion.",),
        )

    rendered = engineering.render_outcome_learning_signals(
        repo,
        task="Fix legacy onboarding path convergence for operator UI views",
    )

    assert "ECC Outcome Learning Signals" in rendered
    assert "Scope: selected skills: base_prompt_quality, navigation_surface_convergence, implementation_fix" in rendered
    assert "Recurring failed validation: not run (2 outcomes)." in rendered
    assert "Recurring changed-file mismatch: Changed file outside expected plan context: Views/Unexpected.cshtml (2 outcomes)." in rendered
    assert "Skill repeatedly lacks benchmark coverage: localization_completion (2 outcomes)." in rendered
    assert "Repeated follow-up prompt: Revise or justify the Codex result for: Fix legacy onboarding path convergence. (2 outcomes)." in rendered
    assert "Repeated drift/deviation pattern: dashboard mentioned without task scope (2 outcomes)." in rendered


def test_plan_with_history_surfaces_recurring_learning_signals(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")
    for _ in range(2):
        _append_outcome_summary(
            repo,
            validations=("not run",),
            deviations=("dashboard mentioned without task scope",),
            follow_up="Revise or justify the Codex result for: Fix legacy onboarding path convergence.",
        )

    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        include_history=True,
    )

    assert "Outcome learning signals:" in rendered.text
    assert "Recurring failed validation: not run (2 outcomes)." in rendered.text
    assert "Repeated drift/deviation pattern: dashboard mentioned without task scope (2 outcomes)." in rendered.text
    assert "Recommendation: state the anti-drift constraint explicitly before Codex runs." in rendered.text
    assert "Automation Readiness:" in rendered.text
    assert "Classification: supervised_only" in rendered.text
    assert "Negative learning signal must be resolved before automation_ready: Recurring failed validation: not run (2 outcomes)." in rendered.text
    assert (
        "Remaining uncertainty: Historical failure must be falsified by a clean supervised outcome: "
        "Recurring failed validation: not run (2 outcomes)." in rendered.text
    )
    assert "Recommended next step:\n- resolve RDR uncertainty before automation_ready:" in rendered.text


def test_plan_with_history_applies_explainable_adaptive_adjustments(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")
    for _ in range(2):
        _append_outcome_summary(
            repo,
            validations=("not run",),
            deviations=(
                "Changed file outside expected plan context: Views/Unexpected.cshtml",
                "dashboard mentioned without task scope",
                "PASS claimed without validation evidence",
            ),
            follow_up="Revise or justify the Codex result for: Fix legacy onboarding path convergence.",
            benchmark_additions=("Add prompt benchmark coverage for uncovered pack skill: localization_completion.",),
        )

    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        include_history=True,
    )

    assert "Adaptive planning adjustments:" in rendered.text
    assert "Strengthen validation expectations: require explicit build/test/smoke evidence" in rendered.text
    assert "Increase benchmark emphasis: call out benchmark coverage risk" in rendered.text
    assert "Increase anti-drift guidance: restate the exact task boundary" in rendered.text
    assert "Surface likely changed-file area: Views/Unexpected.cshtml." in rendered.text
    assert "Pre-apply repeated follow-up: include the recurring correction" in rendered.text
    assert "Triggered by learning signal: Recurring failed validation: not run (2 outcomes)." in rendered.text
    assert "Triggered by learning signal: Skill repeatedly lacks benchmark coverage: localization_completion (2 outcomes)." in rendered.text
    assert (
        "Triggered by learning signal: Recurring changed-file mismatch: Changed file outside expected plan context: "
        "Views/Unexpected.cshtml (2 outcomes)." in rendered.text
    )


def test_automation_readiness_blocks_missing_pack_skill_coverage(tmp_path: Path, monkeypatch) -> None:
    def no_coverage() -> dict[str, tuple[str, ...]]:
        return {}

    monkeypatch.setattr(engineering, "discover_pack_prompt_skill_benchmark_coverage", no_coverage)

    rendered = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=_build_repo(tmp_path / "repo"),
        repo_alias="FinanciacionCore",
    )

    assert "Automation Readiness:" in rendered.text
    assert "Classification: blocked" in rendered.text
    assert "Selected pack skills missing benchmark coverage: navigation_surface_convergence." in rendered.text
    assert "Blocking rationale: Selected pack skills missing benchmark coverage: navigation_surface_convergence." in rendered.text
    assert "Decision: blocked wins because required evidence is missing" in rendered.text
    assert (
        "Recommended next step:\n- restore missing readiness evidence first: "
        "Selected pack skills missing benchmark coverage: navigation_surface_convergence." in rendered.text
    )


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


def test_default_prompt_omits_legacy_files_output_sections(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Fix regression: missing signature button")

    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt
    assert "Mandatory output:\n- Summary\n- Validation\n- PASS/FAIL" in rendered.codex_prompt


def test_prompt_includes_legacy_files_output_sections_when_requested(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), "Fix regression and include Files read and Files changed in the output")

    assert "Mandatory output:\n- Files read\n- Files changed\n- Summary\n- Validation\n- PASS/FAIL" in rendered.codex_prompt


def test_task_mode_output_requirements_are_minimal(tmp_path: Path) -> None:
    planning = _render(_build_repo(tmp_path / "planning"), "Plan roadmap for payment workflow")
    diagnostic = _render(_build_repo(tmp_path / "diagnostic"), "Diagnostic bootstrap: smoke real provider")

    assert "Mandatory output:\n- Plan\n- Risks\n- Validation\n- PASS/FAIL" in planning.codex_prompt
    assert "- Files read" not in planning.codex_prompt
    assert "- Files changed" not in planning.codex_prompt
    assert "Mandatory output:\n- Diagnostics run\n- Evidence\n- Validation\n- PASS/FAIL" in diagnostic.codex_prompt
    assert "- Files read" not in diagnostic.codex_prompt
    assert "- Files changed" not in diagnostic.codex_prompt


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


def test_spanish_ui_implementation_request_is_actionable_and_unicode_safe(tmp_path: Path) -> None:
    task = (
        "Eliminemos la opción de añadir más de una cuenta desde FormNewBankDataId. "
        "Quitar el botón Añadir de la tabla de cuentas en las vistas de creación."
    )

    rendered = _render(_build_repo(tmp_path), task)

    assert "Title: Remove the option to add more than one" in rendered.codex_prompt
    assert "Task mode: ui_runtime_bug" in rendered.codex_prompt
    assert "Original user task:" not in rendered.codex_prompt
    assert task not in rendered.codex_prompt
    assert "Objective: Remove the option to add more than one account" in rendered.codex_prompt
    assert "Task details:\nObserved state:" in rendered.codex_prompt
    assert "Expected behavior:" in rendered.codex_prompt
    assert "Remove the option to add more than one account" in rendered.codex_prompt
    assert "Objective: Eliminemos" not in rendered.codex_prompt
    assert "Task details:\nEliminemos" not in rendered.codex_prompt
    assert "Do not edit files unless explicitly requested by this task." not in rendered.codex_prompt
    assert "Do not edit files unless the caller explicitly grants editing" not in rendered.codex_prompt
    assert "Inspect only enough code to locate the faulty condition, then fix surgically." in rendered.codex_prompt
    assert "Scoped edits are allowed when needed to fix the requested UI/runtime issue." in rendered.codex_prompt
    assert "Task details:\n(none)" not in rendered.codex_prompt
    assert "FormNewBankDataId" in rendered.codex_prompt
    assert "botón Añadir" in rendered.codex_prompt
    assert "tabla de cuentas" in rendered.codex_prompt
    assert "vistas de creación" in rendered.codex_prompt
    for text in ("botón Añadir", "tabla de cuentas", "vistas de creación"):
        assert text in rendered.codex_prompt
    for mojibake in ("opci¾n", "a±adir", "mßs"):
        assert mojibake not in rendered.codex_prompt


def test_spanish_review_only_request_remains_read_only(tmp_path: Path) -> None:
    task = "Revisar sin implementar la opción de añadir más de una cuenta en las vistas de creación."

    rendered = _render(_build_repo(tmp_path), task)

    assert "Original user task:" not in rendered.codex_prompt
    assert "Revisar sin implementar" not in rendered.codex_prompt
    assert "Title: Review the option to add more than one" in rendered.codex_prompt
    assert "Task mode: review_only" in rendered.codex_prompt
    assert "Objective: Review the option to add more than one account without implementing changes" in rendered.codex_prompt
    assert "Objective: Revisar" not in rendered.codex_prompt
    assert "Observed state:" in rendered.codex_prompt
    assert "Do not edit files unless explicitly requested by this task." in rendered.codex_prompt
    assert "start with read-only inspection of the selected context" in rendered.codex_prompt
    assert "Scoped edits are allowed" not in rendered.codex_prompt


def test_english_task_uses_structured_prompt_without_original_task_section(tmp_path: Path) -> None:
    task = "Fix regression: missing signature button"

    rendered = _render(_build_repo(tmp_path), task)

    assert "Original user task:" not in rendered.codex_prompt
    assert "Title: Fix regression missing signature button" in rendered.codex_prompt
    assert "Objective: Fix regression: missing signature button" in rendered.codex_prompt


def test_structured_regression_task_preserves_details(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), STRUCTURED_SIGNATURE_REGRESSION_TASK)

    assert "Title: Restore missing platform-company signature action" in rendered.codex_prompt
    assert "Task mode: continuation_followup" in rendered.codex_prompt
    expected_objective = (
        "Objective: Find the exact action-eligibility condition that hides the signature action after ProviderStatus=Success "
        "and restore the correct behavior."
    )
    assert expected_objective in rendered.codex_prompt
    assert "Observed state:\nAfter the successful Lleida.net integration fix:" in rendered.codex_prompt
    assert '* "Enviar a firmar" disappeared' in rendered.codex_prompt
    assert "Expected behavior:\nProvider dispatch success means only that Lleida.net accepted the request." in rendered.codex_prompt
    assert "Restrictions:\n* Do not modify ConfigId handling." in rendered.codex_prompt
    assert "Provider dispatch success only proves provider acceptance" in rendered.codex_prompt
    assert "* Fix only the regression." in rendered.codex_prompt
    assert "Validation:\n* Build." in rendered.codex_prompt
    assert "renders the signature action again." in rendered.codex_prompt
    assert "Objective: Objective:" not in rendered.codex_prompt


def test_golden_email_password_autofill_prompt_quality(tmp_path: Path) -> None:
    task = "Fix bug: the login form autofills Email and contraseña unexpectedly. Keep the login behavior scoped to the existing UI."

    rendered = _render(_build_repo(tmp_path), task)
    task_mode = engineering._classify_task_mode(task)
    selected = engineering._selected_prompt_skills(task, task_mode)
    names = tuple(skill.name for skill in selected)
    sources = {skill.name: skill.source for skill in selected}

    assert "Task mode: ui_runtime_bug" in rendered.codex_prompt
    assert "form_security_autofill_bug" in names
    assert "ui_runtime_bug" not in names
    assert "navigation_surface_convergence" not in names
    assert "mvp_surface_completion" not in names
    assert sources["form_security_autofill_bug"] == "pack"
    assert sources["implementation_fix"] == "runtime_generic"
    assert "Original user task:" not in rendered.codex_prompt
    assert task not in rendered.codex_prompt
    assert "Observed state:" in rendered.codex_prompt
    assert "Email/password fields are being autofilled" in rendered.codex_prompt
    assert "Expected behavior:" in rendered.codex_prompt
    assert "Objective:" in rendered.codex_prompt
    assert "Scope:" in rendered.codex_prompt
    assert "Restrictions:" in rendered.codex_prompt
    assert "Validation:" in rendered.codex_prompt
    assert "Email" in rendered.codex_prompt
    assert "contraseña" in rendered.codex_prompt
    assert "authentication/form field behavior" in rendered.codex_prompt
    assert "Verify the `Email` and `contraseña` fields render with the expected autofill behavior." in rendered.codex_prompt
    assert "Prove the exact render/hide/loading condition" not in rendered.codex_prompt
    assert "prove the exact hide/render condition" not in rendered.codex_prompt
    assert "loading state" not in rendered.codex_prompt
    assert "spinner/loading state" not in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_prompt_skill_selection_prefers_form_security_over_generic_ui(tmp_path: Path) -> None:
    task = "Fix bug: the login form autofills Email and contraseña unexpectedly."

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    sources = {skill.name: skill.source for skill in skills}
    assert names == (
        "base_prompt_quality",
        "form_security_autofill_bug",
        "implementation_fix",
        "spanish_implementation_task_preservation",
    )
    assert sources == {
        "base_prompt_quality": "runtime_generic",
        "form_security_autofill_bug": "pack",
        "implementation_fix": "runtime_generic",
        "spanish_implementation_task_preservation": "pack",
    }
    assert "ui_runtime_bug" not in names
    rendered = _render(_build_repo(tmp_path), task)
    assert "Prove the exact render/hide/loading condition" not in rendered.codex_prompt


def test_prompt_skill_selection_includes_ui_runtime_skill_for_non_form_ui_bug() -> None:
    task = "Fix UI regression: modal button stays hidden after the view loads."

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    sources = {skill.name: skill.source for skill in skills}
    assert "base_prompt_quality" in names
    assert "ui_runtime_bug" in names
    assert "form_security_autofill_bug" not in names
    assert sources["ui_runtime_bug"] == "runtime_generic"


def test_prompt_skill_selection_includes_provider_api_skill() -> None:
    task = "Fix provider API dispatch regression after Lleida signature send."

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    sources = {skill.name: skill.source for skill in skills}
    assert "base_prompt_quality" in names
    assert "provider_api_bug" in names
    assert sources["provider_api_bug"] == "pack"


def test_prompt_skill_selection_prefers_provider_bootstrap_over_provider_api() -> None:
    task = "Diagnostic bootstrap: verify Lleida SET_CONFIG/config before provider smoke."

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    assert "provider_bootstrap_diagnostic" in names
    assert "provider_api_bug" not in names


def test_prompt_skill_selection_includes_spanish_preservation_skill() -> None:
    task = "Eliminemos la opción de añadir más de una cuenta desde FormNewBankDataId."

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    assert "base_prompt_quality" in names
    assert "spanish_implementation_task_preservation" in names


def test_prompt_skill_selection_includes_pack_owned_localization_skill() -> None:
    task = "Administration multilingual completion"

    task_mode = engineering._classify_task_mode(task)
    skills = engineering._selected_prompt_skills(task, task_mode)

    names = tuple(skill.name for skill in skills)
    sources = {skill.name: skill.source for skill in skills}
    assert task_mode == "implementation_fix"
    assert "localization_completion" in names
    assert "implementation_fix" in names
    assert sources["localization_completion"] == "pack"
    assert sources["implementation_fix"] == "runtime_generic"


def test_golden_single_bank_account_ui_prompt_quality(tmp_path: Path) -> None:
    task = (
        "Eliminemos la opción de añadir más de una cuenta en todos lados para cualquier entidad. "
        "Aplicar el patrón desde FormNewBankDataId y quitar el botón Añadir de la tabla de cuentas en las vistas de creación."
    )

    rendered = _render(_build_repo(tmp_path), task)
    task_mode = engineering._classify_task_mode(task)
    names = tuple(skill.name for skill in engineering._selected_prompt_skills(task, task_mode))

    assert "Title: Remove the option to add more than one" in rendered.codex_prompt
    assert "Task mode: ui_runtime_bug" in rendered.codex_prompt
    assert "global_pattern_change" in names
    assert "operational_workflow_convergence" in names
    assert "localization_completion" not in names
    assert "ui_runtime_bug" not in names
    assert "navigation_surface_convergence" not in names
    assert "Original user task:" not in rendered.codex_prompt
    assert task not in rendered.codex_prompt
    assert "Observed state:" in rendered.codex_prompt
    assert "repeated pattern and inconsistent implementations can drift across usages" in rendered.codex_prompt
    assert "Expected behavior:" in rendered.codex_prompt
    assert "Scope:" in rendered.codex_prompt
    assert "Find the shared pattern or all targeted usages; change them consistently without broad redesign." in rendered.codex_prompt
    assert "Trace the real workflow state, action eligibility, and rendered operator surface before editing." in rendered.codex_prompt
    assert "Preserve exact references: FormNewBankDataId, botón Añadir, tabla de cuentas, vistas de creación." in rendered.codex_prompt
    assert "Do not expand scope beyond the named global pattern or entity set." in rendered.codex_prompt
    assert "Task details:\n(none)" not in rendered.codex_prompt
    assert "Original user task:" not in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_golden_accounting_navigation_prompt_quality(tmp_path: Path) -> None:
    task = (
        "Fix accounting navigation/dashboard/views convergence so operator menu/index entries open the useful operational surface. "
        "Complete the MVP operator surface."
    )

    rendered = _render(_build_repo(tmp_path), task)
    task_mode = engineering._classify_task_mode(task)
    names = tuple(skill.name for skill in engineering._selected_prompt_skills(task, task_mode))

    assert "navigation_surface_convergence" in names
    assert "mvp_surface_completion" in names
    assert "ui_runtime_bug" not in names
    assert "Navigation, menu, dashboard, or view surfaces are not converging" in rendered.codex_prompt
    assert "Operators should reach the same useful workflow surface consistently" in rendered.codex_prompt
    assert "Trace route/menu/view entry points and converge only the requested navigation surface." in rendered.codex_prompt
    assert "Complete only the requested operator-facing surface needed for the daily workflow." in rendered.codex_prompt
    assert "Do not redesign dashboards or unrelated navigation" in rendered.codex_prompt
    assert "spinner" not in rendered.codex_prompt.lower()
    assert "button render" not in rendered.codex_prompt.lower()
    assert "Original user task:" not in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_golden_lleida_set_config_prompt_quality(tmp_path: Path) -> None:
    task = "Diagnostic bootstrap: verify Lleida SET_CONFIG/config before provider smoke. Do not change START_SIGNATURE payloads."

    rendered = _render(_build_repo(tmp_path), task)
    task_mode = engineering._classify_task_mode(task)
    names = tuple(skill.name for skill in engineering._selected_prompt_skills(task, task_mode))

    assert "provider_bootstrap_diagnostic" in names
    assert "provider_api_bug" not in names
    assert "Provider bootstrap/configuration needs bounded verification" in rendered.codex_prompt
    assert "Limit work to explicit provider bootstrap, config, SET_CONFIG, or smoke-diagnostic checks." in rendered.codex_prompt
    assert "Report exact bounded diagnostic/config evidence and whether provider calls were made." in rendered.codex_prompt
    assert "Do not modify START_SIGNATURE payloads unless explicitly targeted." in rendered.codex_prompt
    assert "Original user task:" not in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_golden_lleida_spinner_provider_hosted_send_prompt_quality(tmp_path: Path) -> None:
    task = (
        "Fix regression: Lleida provider-hosted send leaves the spinner visible after ProviderStatus=Success. "
        "Do not modify ConfigId, SET_CONFIG, or START_SIGNATURE payloads."
    )

    rendered = _render(_build_repo(tmp_path), task)

    assert "Task mode: provider_api_bug" in rendered.codex_prompt
    assert "Observed state:" in rendered.codex_prompt
    assert "spinner/loading state" in rendered.codex_prompt
    assert "Provider dispatch success only proves provider acceptance" in rendered.codex_prompt
    assert "it does not prove signature completion or workflow completion" in rendered.codex_prompt
    assert "Do not modify ConfigId or SET_CONFIG handling unless explicitly targeted." in rendered.codex_prompt
    assert "Do not modify START_SIGNATURE payloads unless explicitly targeted." in rendered.codex_prompt
    assert "Verify provider dispatch success remains intact" in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_golden_missing_resend_signature_action_prompt_quality(tmp_path: Path) -> None:
    rendered = _render(_build_repo(tmp_path), STRUCTURED_SIGNATURE_REGRESSION_TASK)
    task_mode = engineering._classify_task_mode(STRUCTURED_SIGNATURE_REGRESSION_TASK)
    names = tuple(skill.name for skill in engineering._selected_prompt_skills(STRUCTURED_SIGNATURE_REGRESSION_TASK, task_mode))

    assert "Task mode: continuation_followup" in rendered.codex_prompt
    assert "provider_api_bug" in names
    assert "provider_bootstrap_diagnostic" not in names
    assert "operational_workflow_convergence" in names
    assert "Observed state:" in rendered.codex_prompt
    assert "ProviderStatus = Success" in rendered.codex_prompt
    assert "ProviderCorrelationId exists" in rendered.codex_prompt
    assert '"Enviar a firmar" disappeared' in rendered.codex_prompt
    assert "Expected behavior:" in rendered.codex_prompt
    assert "Provider dispatch success means only that Lleida.net accepted the request." in rendered.codex_prompt
    assert "Provider dispatch success only proves provider acceptance" in rendered.codex_prompt
    assert 'Keep "Ver estado firma" working.' in rendered.codex_prompt
    assert "Verify the signature send/resend action renders when the signature remains pending." in rendered.codex_prompt
    assert "- Files read" not in rendered.codex_prompt
    assert "- Files changed" not in rendered.codex_prompt


def test_prompt_benchmark_fixtures_pass(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path / "repo")

    results = engineering.run_prompt_benchmarks(benchmarks_dir=_prompt_benchmarks_dir(), repo=repo)

    assert {result.name for result in results} == {
        "email-password-autofill",
        "administration-multilingual-completion",
        "legacy-onboarding-path-convergence",
        "navigation-convergence",
        "single-bank-account-everywhere",
        "lleida-set-config-bootstrap",
        "resend-signature-after-provider-success",
        "spinner-after-provider-dispatch",
    }
    assert all(result.passed for result in results)


def test_prompt_benchmark_reports_failed_characteristics(tmp_path: Path) -> None:
    benchmarks = tmp_path / "prompt_benchmarks"
    _write_prompt_benchmark(
        benchmarks,
        "bad-case",
        "Fix bug: the login form autofills Email unexpectedly.",
        """# Expected Characteristics

## Task mode
- planning_only

## Required skills
- missing_skill

## Forbidden prompt text
- Codex Prompt:
""",
    )

    results = engineering.run_prompt_benchmarks(benchmarks_dir=benchmarks, repo=_build_repo(tmp_path / "repo"))
    report = engineering.render_prompt_benchmark_report(results)

    assert len(results) == 1
    assert not results[0].passed
    assert "task mode: expected planning_only, got ui_runtime_bug" in results[0].failures
    assert "required skill missing: missing_skill" in results[0].failures[1]
    assert "forbidden prompt text present: Codex Prompt:" in results[0].failures
    assert "FAIL bad-case" in report
    assert "Summary: 0 passed, 1 failed" in report


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


def test_cli_task_output_preserves_spanish_unicode(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    output = tmp_path / "prompt.md"
    task = (
        "Eliminemos la opción de añadir más de una cuenta desde FormNewBankDataId. "
        "Quitar el botón Añadir de la tabla de cuentas en las vistas de creación."
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used with --no-copy"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["task", "--repo", str(repo), "--output", str(output), "--no-copy", task]) == 0

    stdout = capsys.readouterr().out
    prompt = output.read_text(encoding="utf-8")
    assert "Title: Remove the option to add more than one" in prompt
    assert "Task mode: ui_runtime_bug" in prompt
    assert "Original user task:" not in prompt
    assert task not in prompt
    assert "Objective: Remove the option to add more than one account" in prompt
    assert "Task details:\nObserved state:" in prompt
    assert "Expected behavior:" in prompt
    assert "Remove the option to add more than one account" in prompt
    assert "Objective: Eliminemos" not in prompt
    assert "Task details:\nEliminemos" not in prompt
    assert "FormNewBankDataId" in prompt
    assert "botón Añadir" in prompt
    assert "tabla de cuentas" in prompt
    assert "vistas de creación" in prompt
    assert "Task details:\n(none)" not in prompt
    assert "Do not edit files unless explicitly requested by this task." not in prompt
    for text in ("botón Añadir", "tabla de cuentas", "vistas de creación"):
        assert text in prompt
        assert text in stdout
    for mojibake in ("opci¾n", "a±adir", "mßs"):
        assert mojibake not in prompt
        assert mojibake not in stdout


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


def test_cli_plan_prints_execution_plan(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")

    assert cli.main(["plan", "--repo", str(repo), "Fix", "legacy", "operator", "navigation", "views"]) == 0

    stdout = capsys.readouterr().out
    assert "Execution Plan:" in stdout
    assert "Profile: supervised_implementation" in stdout
    assert "- planner:" in stdout
    assert "- context-loader:" in stdout
    assert "- implementer:" in stdout
    assert "- reviewer:" in stdout
    assert "- validation-runner:" in stdout
    assert "- drift-guard:" in stdout
    assert "Skills:" in stdout
    assert "- navigation_surface_convergence [pack]:" in stdout
    assert "- implementation_fix [runtime_generic]:" in stdout
    assert "Reviewer chain:" in stdout
    assert "Safety gates:" in stdout
    assert "Codex execution: disabled" in stdout


def test_cli_benchmark_runs_prompt_benchmarks(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")

    assert cli.main(["benchmark", "--benchmarks", str(_prompt_benchmarks_dir()), "--repo", str(repo)]) == 0

    stdout = capsys.readouterr().out
    assert "Prompt Benchmark Results" in stdout
    assert "PASS email-password-autofill" in stdout
    assert "PASS administration-multilingual-completion" in stdout
    assert "PASS spinner-after-provider-dispatch" in stdout
    assert "Summary: 8 passed, 0 failed" in stdout


def test_cli_benchmark_returns_nonzero_for_failed_characteristics(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    benchmarks = tmp_path / "prompt_benchmarks"
    _write_prompt_benchmark(
        benchmarks,
        "bad-case",
        "Fix bug: the login form autofills Email unexpectedly.",
        """# Expected Characteristics

## Task mode
- planning_only
""",
    )

    assert cli.main(["benchmark", "--benchmarks", str(benchmarks), "--repo", str(repo)]) == 1

    stdout = capsys.readouterr().out
    assert "FAIL bad-case" in stdout
    assert "task mode: expected planning_only, got ui_runtime_bug" in stdout
    assert "Summary: 0 passed, 1 failed" in stdout


def test_cli_run_is_temporarily_disabled(monkeypatch, capsys) -> None:
    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used while ph run is disabled"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["run", "FinanciacionCore"]) == 1

    stdout = capsys.readouterr().out
    assert "ph run is temporarily disabled. Use ph task to generate/copy prompts and ph review/ph review-codex for review." in stdout
    assert "Paste or type the task" not in stdout
    assert "Codex prompt copied to clipboard" not in stdout


def test_cli_review_codex_copies_reviewer_prompt_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-good.txt"
    codex_output.write_text(
        """Summary
- Implemented the small task.

Validation
- smoke check passed.

PASS
""",
        encoding="utf-8",
    )
    copied: list[str] = []

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["review-codex", str(repo), "--codex-output", str(codex_output), "small", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Codex reviewer prompt copied to clipboard" in stdout
    assert len(copied) == 1
    assert copied[0].startswith("Codex Reviewer Prompt\n")
    assert "Original task:" not in copied[0]
    assert "Original user task:" not in copied[0]
    assert "Task mode: review_only" in copied[0]
    assert "Selected context paths:\n- AGENTS.md\n- MEMORY.md" in copied[0]
    assert "Generated implementation prompt:\nCodex Prompt:" in copied[0]
    assert "Implementation Codex output:" in copied[0]
    assert "Deterministic reviewer findings:" in copied[0]
    assert "Review verdict: PASS / FAIL / NEEDS_FOLLOW_UP" in copied[0]
    assert (
        "Do not require `Files read` or `Files changed` unless the generated implementation prompt explicitly asks for those sections." in copied[0]
    )


def test_cli_review_codex_bad_output_includes_deterministic_findings(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-bad.txt"
    codex_output.write_text(
        """Summary
- Added dashboard memory graph MCP governance documentation changes.

Validation
- not run

PASS
""",
        encoding="utf-8",
    )
    copied: list[str] = []

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["review-codex", str(repo), "--codex-output", str(codex_output), "small", "payment", "workflow", "task"]) == 0

    capsys.readouterr()
    assert "Status: REVIEW_NEEDED" in copied[0]
    assert "- dashboard mentioned without task scope" in copied[0]
    assert "- PASS claimed without validation evidence" in copied[0]
    assert "Follow-up prompt if needed" in copied[0]


def test_cli_review_codex_output_writes_only_reviewer_prompt(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-good.txt"
    output = tmp_path / "reviewer-prompt.md"
    codex_output.write_text(
        """Summary
- Implemented the small task.

Validation
- smoke check passed.

PASS
""",
        encoding="utf-8",
    )
    copied: list[str] = []

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        copied.append(text)
        return cli._ClipboardCopyResult(copied=True)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["review-codex", str(repo), "--codex-output", str(codex_output), "--output", str(output), "small", "task"]) == 0

    stdout = capsys.readouterr().out
    assert "Codex reviewer prompt copied to clipboard" in stdout
    assert output.read_text(encoding="utf-8") == f"{copied[0]}\n"
    assert output.read_text(encoding="utf-8").startswith("Codex Reviewer Prompt\n")


def test_cli_review_codex_no_copy_prints_prompt_without_clipboard(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-good.txt"
    codex_output.write_text(
        """Summary
- Implemented the small task.

Validation
- smoke check passed.

PASS
""",
        encoding="utf-8",
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used with --no-copy"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert cli.main(["review-codex", str(repo), "--codex-output", str(codex_output), "--no-copy", "small", "task"]) == 0

    stdout = capsys.readouterr().out
    assert stdout.startswith("Codex Reviewer Prompt\n")
    assert "Implementation Codex output:" in stdout
    assert "Deterministic reviewer findings:" in stdout


def test_cli_outcome_prints_supervised_outcome_report(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text(
        """Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        encoding="utf-8",
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used by ph outcome"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert (
        cli.main(
            [
                "outcome",
                "--repo",
                str(repo),
                "--codex-output",
                str(codex_output),
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert stdout.startswith("ECC Supervised Outcome Report\n")
    assert "Status: accepted" in stdout
    assert "Actual files changed:\n- Views/Operator/Index.cshtml" in stdout
    assert "Supervision boundaries:" in stdout


def test_cli_outcome_can_save_and_list_history(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text(
        """Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        encoding="utf-8",
    )

    def copy_to_clipboard(text: str) -> cli._ClipboardCopyResult:
        _ = text
        msg = "clipboard should not be used by ph outcome"
        raise AssertionError(msg)

    monkeypatch.setattr(cli, "_copy_to_clipboard", copy_to_clipboard)

    assert (
        cli.main(
            [
                "outcome",
                "--repo",
                str(repo),
                "--codex-output",
                str(codex_output),
                "--save-history",
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert "Outcome history saved:" in stdout
    assert (repo / ".protector-harness" / "outcome-history.jsonl").is_file()

    assert cli.main(["outcome-history", "--repo", str(repo), "--limit", "5"]) == 0

    history = capsys.readouterr().out
    assert history.startswith("ECC Outcome History\n")
    assert "Entries shown: 1" in history
    assert "accepted | protector-financiacioncore" in history
    assert "changed files: Views/Operator/Index.cshtml" in history
    assert "validations: Build verified and UI workflow route smoke test check passed." in history


def test_cli_plan_with_history_surfaces_recent_signals(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    report = engineering.render_supervised_outcome_report(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
        codex_output="""Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        source="codex-output.txt",
    )
    engineering.append_outcome_history(repo, report)

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--with-history",
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert "Recent outcome signals:" in stdout
    assert "accepted | skills=base_prompt_quality, implementation_fix, navigation_surface_convergence" in stdout


def test_cli_plan_can_export_automation_candidate_json(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    output = tmp_path / "candidate.json"

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--candidate-json",
                str(output),
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    approval_sha = engineering.automation_candidate_approval_sha(payload)
    assert stdout.startswith("Execution Plan:")
    assert "Automation Readiness:" in stdout
    assert "ecc-automation-candidate-v1" not in stdout
    assert "Candidate Execution Guide (PowerShell)" in stdout
    assert "1. Generate candidate:" in stdout
    assert "2. Dry-run candidate:" in stdout
    assert "3. Copy approval SHA:" in stdout
    assert "4. Execute candidate once:" in stdout
    assert "5. Review outcome and save history:" in stdout
    assert f"Approval SHA: {approval_sha}" in stdout
    assert f"ph candidate-dry-run {output.resolve()}" in stdout
    assert f"ph candidate-execute --approve-sha {approval_sha}" in stdout
    assert f"--output {output.with_suffix('.codex-output.txt').resolve()}" in stdout
    assert f"ph candidate-outcome {output.resolve()}" in stdout
    assert "--save-history" in stdout
    assert "candidate-execute runs one foreground Codex command only." in stdout
    assert payload["schema_version"] == "ecc-automation-candidate-v1"
    assert payload["selected_pack"]["benchmark_validation_status"] == "passing"
    assert payload["task"]["mode"] == "ui_runtime_bug"
    assert payload["automation_readiness"]["classification"] == "automation_ready"
    assert payload["automation_readiness"]["readiness_decision_record"]["decision"].startswith(
        "automation_ready wins because required evidence is present"
    )
    assert payload["review_contract"]["pass_fail_criteria"]
    assert payload["proposed_codex_prompt"].startswith("Codex Prompt:")


def test_cli_candidate_dry_run_imports_exported_candidate_json(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    output = tmp_path / "candidate.json"

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--candidate-json",
                str(output),
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert cli.main(["candidate-dry-run", str(output)]) == 0

    stdout = capsys.readouterr().out
    assert stdout.startswith("ECC Candidate Import Dry Run\n")
    assert "Schema validation: PASS" in stdout
    assert "Decision: would execute" in stdout
    assert "Readiness Decision Record:" in stdout
    assert "automation_ready wins because required evidence is present" in stdout
    assert "- Selected pack still valid: yes" in stdout
    assert "- Selected skills still available: yes" in stdout
    assert "- Benchmark coverage still valid: yes" in stdout
    assert "- Readiness classification explainable: yes" in stdout
    assert "Codex prompt preview:\nCodex Prompt:" in stdout
    assert "- Codex execution: disabled" in stdout
    assert "- Model calls: disabled" in stdout
    assert "- File edits: disabled by dry-run importer" in stdout
    assert "- Autonomous loops: disabled" in stdout
    assert "- Workflow engine: disabled" in stdout


def test_cli_candidate_outcome_reviews_exported_candidate_and_saves_history(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text(
        """Files read
- AGENTS.md
- Views/Operator/Index.cshtml

Files changed
- Views/Operator/Index.cshtml

Summary
- Updated navigation route menu view operator onboarding convergence while preserving workflow.

Validation
- Build verified and UI workflow route smoke test check passed.

PASS
""",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "plan",
                "--repo",
                str(repo),
                "--candidate-json",
                str(candidate),
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        cli.main(
            [
                "candidate-outcome",
                "--repo",
                str(repo),
                "--codex-output",
                str(codex_output),
                "--save-history",
                str(candidate),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert stdout.startswith("ECC Supervised Outcome Report\n")
    assert "Status: accepted" in stdout
    assert "Candidate source:\n-" in stdout
    assert "Automation readiness:\n- Classification: automation_ready" in stdout
    assert "Outcome history saved:" in stdout
    history_path = repo / ".protector-harness" / "outcome-history.jsonl"
    payload = json.loads(history_path.read_text(encoding="utf-8").strip())
    assert payload["status"] == "accepted"
    assert payload["changed_files"] == ["Views/Operator/Index.cshtml"]
    assert {"name": "navigation_surface_convergence", "source": "pack"} in payload["selected_skills"]
    assert "codex_output" not in payload
    assert "Files read" not in history_path.read_text(encoding="utf-8")


def test_cli_candidate_outcome_requires_repo_to_save_history(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    codex_output = tmp_path / "codex-output.txt"
    codex_output.write_text("PASS\n", encoding="utf-8")
    candidate.write_text(
        json.dumps(
            engineering.render_controlled_execution_plan(
                task="Fix legacy onboarding path convergence for operator UI views",
                repo=repo,
            ).automation_candidate
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["candidate-outcome", "--codex-output", str(codex_output), "--save-history", str(candidate)])

    assert exc_info.value.code == 2
    assert "--save-history requires --repo for candidate-outcome" in capsys.readouterr().err


def test_cli_candidate_execute_refuses_invalid_candidate(tmp_path: Path, monkeypatch, capsys) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"schema_version": "bad"}), encoding="utf-8")

    def fail_popen(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        msg = "candidate-execute must not invoke Codex for invalid candidates"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.subprocess, "Popen", fail_popen)

    assert cli.main(["candidate-execute", str(candidate)]) == 1

    stdout = capsys.readouterr().out
    assert "Schema validation: FAIL" in stdout
    assert "Candidate execution refused: invalid candidate." in stdout


def test_cli_candidate_execute_refuses_non_automation_ready_candidate(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    payload = engineering.render_controlled_execution_plan(
        task="Review payment implementation notes",
        repo=repo,
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)

    def fail_popen(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        msg = "candidate-execute must not invoke Codex for supervised_only candidates"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.subprocess, "Popen", fail_popen)

    assert cli.main(["candidate-execute", "--approve-sha", approval_sha, str(candidate)]) == 1

    stdout = capsys.readouterr().out
    assert "candidate readiness is supervised_only; expected automation_ready" in stdout
    assert f"Candidate approval SHA: {approval_sha}" in stdout


def test_cli_candidate_execute_refuses_missing_or_wrong_approval_sha(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    payload = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)

    def fail_popen(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        msg = "candidate-execute must not invoke Codex before approval"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.subprocess, "Popen", fail_popen)

    assert cli.main(["candidate-execute", str(candidate)]) == 1
    missing_stdout = capsys.readouterr().out
    assert "approval SHA is missing" in missing_stdout
    assert f"Candidate approval SHA: {approval_sha}" in missing_stdout

    assert cli.main(["candidate-execute", "--approve-sha", "wrong", str(candidate)]) == 1
    wrong_stdout = capsys.readouterr().out
    assert "approval SHA does not match" in wrong_stdout
    assert f"Re-run with: --approve-sha {approval_sha}" in wrong_stdout


def test_cli_candidate_execute_invokes_fake_codex_once_and_writes_explicit_output(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "codex-output.txt"
    payload = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)
    calls: list[dict[str, object]] = []

    class FakeStdin:
        def __init__(self) -> None:
            self.text = ""
            self.closed = False

        def write(self, text: str) -> None:
            self.text += text

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, command: object, **kwargs: object) -> None:
            self.stdin = FakeStdin()
            self.stdout = iter(("fake codex output\n",))
            self.command = command
            self.kwargs = kwargs
            calls.append({"command": command, "kwargs": kwargs, "process": self})

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", FakeProcess)

    assert (
        cli.main(
            [
                "candidate-execute",
                "--codex-cmd",
                "fake-codex",
                "--approve-sha",
                approval_sha,
                "--output",
                str(output),
                "--repo",
                str(repo),
                str(candidate),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert len(calls) == 1
    process = calls[0]["process"]
    assert calls[0]["command"] == ("fake-codex",)
    assert calls[0]["kwargs"]["stdin"] == cli.subprocess.PIPE
    assert calls[0]["kwargs"]["stdout"] == cli.subprocess.PIPE
    assert process.stdin.text.startswith("Codex Prompt:\n")
    assert process.stdin.closed
    assert "fake codex output" in stdout
    assert output.read_text(encoding="utf-8") == "fake codex output\n"
    assert "Next review command:" in stdout
    assert f"ph candidate-outcome {candidate.resolve()} --codex-output {output.resolve()} --repo {repo} --save-history" in stdout


def test_cli_candidate_execute_does_not_persist_raw_output_without_output_option(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    payload = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)

    class FakeStdin:
        def write(self, text: str) -> None:
            assert text.startswith("Codex Prompt:\n")

        def close(self) -> None:
            return None

    class FakeProcess:
        stdin = FakeStdin()
        stdout = iter(("transient output only\n",))

        def __init__(self, command: object, **kwargs: object) -> None:
            _ = command, kwargs

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", FakeProcess)

    assert cli.main(["candidate-execute", "--codex-cmd", "fake-codex", "--approve-sha", approval_sha, str(candidate)]) == 0

    stdout = capsys.readouterr().out
    assert "transient output only" in stdout
    assert "--codex-output <codex-output>" in stdout
    assert not (tmp_path / "codex-output.txt").exists()


def test_cli_candidate_execute_interrupt_terminates_child_and_preserves_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "codex-output.txt"
    payload = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)
    terminated: list[object] = []

    class FakeStdin:
        def write(self, text: str) -> None:
            assert text.startswith("Codex Prompt:\n")

        def close(self) -> None:
            return None

    class InterruptingStdout:
        def __init__(self) -> None:
            self.index = 0

        def __iter__(self) -> "InterruptingStdout":
            return self

        def __next__(self) -> str:
            self.index += 1
            if self.index == 1:
                return "partial codex output\n"
            raise KeyboardInterrupt

    class FakeProcess:
        pid = 12345

        def __init__(self, command: object, **kwargs: object) -> None:
            _ = command, kwargs
            self.stdin = FakeStdin()
            self.stdout = InterruptingStdout()

        def wait(self) -> int:
            msg = "wait should not be reached after KeyboardInterrupt"
            raise AssertionError(msg)

    def fake_terminate(process: object) -> bool:
        terminated.append(process)
        return True

    monkeypatch.setattr(cli.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(cli, "_terminate_process_tree", fake_terminate)

    assert (
        cli.main(
            [
                "candidate-execute",
                "--codex-cmd",
                "fake-codex",
                "--approve-sha",
                approval_sha,
                "--output",
                str(output),
                "--repo",
                str(repo),
                str(candidate),
            ]
        )
        == 130
    )

    stdout = capsys.readouterr().out
    assert len(terminated) == 1
    assert "partial codex output" in stdout
    assert output.read_text(encoding="utf-8") == "partial codex output\n"
    assert "interrupted_by_operator" in stdout
    assert "child_process_terminated" in stdout
    assert "Next recommended command:" in stdout
    assert f"ph candidate-outcome {candidate.resolve()} --codex-output {output.resolve()} --repo {repo} --save-history" in stdout


def test_cli_candidate_execute_interrupt_reports_termination_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    candidate = tmp_path / "candidate.json"
    payload = engineering.render_controlled_execution_plan(
        task="Fix legacy onboarding path convergence for operator UI views",
        repo=repo,
        repo_alias="FinanciacionCore",
    ).automation_candidate
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    approval_sha = engineering.automation_candidate_approval_sha(payload)

    class FakeStdin:
        def write(self, text: str) -> None:
            assert text.startswith("Codex Prompt:\n")

        def close(self) -> None:
            return None

    class InterruptingStdout:
        def __iter__(self) -> "InterruptingStdout":
            return self

        def __next__(self) -> str:
            raise KeyboardInterrupt

    class FakeProcess:
        stdin = FakeStdin()
        stdout = InterruptingStdout()

        def __init__(self, command: object, **kwargs: object) -> None:
            _ = command, kwargs

    def fake_terminate_failure(process: object) -> bool:
        _ = process
        return False

    monkeypatch.setattr(cli.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(cli, "_terminate_process_tree", fake_terminate_failure)

    assert cli.main(["candidate-execute", "--codex-cmd", "fake-codex", "--approve-sha", approval_sha, str(candidate)]) == 130

    stdout = capsys.readouterr().out
    assert "interrupted_by_operator" in stdout
    assert "termination_failed" in stdout
    assert "Next recommended command:" not in stdout


def test_run_codex_once_replaces_invalid_utf8_output(capsys) -> None:
    result = cli._run_codex_once(
        (
            sys.executable,
            "-c",
            "import sys; sys.stdin.read(); sys.stdout.buffer.write(b'valid\\xff\\n')",
        ),
        "prompt",
    )

    stdout = capsys.readouterr().out
    assert result.returncode == 0
    assert result.output == "valid\ufffd\n"
    assert stdout == "valid\ufffd\n"


def test_cli_outcome_learning_lists_derived_signals(tmp_path: Path, capsys) -> None:
    repo = _build_repo(tmp_path / "repo")
    for _ in range(2):
        _append_outcome_summary(
            repo,
            validations=("not run",),
            deviations=("dashboard mentioned without task scope",),
            follow_up="Revise or justify the Codex result for: Fix legacy onboarding path convergence.",
        )

    assert (
        cli.main(
            [
                "outcome-learning",
                "--repo",
                str(repo),
                "--limit",
                "10",
                "Fix",
                "legacy",
                "onboarding",
                "path",
                "convergence",
                "for",
                "operator",
                "UI",
                "views",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    assert stdout.startswith("ECC Outcome Learning Signals\n")
    assert "Recurring failed validation: not run (2 outcomes)." in stdout
    assert "Repeated drift/deviation pattern: dashboard mentioned without task scope (2 outcomes)." in stdout


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
