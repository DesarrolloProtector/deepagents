# Protector ECC Architecture

Protector is an ECC-backed specialization layer. It should not become a competing agentic platform.

## ECC-Owned Capabilities

ECC owns the reusable agentic engineering platform:

- Agent catalog and role definitions.
- Skill catalog, skill packaging, and skill evolution.
- Session adapters and session state.
- Memory persistence and continuous learning.
- Generic workflows, verification loops, and quality gates.
- Hook definitions where the target harness supports hooks.
- Dashboard and operator control-plane surfaces.
- Codex installation/support assets, including Codex-oriented config, skills, session adapters, and instruction surfaces.
- Multi-agent orchestration primitives.
- Generic sandbox, review, and loop policy foundations.

Protector may discover these capabilities from ECC, but must not duplicate ECC registries.

## Protector-Owned Capabilities

Protector owns only the FinanciacionCore/Vameco specialization:

- Prompt-quality benchmarks and regression fixtures.
- Repo/domain knowledge files.
- Codex handoff and pasted-output review adapter.
- Anti-drift rules tied to known product and workflow constraints.
- FinanciacionCore/Vameco engineering method and convergence guidance.
- Compact prompt enrichment that keeps the task-specific request dominant.

## Components To Stop Building In Protector

Stop expanding Protector-local versions of:

- agent registries
- skill registries
- execution profile registries
- sandbox policy registries
- reviewer chain registries
- loop policy registries
- session stores
- memory systems
- dashboard/control-plane frameworks
- generic multi-agent orchestration engines

Existing compatibility code can remain until migrated, but new work should route through ECC discovery or ECC-owned APIs.

## Integration Roadmap

1. Add read-only ECC discovery for path, relevant agents, relevant skills, and install profiles.
2. Expose ECC discovery through `ph ecc-status` for operator visibility.
3. Keep prompt rendering, knowledge loading, benchmark checks, and Codex output review in Protector.
4. Replace Protector-local generic planning components with ECC-backed selections once an ECC API boundary is available.
5. Move any future Operator UI platform display to ECC status/discovery data, not a parallel dashboard model.
6. Only after explicit approval, evaluate supervised Codex execution through ECC sandbox/review gates.

Current restrictions remain:

- No Codex execution.
- No model calls.
- No autonomous loops.
- No dashboard rewrite.
- No migration of existing behavior in this slice.
