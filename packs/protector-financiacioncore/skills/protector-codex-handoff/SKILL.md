---
name: protector-codex-handoff
description: Generate compact Codex handoff prompts for Protector tasks without executing Codex or changing runtime behavior.
origin: Protector ECC pack
---

# Protector Codex Handoff

Use this skill when turning a Protector task into a Codex-ready implementation, review, diagnostic, or planning prompt.

## Objective

Produce a bounded handoff that tells Codex what to inspect, what to preserve, and how to validate without broadening the task.

## Rules

- Keep the user's task dominant.
- Select only relevant repo knowledge.
- Preserve explicit scope, restrictions, validation, and PASS boundaries.
- Require focused route, view, service, or test inspection when the task depends on real behavior.
- Do not ask Codex to run autonomous loops.
- Do not imply that Protector executes Codex.

## Output Discipline

- Prefer compact English structure even when source task text is Spanish.
- Preserve exact identifiers, route names, UI labels, provider terms, and workflow state names.
- Avoid `Files read` and `Files changed` unless explicitly requested.
