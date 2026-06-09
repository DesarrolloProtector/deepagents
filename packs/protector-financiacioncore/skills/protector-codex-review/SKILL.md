---
name: protector-codex-review
description: Review pasted Codex output against Protector scope, validation, and anti-drift expectations.
origin: Protector ECC pack
---

# Protector Codex Review

Use this skill when reviewing Codex output produced from a Protector prompt.

## Objective

Decide whether the output satisfies the requested task without scope drift, unsupported PASS claims, or missing validation evidence.

## Review Checks

- Confirm the output addresses the requested repo/workflow surface.
- Check whether protected decisions and must-not-touch areas were respected.
- Require proportional validation before PASS.
- Flag broad audits, governance expansion, dashboards, memory systems, MCP, autonomous execution, or unrelated architecture unless explicitly requested.
- Produce follow-up instructions only for the missing or risky parts.

## Not Owned Here

- Generic reviewer chains belong to ECC.
- Session persistence and dashboard review workflows belong to ECC.
