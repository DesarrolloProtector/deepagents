---
name: protector-anti-drift
description: Apply Protector-specific anti-drift rules for scoped engineering tasks and Codex handoffs.
origin: Protector ECC pack
---

# Protector Anti-Drift

Use this skill when a Protector task could expand beyond the requested slice.

## Objective

Keep work bounded to the user's requested behavior, repo surface, and validation path.

## Rules

- Do not turn implementation requests into broad audits unless explicitly requested.
- Do not expand into governance, dashboards, memory, MCP, autonomous execution, or generic agent infrastructure.
- Do not touch unrelated workflows, telemetry, resilience, accounting, or provider configuration unless the task targets them.
- Preserve existing routes, permissions, workflows, handlers, persistence, and public contracts unless explicitly targeted.
- Validate the narrow behavior that changed, not an unrelated success path.

## ECC Boundary

Protector anti-drift rules are domain specialization. Generic enforcement mechanisms, hooks, loops, and dashboards belong to ECC.
