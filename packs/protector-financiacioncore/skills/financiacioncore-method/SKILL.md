---
name: financiacioncore-method
description: Apply FinanciacionCore/Vameco engineering method: MVP convergence, contract-first onboarding, route-backed proof, and protected workflow decisions.
origin: Protector ECC pack
---

# FinanciacionCore Method

Use this skill for FinanciacionCore or Vameco tasks where project-specific engineering method matters.

## Objective

Move the product toward usable SaaS MVP convergence with small, reversible, evidence-backed changes.

## Method

- Prefer the highest-impact remaining workflow blocker over polish or speculative architecture.
- Trace the real rendered route, view, JavaScript handler, service, and test boundary before changing behavior.
- Prefer SQL-backed or route-backed runtime evidence when workflow correctness is at stake.
- Preserve contract-first company and financer onboarding.
- Keep legacy direct routes backend-compatible when needed, but do not promote them in normal UI.

## Protected Areas

- Do not touch Contabilidad unless the task targets accounting.
- Do not touch telemetry unless the task targets telemetry.
- Do not touch resilience unless the task targets resilience.
- Do not introduce new onboarding subsystems, BPM, parallel workflow engines, or broad rewrites.
