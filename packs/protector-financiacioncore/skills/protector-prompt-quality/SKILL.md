---
name: protector-prompt-quality
description: Preserve Protector prompt benchmark quality while keeping generated Codex prompts compact, task-specific, and free of generic drift.
origin: Protector ECC pack
---

# Protector Prompt Quality

Use this skill when a Protector prompt must be generated, reviewed, or benchmarked.

## Objective

Keep the task-specific request dominant while adding only compact context that improves correctness.

## Rules

- Do not paste raw knowledge files into prompts.
- Do not compare exact prompt text as the primary quality contract.
- Prefer structured characteristics: task mode, selected skills, required guardrails, forbidden generic wording, and benchmark PASS/FAIL details.
- Keep prompt benchmarks fixture-driven with `task.txt` and `expected_characteristics.md`.
- Preserve `ph benchmark` as the quality gate during the ECC-pack transition.

## Not Owned Here

- Agent orchestration belongs to ECC.
- Skill registry mechanics belong to ECC.
- Model calls and Codex execution are out of scope.
