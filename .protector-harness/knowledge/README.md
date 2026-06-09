# Protector Knowledge Base

Repo-specific knowledge files give Protector Operator compact context for prompt generation.

Create one Markdown file per repo alias:

```txt
.protector-harness/knowledge/<RepoAlias>.md
```

Supported sections only:

```md
## Current phase
## Current priorities
## Authoritative workflows
## Protected decisions
## Forbidden directions
## Known useful routes/views/tests
```

Keep entries short and factual. The harness uses these files to synthesize a few prompt guardrails; it does not paste the raw file into generated prompts.
