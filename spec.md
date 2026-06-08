# Protector Operator UI MVP

## Problem Statement

The current Protector Harness workflow is too painful for daily use. The operator must type long PowerShell commands, manage pasted task text, copy generated prompts, save Codex output to a file, and run separate review commands. The CLI backend is useful and deterministic, but the operator workflow is brittle and easy to mis-sequence.

## Current Broken CLI Workflow

The current manual flow is:

1. Run `ph task <repo> "<task>"`.
2. Paste the generated Codex prompt into an already-open Codex session.
3. Copy Codex output.
4. Save that output to a temporary file.
5. Run `ph review <repo> --codex-output <file> "<task>"`.
6. Optionally run `ph review-codex <repo> --codex-output <file> "<task>"`.

`ph run` is temporarily disabled because the interactive flow can enter review with wrong or residual input. It must not be the daily operator workflow until redesigned.

## Target Operator Workflow

The target workflow is one desktop screen:

1. Select a repository alias.
2. Paste or write the task.
3. Generate the Codex prompt.
4. The app copies the generated prompt to the clipboard.
5. Paste the prompt into an already-open Codex session.
6. Paste Codex output into the app.
7. Run deterministic review or generate a Codex reviewer prompt.
8. Read the result in the app without managing temporary files.

## MVP UI

The MVP desktop screen contains:

- Repo selector populated from existing `ph repos` aliases.
- Large task editor.
- `Generate Codex Prompt` button.
- Generated prompt panel.
- Automatic clipboard copy for generated prompts.
- Large Codex output editor.
- `Deterministic Review` button.
- `Generate Codex Reviewer Prompt` button.
- Review/result panel.
- Status bar with selected repo, task mode, selected context count, and clipboard status.

The UI must preserve Spanish Unicode in task text, generated prompts, pasted Codex output, review text, and clipboard operations.

## Non-Goals

- Do not invoke Codex.
- Do not call models.
- Do not add autonomy.
- Do not add dashboards.
- Do not add memory, MCP, graph storage, background workers, or Kimi.
- Do not replace the Protector Harness backend.
- Do not redesign task classification or deterministic review in this UI slice.

## Integration Approach

The desktop app is a thin operator shell over the existing Python CLI:

- `ph repos` loads repo aliases.
- `ph task <repo> --output <temp> --overwrite --no-copy <task>` generates the implementation prompt.
- `ph review <repo> --codex-output <temp> <task>` runs deterministic review.
- `ph review-codex <repo> --codex-output <temp> --no-copy <task>` generates a Codex reviewer prompt.

Temporary files are UTF-8 Markdown/text files created internally by the app and deleted after the command completes. The user does not manage temp files.

The app should prefer the installed `ph` command. If it is unavailable during local development, it may fall back to `uv run ph` from `libs/deepagents`.

## Validation

MVP validation requires:

- App builds.
- Repo selector includes `FinanciacionCore` when the existing `ph repos` aliases expose it.
- A long Spanish task can be pasted or written without mojibake.
- `Generate Codex Prompt` produces a prompt and copies it to the clipboard.
- A pasted Codex output can be reviewed deterministically.
- Review output shows `PASS` or `REVIEW_NEEDED`.
- Codex reviewer prompt generation works.
- No Codex/model execution occurs.
