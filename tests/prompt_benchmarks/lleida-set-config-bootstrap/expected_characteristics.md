# Expected Characteristics

## Task mode
- diagnostic_bootstrap

## Required skills
- provider_bootstrap_diagnostic

## Forbidden skills
- provider_api_bug
- ui_runtime_bug

## English labels
- required

## Required prompt text
- Provider bootstrap/configuration needs bounded verification before changing runtime workflow behavior.
- Diagnostics should prove provider configuration and bootstrap state without treating acceptance as workflow completion.
- Limit work to explicit provider bootstrap, config, SET_CONFIG, or smoke-diagnostic checks.
- Do not modify provider payloads or workflow lifecycle unless the diagnostic proves that exact target.
- Report exact bounded diagnostic/config evidence and whether provider calls were made.
- Diagnostics run

## Forbidden prompt text
- Original user task:
- - Files read
- - Files changed
