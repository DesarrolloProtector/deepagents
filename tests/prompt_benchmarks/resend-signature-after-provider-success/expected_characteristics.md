# Expected Characteristics

## Task mode
- implementation_fix

## Required skills
- operational_workflow_convergence
- provider_api_bug
- regression_fix
- implementation_fix

## Forbidden skills
- provider_bootstrap_diagnostic
- ui_runtime_bug

## English labels
- required

## Required prompt text
- A signature send/resend action is missing even though the workflow still needs operator action.
- The operator should have the correct next action while the workflow remains pending.
- Provider dispatch success only proves provider acceptance; it does not prove signature completion or workflow completion.
- Preserve ProviderStatus/ProviderCorrelationId handling that already records successful dispatch.
- Keep "Ver estado firma" working.
- Verify the signature send/resend action renders when the signature remains pending.

## Forbidden prompt text
- Original user task:
- - Files read
- - Files changed
