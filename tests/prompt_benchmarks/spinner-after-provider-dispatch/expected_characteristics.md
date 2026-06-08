# Expected Characteristics

## Task mode
- provider_api_bug

## Required skills
- provider_api_bug
- regression_fix
- implementation_fix

## Forbidden skills
- provider_bootstrap_diagnostic
- form_security_autofill_bug

## English labels
- required

## Required prompt text
- The visible UI remains in a spinner/loading state around the provider-hosted send flow.
- Provider dispatch success means the provider accepted the request; it must not be treated as workflow completion.
- Provider dispatch success only proves provider acceptance; it does not prove signature completion or workflow completion.
- Protect config, payload, and dispatch behavior unless the task explicitly targets them.
- Do not modify ConfigId or SET_CONFIG handling unless explicitly targeted.
- Verify provider dispatch success remains intact and the workflow state is not incorrectly marked complete.

## Forbidden prompt text
- Original user task:
- - Files read
- - Files changed
