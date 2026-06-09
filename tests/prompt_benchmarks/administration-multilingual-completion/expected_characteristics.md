# Expected Characteristics

## Task mode
- implementation_fix

## Required skills
- localization_completion
- implementation_fix

## Forbidden skills
- navigation_surface_convergence
- provider_bootstrap_diagnostic

## English labels
- required

## Required prompt text
- The selected UI area has incomplete ES/EN localization or is missing the promoted language selector.
- The selected area should complete visible user-facing ES/EN text through existing i18n/localizer infrastructure.
- Use the existing i18n/localizer/resource pattern; add ES/EN resources where needed instead of relocalizing from scratch.
- Inspect only the normal UI-reachable area and preserve the current page/context where practical.
- Add a visible language selector to promoted pages when missing.
- Do not change routes, forms, handlers, permissions, submitted values, business logic, persistence, or workflows.
- Verify the selected area renders localized titles, buttons, labels, headers, empty states, validation/errors, confirmations, navigation, and language selector behavior.

## Forbidden prompt text
- Original user task:
- contract-first onboarding
- Legacy direct routes
- Do not touch Contabilidad, telemetry, or resilience unless targeted.
- - Files read
- - Files changed
