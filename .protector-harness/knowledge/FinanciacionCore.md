# FinanciacionCore

## Current phase

- MVP convergence toward a usable SaaS as soon as possible.

## Current priorities

- Legacy onboarding reachability.
- Accounting final gaps.
- Visual fixes.
- Languages.
- Telemetry.
- Resilience.

## Authoritative workflows

- Company and financer onboarding should be contract-first.
- Promoted normal UI flows should create and review contracts before activation.

## Protected decisions

- Legacy direct routes may remain backend-compatible.
- Legacy direct routes should not be promoted in normal UI.
- Avoid broad audits unless explicitly requested.

## Forbidden directions

- Do not touch Contabilidad unless the task targets it.
- Do not touch telemetry unless the task targets it.
- Do not touch resilience unless the task targets it.

## Known useful routes/views/tests

- /Master/NewCompany -> Master.js -> CreatePlatformCompanyContract.
- PlatformCompanyContract pending review before Company/User/BankData activation.
- Route-backed onboarding tests should prove the contract-first flow.
