# Git workflow rules

## Repository

- Remote repository: `https://github.com/SENATOROVAI/applied-ai-cv-project.git`.
- Main branch: `main`.
- `main` should contain a reproducible, readable state of the project.

## Commit rules

- Make commits after a small, checkable block of work is complete.
- Keep commits thematic: do not mix unrelated documentation, source code, data preparation, training results, and report edits in one commit.
- Use short imperative messages, for example:
  - `add project git workflow`
  - `add data contracts`
  - `add practice cli skeleton`
  - `add tests for artifact storage`
- Before committing code, run the relevant fast checks when they exist. If checks cannot be run, mention that in the final handoff for the task.
- Do not commit raw datasets, prepared datasets, checkpoints, model weights, generated reports, local environments, caches, or machine-specific files.
- Commit source notebooks, project instructions, requirements, specifications, plans, configuration files, source code, tests, and small deterministic fixtures.

## Push rules

- Push after each finished and checked milestone, not after every tiny edit.
- Push before starting expensive or long-running work such as dataset download, smoke-run, full training, or PDF generation.
- Push immediately after important documentation milestones: requirements, specification, implementation plan, and accepted workflow rules.
- During implementation, prefer 2-4 meaningful commits per large stage instead of one very large commit.
- If a stage is blocked, commit only the useful completed part and describe the blocker in the task handoff.

## Suggested milestone rhythm

1. Project setup and repository workflow.
2. Configurations, data contracts, and artifact storage.
3. Base modules, CLI, and error handling.
4. Unit tests and synthetic integration checks.
5. Real data preparation, after explicit approval.
6. Smoke-runs, after data is available.
7. Full training and evaluation.
8. Final notebooks and PDF reports.

