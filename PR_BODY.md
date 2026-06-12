### Related Issues/PRs

Relates to #21811
Relates to #22697

### What changes are proposed in this pull request?

This PR implements numeric comparison operators for ASSESSMENT-based trace search filters so backend filtering supports the operators already shipped in the UI PR #22697.

- Adds `>`, `>=`, `<`, and `<=` to valid feedback/expectation assessment comparators.
- Parses unquoted integer and float literals for numeric feedback/expectation comparisons while preserving existing quoted-string behavior for `=`, `!=`, `LIKE`, `ILIKE`, `RLIKE`, `IS NULL`, and `IS NOT NULL`.
- Raises a clear `MlflowException` when a numeric assessment comparator is paired with a non-numeric literal, instead of silently falling back to string comparison.
- Casts JSON-serialized assessment values to numeric at query time for numeric operators, with guards that make JSON strings, objects, arrays, booleans, and null evaluate as `NULL` and therefore not match numeric comparisons.
- Implements dialect-aware SQL generation for SQLite, Postgres, MySQL, and MSSQL, based on the prototype in #21811. The phase-1 benchmark returned GO: runtime CAST-to-float adds a stable, bounded ~2.2x overhead across 10k/100k/1M rows, with inequality operators showing the same overhead as `=` at equal selectivity.

### How is this PR tested?

- [x] Existing unit/integration tests
- [x] New unit/integration tests

SQLite/local test coverage:

```bash
.venv/bin/python -m pytest tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py::test_search_traces_with_assessment_numeric_filters tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py::test_search_traces_with_assessment_numeric_filters_is_workspace_scoped
```

Result: `3 passed, 1 warning in 2.32s`.

```bash
.venv/bin/python -m pytest tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py::test_search_traces_with_feedback_and_expectation_filters tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py::test_search_traces_with_assessment_numeric_filters tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py::test_search_traces_is_workspace_scoped tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py::test_assessment_operations_are_workspace_scoped tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py::test_search_traces_with_assessment_numeric_filters_is_workspace_scoped
```

Result: `7 passed, 1 warning in 2.08s`.

```bash
.venv/bin/python -m ruff check mlflow/utils/search_utils.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py
.venv/bin/python -m ruff format --check mlflow/utils/search_utils.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py
.venv/bin/python -m pre_commit run --files mlflow/utils/search_utils.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_traces.py tests/store/tracking/sqlalchemy_store/test_sqlalchemy_workspace_store.py
```

Results: Ruff passed, format check passed, pre-commit passed.

New tests cover feedback and expectation numeric filters with all four operators, boundary values, integer and float literals, combined numeric/string filters, non-numeric stored values being excluded without query-time errors, malformed literals raising `MlflowException`, and a workspace-aware variant that verifies workspace scoping.

Only SQLite tests were run locally; SQL for Postgres/MySQL/MSSQL was compile-checked carefully.

### Does this PR require documentation update?

- [x] No.

### Does this PR require updating the [MLflow Skills](https://github.com/mlflow/skills) repository?

- [x] No.

### Release Notes

#### Is this a user-facing change?

- [x] Yes. Give a description of this change to be included in the release notes for MLflow users.

MLflow trace search now supports numeric assessment filters using `>`, `>=`, `<`, and `<=` for feedback and expectation values.

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [x] `area/tracking`: Tracking Service, tracking client APIs, autologging
- [x] `area/tracing`: MLflow Tracing features, tracing APIs, and LLM tracing functionality

<a name="release-note-category"></a>

#### How should the PR be classified in the release notes? Choose one:

- [x] `rn/feature` - A new user-facing feature worth mentioning in the release notes

#### Is this PR a critical bugfix or security fix that should go into the next patch release?

<details>
<summary>What is a minor/patch release?</summary>

- Minor release: a release that increments the second part of the version number (e.g., 1.2.0 -> 1.3.0).
  Minor releases are expected to contain larger changes, such as new features and improvements. Non-critical bug fixes and doc updates can be included as well. By default, your PR should target the next minor release.
- Patch release: a release that increments the third part of the version number (e.g., 1.2.1).
  Patch releases are typically only performed when there has been a major regression or bug in the latest release. For the sake of stability, your PR should not be included in a patch release unless it is a critical fix, or if the risk level of your PR is exceedingly low.

</details>

- [ ] This PR is critical and needs to be in the next patch release
- [x] This PR can wait for the next minor release
