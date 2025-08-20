### Related Issues/PRs

<!-- Addresses reviewer feedback from original PR #17246 -->

### What changes are proposed in this pull request?

This PR introduces a new `mlflow traces` CLI module that provides comprehensive trace management capabilities for MLflow users. The implementation adds powerful search, filtering, and management operations for traces stored in MLflow experiments.

#### Core Features

**🔍 Trace Search & Discovery**
- Search traces across experiments with advanced filtering options
- Support for complex field selection using JSONPath-like dot notation with backtick escaping for dotted field names (e.g., `info.tags.`mlflow.traceName``)
- Wildcard support for exploring nested structures (`data.spans.*.name`)
- Multiple output formats (table/JSON) with customizable field selection
- Verbose mode (`--verbose` flag) for detailed field validation errors

**📊 Schema Documentation** 
- Detailed trace schema documentation built into CLI help
- Examples for field selection patterns and common use cases
- Clear documentation of trace structure (info, data, spans, assessments)

**🏷️ Assessment Management**
- Log feedback scores and expectations to traces
- Get, update, and delete assessments with full metadata support
- Support for human annotations, LLM judgments, and automated evaluations

**🏷️ Tag Operations**
- Set and delete custom tags on traces
- Support for both user-defined and system tags

**🗑️ Cleanup Operations**
- Delete traces from multiple experiments with proper iteration
- Bulk operations for efficient trace management

#### Available Commands

- `mlflow traces search` - Search traces with filtering, sorting, and field selection
- `mlflow traces get` - Retrieve detailed trace information as JSON
- `mlflow traces delete` - Delete traces by ID or timestamp criteria
- `mlflow traces set-tag` / `delete-tag` - Add/remove tags from traces
- `mlflow traces log-feedback` / `log-expectation` - Log evaluation feedback/scores
- `mlflow traces get-assessment` / `update-assessment` / `delete-assessment` - Manage assessments

#### Technical Implementation

- Lightweight JSONPath implementation without external dependencies
- Efficient field extraction and validation with backtick support for dotted field names
- Structure-preserving filtering when extracting specific fields
- Proper CLI structure in `mlflow/cli/` directory
- Table formatting with smart value display (timestamps, durations, etc.)

### How is this PR tested?

- [x] Existing unit/integration tests
- [x] New unit/integration tests  
- [x] Manual tests

**New Tests Added:**
- Complete CLI test suite: 9 tests covering all major functionality
- JSONPath utilities test suite: 25 tests covering field extraction, wildcards, backtick escaping
- Parametrized tests for better maintainability

**Manual Testing:**
- Verified with real Databricks traces using environment variables
- Tested backtick functionality: `--extract-fields "info.trace_id,info.tags.`mlflow.traceName`"`
- Confirmed both JSON and table output formats work correctly
- Validated verbose mode error reporting

### Does this PR require documentation update?

- [x] Yes. I've updated:
  - [x] Examples - Comprehensive examples in CLI help text and command documentation
  - [x] API references - Complete schema documentation built into CLI help
  - [x] Instructions - Clear usage instructions and field selection patterns

### Release Notes

#### Is this a user-facing change?

- [x] Yes. This PR adds a comprehensive new CLI module for MLflow traces management.

This change introduces the `mlflow traces` CLI command with powerful search, filtering, field selection, assessment management, and cleanup capabilities for MLflow traces. Users can now manage traces entirely from the command line with advanced features like JSONPath-style field selection, backtick escaping for dotted field names, verbose error reporting, and multiple output formats.

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [x] `area/tracing`: MLflow Tracing features, tracing APIs, and LLM tracing functionality  
- [x] `area/uiux`: Front-end, user experience (CLI interface improvements)

#### How should the PR be classified in the release notes? Choose one:

- [x] `rn/feature` - A new user-facing feature worth mentioning in the release notes

#### Should this PR be included in the next patch release?

- [ ] No (this PR will be included in the next minor release)

This is a new feature that adds substantial CLI functionality, so it should go in a minor release rather than a patch release.
