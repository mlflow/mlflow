
### Related Issues/PRs

<!-- Uncomment 'Resolve' if this PR can close the linked items. -->
<!-- Resolve --> 

### What changes are proposed in this pull request?

This PR centralizes the logic for unwrapping LLM input data when it is wrapped in an `{"inputs": ...}` dictionary.

Previously, this unwrapping logic was implemented in `mlflow.llama_index.pyfunc_wrapper` with a TODO to migrate it to a more central location. This PR moves that logic into `_convert_llm_input_data` within `mlflow.models.utils`, making it available for other flavors and ensuring consistent behavior.

Specific changes:
- Updated `mlflow.models.utils._convert_llm_input_data` to check for and unwrap the `inputs` key if present in a dictionary.
- Removed the temporary `_convert_llm_input_data_with_unwrapping` function from `mlflow.llama_index.pyfunc_wrapper.py` and replaced its usage with `_convert_llm_input_data`.
- Verified consistent behavior for nested dictionaries and lists through manual testing.

### How is this PR tested?

- [ ] Existing unit/integration tests
- [ ] New unit/integration tests
- [x] Manual tests

Verified manually with a script to ensure:
- `{"inputs": "value"}` -> `"value"`
- `{"inputs": {"nested": "dict"}}` -> `{"nested": "dict"}`
- `{"inputs": ["list"]}` -> `["list"]`
- `{"other": "value"}` -> `{"other": "value"}` (unchanged)

### Does this PR require documentation update?

- [x] No. You can skip the rest of this section.
- [ ] Yes. I've updated:
  - [ ] Examples
  - [ ] API references
  - [ ] Instructions

### Release Notes

#### Is this a user-facing change?

- [ ] No. You can skip the rest of this section.
- [x] Yes. Give a description of this change to be included in the release notes for MLflow users.

Refactor input handling utility to consistently unwrap dictionary inputs containing an `inputs` key across flavors, resolving a technical debt in LlamaIndex pyfunc wrapper.

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [ ] `area/tracking`: Tracking Service, tracking client APIs, autologging
- [x] `area/models`: MLmodel format, model serialization/deserialization, flavors
- [ ] `area/model-registry`: Model Registry service, APIs, and the fluent client calls for Model Registry
- [x] `area/scoring`: MLflow Model server, model deployment tools, Spark UDFs
- [ ] `area/evaluation`: MLflow model evaluation features, evaluation metrics, and evaluation workflows
- [ ] `area/gateway`: MLflow AI Gateway client APIs, server, and third-party integrations
- [ ] `area/prompts`: MLflow prompt engineering features, prompt templates, and prompt management
- [ ] `area/tracing`: MLflow Tracing features, tracing APIs, and LLM tracing functionality
- [ ] `area/projects`: MLproject format, project running backends
- [ ] `area/uiux`: Front-end, user experience, plotting, JavaScript, JavaScript dev server
- [ ] `area/build`: Build and test infrastructure for MLflow
- [ ] `area/docs`: MLflow documentation pages

<a name="release-note-category"></a>

#### How should the PR be classified in the release notes? Choose one:

- [x] `rn/none` - No description will be included. The PR will be mentioned only by the PR number in the "Small Bugfixes and Documentation Updates" section
- [ ] `rn/breaking-change` - The PR will be mentioned in the "Breaking Changes" section
- [ ] `rn/feature` - A new user-facing feature worth mentioning in the release notes
- [ ] `rn/bug-fix` - A user-facing bug fix worth mentioning in the release notes
- [ ] `rn/documentation` - A user-facing documentation change worth mentioning in the release notes

#### Should this PR be included in the next patch release?

- [x] Yes (this PR will be cherry-picked and included in the next patch release)
- [ ] No (this PR will be included in the next minor release)
