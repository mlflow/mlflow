<!--
Do not remove any sections. Remove unused checkboxes except for the patch release section at the bottom.
Use backticks for code references and file paths in the PR title (e.g., `ClassName`, `function_name`, `utils.py`).
-->

### Related Issues/PRs

<!-- 🚨 Choose either "Closes" (auto-close on merge) or "Relates to" (reference only). 🚨 -->

Closes | Relates to #issue_number

### What changes are proposed in this pull request?

<!-- Please fill in changes proposed in this PR. -->

### How is this PR tested?

- [ ] Existing unit/integration tests
- [ ] New unit/integration tests
- [ ] Manual tests

<!-- Attach code, screenshot, video used for manual testing here. -->

### Does this PR require documentation update?

- [ ] No.
- [ ] Yes. I've updated:
  - [ ] Examples
  - [ ] API references
  - [ ] Instructions

### Does this PR require updating the [MLflow Skills](https://github.com/mlflow/skills) repository?

<!-- When updating APIs or feature usage, please ensure the MLflow Skills repository reflects those changes. -->

- [ ] No.
- [ ] Yes. Please link the corresponding PR or explain how you plan to update it.

<!-- Provide the link to the Skills repository PR or a brief explanation of the changes needed. -->

### Release Notes

#### Is this a user-facing change?

- [ ] No.
- [ ] Yes. Give a description of this change to be included in the release notes for MLflow users.

<!-- Details in 1-2 sentences. You can just refer to another PR with a description if this PR is part of a larger change. -->

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [ ] `area/tracking`: Tracking Service, tracking client APIs, autologging
- [ ] `area/models`: MLmodel format, model serialization/deserialization, flavors
- [ ] `area/model-registry`: Model Registry service, APIs, and the fluent client calls for Model Registry
- [ ] `area/scoring`: MLflow Model server, model deployment tools, Spark UDFs
- [ ] `area/evaluation`: MLflow model evaluation features, evaluation metrics, and evaluation workflows
- [ ] `area/gateway`: MLflow AI Gateway client APIs, server, and third-party integrations
- [ ] `area/prompts`: MLflow prompt engineering features, prompt templates, and prompt management
- [ ] `area/tracing`: MLflow Tracing features, tracing APIs, and LLM tracing functionality
- [ ] `area/projects`: MLproject format, project running backends
- [ ] `area/uiux`: Front-end, user experience, plotting, JavaScript, JavaScript dev server
- [ ] `area/build`: Build and test infrastructure for MLflow
- [ ] `area/docs`: MLflow documentation pages

<!--
Insert an empty named anchor here to allow jumping to this section with a fragment URL
(e.g. https://github.com/mlflow/mlflow/pull/123#user-content-release-note-category).
Note that GitHub prefixes anchor names in markdown with "user-content-".
-->

<a name="release-note-category"></a>

#### How should the PR be classified in the release notes? Choose one:

- [ ] `rn/none` - No description will be included. The PR will be mentioned only by the PR number in the "Small Bugfixes and Documentation Updates" section
- [ ] `rn/breaking-change` - The PR will be mentioned in the "Breaking Changes" section
- [ ] `rn/feature` - A new user-facing feature worth mentioning in the release notes
- [ ] `rn/bug-fix` - A user-facing bug fix worth mentioning in the release notes
- [ ] `rn/documentation` - A user-facing documentation change worth mentioning in the release notes

#### Is this PR a critical bugfix or security fix that should go into the next patch release?

<details>
<summary>What is a minor/patch release?</summary>

- Minor release: a release that increments the second part of the version number (e.g., 1.2.0 -> 1.3.0).
  Minor releases are expected to contain larger changes, such as new features and improvements. Non-critical bug fixes and doc updates can be included as well. By default, your PR should target the next minor release.
- Patch release: a release that increments the third part of the version number (e.g., 1.2.0 -> 1.2.1).
  Patch releases are typically only performed when there has been a major regression or bug in the latest release. For the sake of stability, your PR should not be included in a patch release unless it is a critical fix, or if the risk level of your PR is exceedingly low.

</details>

<!-- Do not modify or remove any text inside the parentheses. Keep both checkboxes below. -->

- [ ] This PR is critical and needs to be in the next patch release
- [ ] This PR can wait for the next minor release
