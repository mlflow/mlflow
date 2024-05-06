### Related Issues/PRs

<!-- Uncomment 'Resolve' if this PR can close the linked items. -->
<!-- Resolve --> #xxx

### What changes are proposed in this pull request?

<!-- Please fill in changes proposed in this PR. -->

### How is this PR tested?

- [ ] Existing unit/integration tests
- [ ] New unit/integration tests
- [ ] Manual tests

<!-- Attach code, screenshot, video used for manual testing here. -->

### Does this PR require documentation update?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. I've updated:
  - [ ] Examples
  - [ ] API references
  - [ ] Instructions

### Release Notes

#### Is this a user-facing change?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. Give a description of this change to be included in the release notes for MLflow users.

<!-- Details in 1-2 sentences. You can just refer to another PR with a description if this PR is part of a larger change. -->

#### What component(s), interfaces, languages, and integrations does this PR affect?

Components

- [ ] `area/artifacts`: Artifact stores and artifact logging
- [ ] `area/build`: Build and test infrastructure for MLflow
- [ ] `area/deployments`: MLflow Deployments client APIs, server, and third-party Deployments integrations
- [ ] `area/docs`: MLflow documentation pages
- [ ] `area/examples`: Example code
- [ ] `area/model-registry`: Model Registry service, APIs, and the fluent client calls for Model Registry
- [ ] `area/models`: MLmodel format, model serialization/deserialization, flavors
- [ ] `area/recipes`: Recipes, Recipe APIs, Recipe configs, Recipe Templates
- [ ] `area/projects`: MLproject format, project running backends
- [ ] `area/scoring`: MLflow Model server, model deployment tools, Spark UDFs
- [ ] `area/server-infra`: MLflow Tracking server backend
- [ ] `area/tracking`: Tracking Service, tracking client APIs, autologging

Interface

- [ ] `area/uiux`: Front-end, user experience, plotting, JavaScript, JavaScript dev server
- [ ] `area/docker`: Docker use across MLflow's components, such as MLflow Projects and MLflow Models
- [ ] `area/sqlalchemy`: Use of SQLAlchemy in the Tracking Service or Model Registry
- [ ] `area/windows`: Windows support

Language

- [ ] `language/r`: R APIs and clients
- [ ] `language/java`: Java APIs and clients
- [ ] `language/new`: Proposals for new client languages

Integrations

- [ ] `integrations/azure`: Azure and Azure ML integrations
- [ ] `integrations/sagemaker`: SageMaker integrations
- [ ] `integrations/databricks`: Databricks integrations

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

#### Should this PR be included in the next patch release?

`Yes` should be selected for bug fixes, documentation updates, and other small changes. `No` should be selected for new features and larger changes. If you're unsure about the release classification of this PR, leave this unchecked to let the maintainers decide.

<details>
<summary>What is a minor/patch release?</summary>

- Minor release: a release that increments the second part of the version number (e.g., 1.2.0 -> 1.3.0).
  Bug fixes, doc updates and new features usually go into minor releases.
- Patch release: a release that increments the third part of the version number (e.g., 1.2.0 -> 1.2.1).
  Bug fixes and doc updates usually go into patch releases.

</details>

<!-- patch -->

- [ ] Yes (this PR will be cherry-picked and included in the next patch release)
- [ ] No (this PR will be included in the next minor release)
