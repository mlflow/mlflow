<!-- 🚨 We recommend pull requests be filed from a non-master branch on a repository fork (e.g. <username>:fix-xxx). 🚨 -->

## Related Issues/PRs

<!--
Please reference any related feature requests, issues, or PRs here. For example, `#123`. To automatically close the referenced items when this PR is merged, please use a closing keyword (close, fix, or resolve). For example, `Close #123`. See https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue for more information.
-->
#xxx

## What changes are proposed in this pull request?

(Please fill in changes proposed in this fix)

## How is this patch tested?

<!--
If you're unsure about what to test, where to add tests, or how to run tests, please feel free to ask. We'd be happy to help.
-->
- [ ] I have written tests (not required for typo or doc fix) and confirmed the proposed feature/bug-fix/change works.

<!--
Please describe how you confirmed the proposed feature/bug-fix/change works here. For example, if you fixed an MLflow client API, you could attach the code that didn't work prior to the fix but works now, or if you added a new feature on MLflow UI, you could attach a video that demonstrates the feature.
-->

## Does this PR change the documentation?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. Make sure the changed pages / sections render correctly by following the steps below.

1. Check the status of the `ci/circleci: build_doc` check. If it's successful, proceed to the
   next step, otherwise fix it.
2. Click `Details` on the right to open the job page of CircleCI.
3. Click the `Artifacts` tab.
4. Click `docs/build/html/index.html`.
5. Find the changed pages / sections and make sure they render correctly.

## Release Notes

### Is this a user-facing change?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. Give a description of this change to be included in the release notes for MLflow users.

(Details in 1-2 sentences. You can just refer to another PR with a description if this PR is part of a larger change.)

### What component(s), interfaces, languages, and integrations does this PR affect?
Components
- [ ] `area/artifacts`: Artifact stores and artifact logging
- [ ] `area/build`: Build and test infrastructure for MLflow
- [ ] `area/docs`: MLflow documentation pages
- [ ] `area/examples`: Example code
- [ ] `area/model-registry`: Model Registry service, APIs, and the fluent client calls for Model Registry
- [ ] `area/models`: MLmodel format, model serialization/deserialization, flavors
- [ ] `area/pipelines`: Pipelines, Pipeline APIs, Pipeline configs, Pipeline Templates
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
### How should the PR be classified in the release notes? Choose one:

- [ ] `rn/breaking-change` - The PR will be mentioned in the "Breaking Changes" section
- [ ] `rn/none` - No description will be included. The PR will be mentioned only by the PR number in the "Small Bugfixes and Documentation Updates" section
- [ ] `rn/feature` - A new user-facing feature worth mentioning in the release notes
- [ ] `rn/bug-fix` - A user-facing bug fix worth mentioning in the release notes
- [ ] `rn/documentation` - A user-facing documentation change worth mentioning in the release notes
