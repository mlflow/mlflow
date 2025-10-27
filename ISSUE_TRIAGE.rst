
This document is a hands-on manual for doing issue and pull request triage for `MLflow issues
on GitHub <https://github.com/mlflow/mlflow/issues>`_ .
The purpose of triage is to speed up issue management and get community members faster responses.

Issue and pull request triage has three steps:

- assign one or more process labels (e.g. ``needs design`` or ``help wanted``),
- mark a priority, and
- label one or more relevant areas, languages, or integrations to help route issues to appropriate contributors or reviewers.

The remainder of the document describes the labels used in each of these steps and how to apply them.

Assign appropriate process labels
#######
Assign at least one process label to every issue you triage.

- ``needs author feedback``: We need input from the author of the issue or PR to proceed.
- | ``needs design``: This feature is large or tricky enough that we think it warrants a design doc
  | and review before someone begins implementation.
- | ``needs committer feedback``: The issue has a design that is ready for committer review, or there is
  | an issue or pull request that needs feedback from a committer about the approach or appropriateness
  | of the contribution.
- | ``needs review``: Use this label for issues that need a more detailed design review or pull
  | requests ready for review (all questions answered, PR updated if requests have been addressed,
  | tests passing).
- ``help wanted``: We would like community help for this issue.
- ``good first issue``: This would make a good first issue.


Assign priority
#######

You should assign a priority to each issue you triage. We use `kubernetes-style <https://github.com/
kubernetes/community/blob/master/contributors/guide/issue-triage.md#define-priority>`_ priority
labels.

- | ``priority/critical-urgent``: This is the highest priority and should be worked on by
  | somebody right now. This should typically be reserved for things like security bugs,
  | regressions, release blockers.
- | ``priority/important-soon``: The issue is worked on by the community currently or will
  | be very soon, ideally in time for the next release.
- | ``priority/important-longterm``: Important over the long term, but may not be staffed or
  | may need multiple releases to complete. Also used for things we know are on a
  | contributor's roadmap in the next few months. We can use this in conjunction with
  | ``help wanted`` to mark issues we would like to get help with. If someone begins actively
  | working on an issue with this label and we think it may be merged by the next release, change
  | the priority to ``priority/important-soon``.
- | ``priority/backlog``: We believe it is useful but don't see it being prioritized in the
  | next few months. Use this for issues that are lower priority than ``priority/important-longterm``.
  | We welcome community members to pick up a ``priority/backlog`` issue, but there may be some
  | delay in getting support through design review or pull request feedback.
- | ``priority/awaiting-more-evidence``: Lowest priority. Possibly useful, but not yet enough
  | support to actually get it done. This is a good place to put issues that could be useful but
  | require more evidence to demonstrate broad value. Don't use it as a way to say no.
  | If we think it doesn't fit in MLflow, we should just say that and why.

Label relevant areas
#######

Assign one more labels for relevant component or interface surface areas, languages, or
integrations. As a principle, we aim to have the minimal set of labels needed to help route issues
and PRs to appropriate contributors. For example, a ``language/python`` label would not be
particularly helpful for routing issues to committers, since most PRs involve Python code.
``language/java`` and ``language/r`` make sense to have, as the clients in these languages differ from the Python client and aren't maintained by many people. As with process labels, we
take inspiration from Kubernetes on naming conventions.

Components
""""""""
- ``area/artifacts``: Artifact stores and artifact logging
- ``area/build``: Build and test infrastructure for MLflow
- ``area/docs``: MLflow documentation pages
- ``area/evaluation``: MLflow model evaluation features, evaluation metrics, and evaluation workflows
- ``area/examples``: Example code
- ``area/gateway``: AI Gateway service, Gateway client APIs, third-party Gateway integrations
- ``area/model-registry``: Model Registry service, APIs, and the fluent client calls for Model Registry
- ``area/models``: MLmodel format, model serialization/deserialization, flavors
- ``area/projects``: MLproject format, project execution backends
- ``area/prompt``: MLflow prompt engineering features, prompt templates, and prompt management
- ``area/scoring``: MLflow Model server, model deployment tools, Spark UDFs
- ``area/server-infra``: MLflow Tracking server backend
- ``area/tracing``: MLflow Tracing features, tracing APIs, and LLM tracing functionality
- ``area/tracking``: Tracking Service, tracking client APIs, autologging

Interface Surface
""""""""
- ``area/uiux``: Front-end, user experience, plotting, JavaScript, JavaScript dev server
- ``area/docker``: Docker use across MLflow's components, such as MLflow Projects and MLflow Models
- ``area/sqlalchemy``: Use of SQLAlchemy in the Tracking Service or Model Registry
- ``area/windows``: Windows support

Language Surface
""""""""
- ``language/r``: R APIs and clients
- ``language/java``: Java APIs and clients
- ``language/new``: Proposals for new client languages

Integrations
""""""""
- ``integrations/azure``: Azure and Azure ML integrations
- ``integrations/sagemaker``: SageMaker integrations
- ``integrations/databricks``: Databricks integrations
