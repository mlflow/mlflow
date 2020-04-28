
This document is a hands-on manual for doing issue and pull request triage for `MLflow issues 
on GitHub <https://github.com/mlflow/mlflow/issues>`_ . 
The purpose of triage is to speed up issue management and get community members faster responses. 

Issue and pull request triage has three steps:

- assign one or more process labels (e.g. ``needs design`` or ``help wanted``)
- mark a priority 
- label one or more relevant areas, languauges, or integrations to help route issues to appropriate contributors or reviewers

The remainder of the document describes the labels used in each of these steps and how to apply them.

Assign appropriate process labels
#######
Assign at least one process label to every issue you triage. 

- ``needs author feedback``: We need input from the author of the issue or PR to proceed.
- | ``needs design``: This feature is large or tricky enough that we think it warrants design doc 
  | and review before someone begins implementation.
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
  | may need multiple releases to complete. Used for things we believe will happen in our
  | roadmap beyond the quarter, or we know is on the roadmap but not actively worked on for a 
  | contributing company. We can use this with ``help wanted`` to mark issues we would like to 
  | get help from. If someone starts working on it actively and we think it could make the next 
  | release, change the priority to ``priority/important-soon``.
- | ``priority/backlog``: We believe it is useful, but don’t see it being prioritized in the 
  | next few months. Use this when it’s lower priority than ``priority/important-longterm``. We'd 
  | welcome someone in the community picking it up, but there may be some delay in getting support 
  | through design review or pull request feedback. 
- | ``priority/awaiting-more-evidence``: Lowest priority. Possibly useful, but not yet enough
  | support to actually get it done. This is a good place to put issues that could be useful but 
  | that we don’t have clear enough evidence it has broad value. Don’t use it as a way to say no. 
  | If we think it doesn’t fit in MLflow, we should just say that and why.

Label relevant areas
#######

Assign one more labels for relevant component or interface surface areas, languages, or 
integrations. As a principle, we aim to have the minimal set of labels needed to help route issues
and PRs to appropriate contributors. For example, ``language/python`` is unnecessary as most PRs 
will involve python code, and so we don't need specific people to take a look at that PR. However,
``language/java`` and ``language/r`` make sense to have as the clients in these languages are 
differ from the python client and aren't maintained by many people. As with process lables, we
take inspiration from kubernetes on naming conventions.

Components 
""""""""
- ``area/artifacts``: Artifact stores and artifact logging
- ``area/docs``: MLflow documentation pages
- ``area/examples``: Example code
- | ``area/model-registry``: Model registry, model registry APIs, and the fluent client calls for
  | model registry 
- ``area/models``: MLmodel format, model serialization/deserialization, flavors
- ``area/projects``: MLproject format, project running backends
- ``area/scoring``: Local serving, model deployment tools, spark UDFs
- ``area/tracking``: Tracking service, tracking client APIs, autologging

Interface Surface
""""""""
- ``area/uiux``: Front-end, user experience, javascript, plotting
- ``area/docker``: Docker use anywhere, such as MLprojects and MLmodels
- ``area/sqlalchemy``: Use of SQL alchemy in tracking service or model registry
- ``area/windows``: Windows support

Language Surface
""""""""
- ``language/r``: R APIs and clients
- ``language/java``: Java APIs and clients

Integrations
""""""""
- ``integrations/azure``: Azure and Azure ML integrations
- ``integrations/sagemaker``: Sagemaker integrations
