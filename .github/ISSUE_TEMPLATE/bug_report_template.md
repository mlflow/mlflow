---
name: Bug Report
about: Use this template for reporting bugs encountered while using MLflow.
labels: 'bug'
title: "[BUG]"
---
Thank you for submitting an issue. Please refer to our [issue policy](https://www.github.com/mlflow/mlflow/blob/master/ISSUE_POLICY.md) for additional information about bug reports. For help with debugging your code, please refer to [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).

**Please fill in this bug report template to ensure a timely and thorough response.**

### Willingness to contribute
The MLflow Community encourages bug fix contributions. Would you or another member of your organization be willing to contribute a fix for this bug to the MLflow code base?

- [ ] Yes. I can contribute a fix for this bug independently.
- [ ] Yes. I would be willing to contribute a fix for this bug with guidance from the MLflow community.
- [ ] No. I cannot contribute a bug fix at this time.

### System information
- **Have I written custom code (as opposed to using a stock example script provided in MLflow)**:
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:
- **MLflow installed from (source or binary)**:
- **MLflow version (run ``mlflow --version``)**:
- **Python version**:
- **npm version, if running the dev UI**:
- **Exact command to reproduce**:

### Describe the problem
Describe the problem clearly here. Include descriptions of the expected behavior and the actual behavior.

### Code to reproduce issue
Provide a reproducible test case that is the bare minimum necessary to generate the problem.

### Other info / logs
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.


### What component(s), interfaces, languages, and integrations does this bug affect?
Components 
- [ ] `area/artifacts`: Artifact stores and artifact logging
- [ ] `area/build`: Build and test infrastructure for MLflow
- [ ] `area/docs`: MLflow documentation pages
- [ ] `area/examples`: Example code
- [ ] `area/model-registry`: Model Registry service, APIs, and the fluent client calls for Model Registry
- [ ] `area/models`: MLmodel format, model serialization/deserialization, flavors
- [ ] `area/projects`: MLproject format, project running backends
- [ ] `area/scoring`: Local serving, model deployment tools, spark UDFs
- [ ] `area/server-infra`: MLflow server, JavaScript dev server
- [ ] `area/tracking`: Tracking Service, tracking client APIs, autologging

Interface 
- [ ] `area/uiux`: Front-end, user experience, JavaScript, plotting
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
