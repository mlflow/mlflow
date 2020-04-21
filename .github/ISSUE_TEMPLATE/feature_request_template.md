---
name: Feature Request
about: Use this template for feature and enhancement proposals.
labels: 'enhancement'
title: "[FR]"
---
Thank you for submitting a feature request. **Before proceeding, please refer to our [issue policy](https://www.github.com/mlflow/mlflow/blob/master/ISSUE_POLICY.md) for guidelines and information about the feature request lifecycle**.

**Please fill in this template. Feature requests that do not complete this template will not be reviewed.**

## Willingness to contribute
The MLflow Community encourages new feature contributions. Would you or another member of your organization be willing to contribute an implementation of this feature (either as an MLflow Plugin or an enhancement to the MLflow code base)?

- [ ] Yes. I can contribute this feature independently.
- [ ] Yes. I would be willing to contribute this feature with guidance from the MLflow community.
- [ ] No. **Explanation:**

## Proposal Summary

(In a few sentences, provide a clear, high-level description of the feature request)

## Motivation
- What is the use case for this feature?
- Why is this use case valuable to support for MLflow users in general?
- (If applicable) Why is this use case valuable to support for your projects or organization?
- Why is it currently difficult to achieve this use case? (please be as specific as possible about why related MLflow features and components are insufficient)

## Proposed Changes

### Can this feature be introduced as an MLflow Plugin?
[MLflow Plugins] enable integration of third-party modules with many of MLflow’s components, allowing you to maintain and iterate on certain features independently of the MLflow Repository.

Please refer to the https://mlflow.org/docs/latest/plugins.html for a list of supported MLflow Plugin types. Afterwards, select the option that best applies and fill in the **Explanation** field if applicable:

- [ ] Yes. I think this feature can be introduced as an MLflow Plugin.
- [ ] No. It does not make sense to structure this feature as an MLflow Plugin. **Explanation:**
- [ ] No. I don’t think MLflow Plugins currently supports this type of feature. **Explanation:**

If you selected “Yes,” please submit this feature request and add the “Plugin” GitHub label. An MLflow Community member will provide further guidance on how to get started with developing this feature as an MLflow Plugin.

If you selected “No”, please complete the subsequent sections.

### Which MLflow component(s) does this feature affect?

- [ ] UI
- [ ] CLI
- [ ] API
- [ ] REST-API
- [ ] Examples
- [ ] Docs
- [ ] Tracking
- [ ] Projects
- [ ] Artifacts
- [ ] Models
- [ ] Model Registry
- [ ] Scoring
- [ ] Serving
- [ ] R
- [ ] Java
- [ ] Python

### Implementation impact
Please answer the following questions to the best of your ability. If you are not familiar with the MLflow code base, please feel free to leave questions about implementation details blank as necessary, and select the following “Needs implementation input” box:
- [ ] Needs implementation input

- Does this feature introduce any new libraries to MLflow? If so, which ones?

- Does this feature introduce changes or additions to the [MLflow REST API](https://mlflow.org/docs/latest/rest-api.html)? If so, why are these changes absolutely necessary? (Please describe alternative approaches you’ve considered and why they won’t work)

- Does this feature introduce new MLflow Client or Fluent APIs? If so, why are these changes absolutely necessary? (Please describe alternative approaches you’ve considered and why they won’t work)

- Does this feature make backwards-incompatible changes to existing MLflow APIs and behaviors? (Please describe why any backwards-incompatible changes cannot be avoided)

- Does this feature introduce changes to any of the following backend APIs (select all that apply)?:
   - [ ] Tracking Artifact Repository
   - [ ] Tracking Backend Store
   - [ ] Model Registry Backend Store
   - [ ] Other backend APIs

   (If you selected any of the fields above, please outline the proposed API changes)

### Further Design

(Use this section to include any additional information about your proposed changes. If you have a design document, please provide a link here.)
