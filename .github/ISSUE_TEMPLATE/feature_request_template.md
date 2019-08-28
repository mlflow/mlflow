---
name: Feature Request
about: Use this template for feature and enhancement proposals.
labels: 'enhancement'
title: "[FR]"
---
Thank you for submitting an issue. Please refer to our [issue policy](https://www.github.com/mlflow/mlflow/blob/master/ISSUE_POLICY.md)
for information on what types of issues we address.
  
Please fill in this template and do not delete it unless you are sure your issue is outside its scope.

-------
## Guidelines

Feature requests typically go through the following lifecycle:

1. Submit feature request with high-level description on GitHub issues (this is what you're doing now)
2. Discuss feature request with a committer, who may ask for a more detailed design
3. After discussion & agreement on feature request, start implementation


## Describe the proposal
Provide a clear high-level description of the feature request in the following sections. Feature requests that are likely to be accepted:
* Are minimal in scope (note that it's always easier to add additional functionality later than remove functionality)
* Are extensible (e.g. if adding an integration with an ML framework, is it possible to add similar integrations with other frameworks?)
* Have user impact & value that justifies the maintenance burden of supporting the feature moving forwards. The [JQuery contributor guide](https://contribute.jquery.org/open-source/#contributing-something-new) has an excellent discussion on this.

### Motivation
What is the use case in mind?  Why is it valuable to support, and why is it currently difficult or impossible to achieve? Could the desired functionality alternatively be implemented as a third-party package using MLflow public APIs? 

### Proposed Changes
For user-facing changes, what APIs are you proposing to add or modify? For internal changes, what code paths will need to be modified? 
