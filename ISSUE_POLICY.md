# Issue Policy
The MLflow Issue Policy outlines the categories of MLflow GitHub issues and discusses the guidelines & processes
associated with each type of issue.

Before filing an issue, make sure to [search for related issues](https://github.com/mlflow/mlflow/issues) and check if
they address yours.

For support (ex. "How do I do X?"), please ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).

## Issue Categories

Our policy is that GitHub issues fall into one of the following categories:

1. Feature Requests
2. Bug reports
3. Documentation fixes
4. Installation issues

Each category has its own GitHub issue template. Please do not delete the issue template unless you are certain your
issue is outside its scope.

### Feature Requests

#### Guidelines
Feature requests that are likely to be accepted:

* Are minimal in scope (note that it's always easier to add additional functionality later than remove functionality)
* Are extensible (e.g. if adding an integration with an ML framework, is it possible to add similar integrations with other frameworks?)
* Have user impact & value that justifies the maintenance burden of supporting the feature moving forwards. The
  [JQuery contributor guide](https://contribute.jquery.org/open-source/#contributing-something-new) has an excellent discussion on this.

#### Lifecycle

Feature requests typically go through the following lifecycle:

1. A feature request GitHub Issue is submitted, which contains a high-level description of the proposal and its motivation.
   We encourage requesters to provide an overview of the feature's implementation as well, if possible.
2. The feature request is discussed with a committer. The committer will provide input on the implementation overview or
   ask for a more detailed design, if applicable.
3. After discussion & agreement on the feature request and its implementation, an implementation owner is identified.
4. The implementation owner begins developing the feature and ultimately files associated pull requests against the
   MLflow Repository or packages the feature as an MLflow Plugin.

### Bug reports

Bug reports typically go through the following lifecycle:

1. A bug report GitHub Issue is submitted, which contains a high-level description of the bug and information required to reproduce it.
2. An MLflow committer reproduces the bug and provides feedback about how to implement a fix.
3. After an approach has been agreed upon, an owner for the fix is identified. MLflow committers may choose to adopt
   ownership of severe bugs to ensure a timely fix.
4. The fix owner begins implementing the fix and ultimately files associated pull requests.

### Documentation fixes

Documentation issues typically go through the following lifecycle:

1. A documentation GitHub Issue is submitted, which contains a description of the issue and its location(s) in the MLflow documentation.
2. An MLflow committer confirms the documentation issue and provides feedback about how to implement a fix.
3. After an approach has been agreed upon, an owner for the fix is identified. MLflow committers may choose to adopt
   ownership of severe documentation issues to ensure a timely fix.
4. The fix owner begins implementing the fix and ultimately files associated pull requests.

### Installation issues

Installation issues typically go through the following lifecycle:

1. An installation GitHub Issue is submitted, which contains a description of the issue and the platforms its affects.
2. An MLflow committer confirms the installation issue and provides feedback about how to implement a fix.
3. After an approach has been agreed upon, an owner for the fix is identified. MLflow committers may choose to adopt
   ownership of severe installation issues to ensure a timely fix.
4. The fix owner begins implementing the fix and ultimately files associated pull requests.
