# UI Preview

Deploy a live preview of the MLflow UI as a [Databricks App](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) when a PR modifies the frontend (`mlflow/server/js/`).

## How it works

1. Add the `ui-preview` label to a PR with UI changes
2. The [UI Preview workflow](../workflows/ui-preview.yml) builds the frontend and deploys it to a Databricks App
3. A comment with the preview URL is posted on the PR
4. The app is automatically deleted when the PR is closed

## Access

Preview apps are only accessible to core maintainers with workspace access.
