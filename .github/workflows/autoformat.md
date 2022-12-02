# Autoformat

## Testing

1. Checkout a new branch and make changes.
1. Push the branch to your fork (https://github.com/{your_username}/mlflow).
1. Switch the default branch of your fork to the branch you just pushed.
1. Create a GitHub token.
1. Create a new Actions secret with the name `MLFLOW_AUTOMATION_TOKEN` and put the token value.
1. Checkout another new branch and run the following commands to make dummy changes.

   ```shell
   # python
   echo "" >> setup.py
   # js
   echo "" >> mlflow/server/js/src/experiment-tracking/components/App.js
   # protos
   echo "message Foo {}" >> mlflow/protos/service.proto
   ```

1. Create a PR from the branch containing the dummy changes in your fork.
1. Comment `@mlflow-automation autoformat` on the PR and ensure the workflow runs successfully.
   The workflow status can be checked at https://github.com/{your_username}/mlflow/actions/workflows/autoformat.yml.
1. Delete the GitHub token and reset the default branch.
