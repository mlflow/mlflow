# UI Preview

Deploy a live preview of the MLflow UI as a [Databricks App](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/) when a PR modifies the frontend (`mlflow/server/js/`).

## How it works

1. Add the `ui-preview` label to a PR with UI changes
2. The [UI Preview workflow](../workflows/ui-preview.yml) builds the frontend and deploys it to a Databricks App
3. A comment with the preview URL is posted on the PR
4. The app is automatically deleted when the PR is closed

## Access

Preview apps are only accessible to core maintainers with workspace access.

## API access

To query or add data to a preview app, set the following environment variables:

```bash
export DATABRICKS_HOST="https://..."
export DATABRICKS_CLIENT_ID="..."
export DATABRICKS_CLIENT_SECRET="..."
export APP_URL="..."
```

Then, obtain an access token:

```bash
export TOKEN=$(curl -s -X POST "$DATABRICKS_HOST/oidc/v1/token" \
  -d "grant_type=client_credentials&client_id=$DATABRICKS_CLIENT_ID&client_secret=$DATABRICKS_CLIENT_SECRET&scope=all-apis" \
  | jq -r '.access_token')
```

Once the token is obtained, run the following command to verify it works:

```bash
curl -s "$APP_URL/api/2.0/mlflow/experiments/search" \
  -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"max_results": 10}' | jq .
```

You can also use the MLflow Python client:

```bash
export MLFLOW_TRACKING_URI="$APP_URL"
export MLFLOW_TRACKING_TOKEN="$TOKEN"
```

```python
import mlflow

mlflow.search_experiments(max_results=10)
```

See [Connect to Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local) for more details on authentication.
