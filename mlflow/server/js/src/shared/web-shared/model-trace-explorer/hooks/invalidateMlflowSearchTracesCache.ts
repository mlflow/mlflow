import type { QueryClient } from '@databricks/web-shared/query-client';

export const SEARCH_MLFLOW_TRACES_QUERY_KEY = 'searchMlflowTraces';

export const invalidateMlflowSearchTracesCache = ({ queryClient }: { queryClient: QueryClient }) => {
  queryClient.invalidateQueries({ queryKey: [SEARCH_MLFLOW_TRACES_QUERY_KEY] });
};
