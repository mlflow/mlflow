import { QueryClient } from './queryClient';

/**
 * Builds the QueryClient used by the OSS MLflow UI entry points.
 *
 * `networkMode: 'always'` is deliberate: React Query's default pauses queries before `queryFn`
 * runs whenever `navigator.onLine` is false, and that flag is not trustworthy. MLflow UI queries
 * target the tracking server that served the page, so if the page loaded, the server is reachable.
 */
export const createMlflowQueryClient = () =>
  // eslint-disable-next-line @databricks/no-singleton-query-client -- OSS entry points own their client
  new QueryClient({
    defaultOptions: {
      queries: { networkMode: 'always' },
    },
  });
