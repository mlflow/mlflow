import { type ReactElement } from 'react';
import { render, type RenderResult } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { TestApolloProvider } from '@mlflow/mlflow/src/common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '@mlflow/mlflow/src/common/utils/TestUtils';
import { SqlWarehouseContextProvider } from '../../experiment-page-tabs/SqlWarehouseContext';

export interface RouteDescriptor {
  /** Path template, e.g. `/experiments/:experimentId/datasets`. */
  path: string;
  /** Element to render when the route matches. */
  element: ReactElement;
}

export interface RenderDatasetsPageOptions {
  initialUrl: string;
  routes: RouteDescriptor[];
  /** Memory-history instance from `setupTestRouter()` — owned by the test so it can read it. */
  history: Parameters<typeof TestRouter>[0]['history'];
  /** Experiment ID seeded into `SqlWarehouseContextProvider`. Default `exp-1`. */
  experimentId?: string;
  /** Custom QueryClient if the test wants to seed cache. Default creates a fresh one with retries disabled. */
  queryClient?: QueryClient;
}

export interface RenderDatasetsPageResult extends RenderResult {
  queryClient: QueryClient;
}

const createDefaultQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
    logger: {
      log: () => {},
      warn: () => {},
      // React Query logs the error before the consumer gets a chance to handle it; silence in tests.
      error: () => {},
    },
  });

/**
 * Mounts the V2 datasets pages in the canonical provider stack used by the rest of the
 * MLflow webapp tests (see `ExperimentEvaluationDatasetsPage.test.tsx` for the legacy reference).
 *
 * Tests own their own `setupTestRouter()` call so they can inspect/assert on `history` —
 * pass that history in via `options.history`.
 */
export const renderDatasetsPage = ({
  initialUrl,
  routes,
  history,
  experimentId = 'exp-1',
  queryClient,
}: RenderDatasetsPageOptions): RenderDatasetsPageResult => {
  const client = queryClient ?? createDefaultQueryClient();

  const result = render(
    <TestApolloProvider disableCache>
      <QueryClientProvider client={client}>
        <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <SqlWarehouseContextProvider experimentId={experimentId}>
                <TestRouter
                  routes={routes.map(({ element, path }) => testRoute(element, path))}
                  history={history}
                  initialEntries={[initialUrl]}
                />
              </SqlWarehouseContextProvider>
            </DesignSystemProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      </QueryClientProvider>
    </TestApolloProvider>,
  );

  return { ...result, queryClient: client };
};
