import { type ReactElement } from 'react';
import { render, type RenderResult } from '@testing-library/react';
import { DesignSystemProvider, Notification } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { TestApolloProvider } from '@mlflow/mlflow/src/common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '@mlflow/mlflow/src/common/utils/TestUtils';
import { SqlWarehouseContextProvider } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-page-tabs/SqlWarehouseContext';
import { MonitoringConfigProvider } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';
import { setLocalStorageItem } from '@databricks/web-shared/hooks';

const WAREHOUSE_STORAGE_PREFIX = 'mlflow_warehouse_experiment_';
const WAREHOUSE_STORAGE_VERSION = 1;

/**
 * Seed the per-experiment SQL-warehouse localStorage key so `SqlWarehouseContextProvider`
 * initializes with a warehouse. V4 trace search is disabled without one, so tests that want the
 * table to load must seed a warehouse *before* rendering; tests exercising the no-warehouse state
 * skip this. Uses the real `setLocalStorageItem` (same version + scoped flag as
 * `usePersistedSqlWarehouseId`) so the stored shape matches exactly what the provider reads.
 */
export const seedWarehouse = (experimentId: string, warehouseId = 'wh-test') => {
  setLocalStorageItem(`${WAREHOUSE_STORAGE_PREFIX}${experimentId}`, WAREHOUSE_STORAGE_VERSION, true, {
    SQL_WAREHOUSE: { id: warehouseId, timestamp: 1 },
  });
};

export interface RouteDescriptor {
  path: string;
  element: ReactElement;
}

export interface RenderTracesV4PageOptions {
  initialUrl: string;
  routes: RouteDescriptor[];
  /** Memory history from `setupTestRouter()`, owned by the test so it can read location. */
  history: Parameters<typeof TestRouter>[0]['history'];
  /** Seeded into `SqlWarehouseContextProvider`. Defaults to `'exp-1'`. */
  experimentId?: string;
  queryClient?: QueryClient;
}

export interface RenderTracesV4PageResult extends RenderResult {
  queryClient: QueryClient;
}

const noop = () => {};

const createDefaultQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
    logger: { log: noop, warn: noop, error: noop },
  });

/**
 * Render helper for V4 traces tests. Mirrors datasets-v2's `renderDatasetsPage` provider stack and
 * additionally wraps `MonitoringConfigProvider` (time range + refresh) and a `Notification`
 * provider/viewport (delete toasts). Router, MSW, and flags are owned by the individual test.
 */
export const renderTracesV4Page = ({
  initialUrl,
  routes,
  history,
  experimentId = 'exp-1',
  queryClient,
}: RenderTracesV4PageOptions): RenderTracesV4PageResult => {
  const client = queryClient ?? createDefaultQueryClient();

  const result = render(
    <TestApolloProvider disableCache>
      <QueryClientProvider client={client}>
        <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <Notification.Provider>
                <SqlWarehouseContextProvider experimentId={experimentId}>
                  <MonitoringConfigProvider>
                    <TestRouter
                      routes={routes.map(({ element, path }) => testRoute(element, path))}
                      history={history}
                      initialEntries={[initialUrl]}
                    />
                  </MonitoringConfigProvider>
                </SqlWarehouseContextProvider>
                <Notification.Viewport />
              </Notification.Provider>
            </DesignSystemProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      </QueryClientProvider>
    </TestApolloProvider>,
  );

  return { ...result, queryClient: client };
};
