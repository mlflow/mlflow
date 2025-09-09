import { renderHook, waitFor } from '@testing-library/react';
import { graphql } from 'msw';
import { setupServer } from '../../../../common/utils/setup-msw';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';
import type { GetMetricHistoryBulkInterval } from '../../../../graphql/__generated__/graphql';
import { GetRun, MlflowRunStatus } from '../../../../graphql/__generated__/graphql';
import { useSampledMetricHistoryGraphQL } from './useSampledMetricHistoryGraphQL';
import { IntlProvider } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import {
  ApolloClient,
  ApolloProvider,
  createHttpLink,
  InMemoryCache,
} from '@mlflow/mlflow/src/common/utils/graphQLHooks';

const createMetrics = (count: number) =>
  Array.from({ length: count }, (_, i) => ({
    __typename: 'MlflowMetricWithRunId' as const,
    timestamp: (i * 1000).toString(),
    step: i.toString(),
    runId: `test-run-uuid-${i % 10}`,
    key: 'test-metric-key',
    value: i,
  }));

describe('useSampledMetricHistoryGraphQL', () => {
  const server = setupServer();

  beforeEach(() => {
    server.resetHandlers();
  });

  const getApolloClient = () => {
    return new ApolloClient({
      cache: new InMemoryCache(),
      link: createHttpLink(),
    });
  };

  const renderTestHook = (runUuids: string[]) =>
    renderHook(
      () =>
        useSampledMetricHistoryGraphQL({
          metricKey: 'test-metric-key',
          runUuids,
          enabled: true,
          autoRefreshEnabled: true,
        }),
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <ApolloProvider client={getApolloClient()}>{children}</ApolloProvider>
          </IntlProvider>
        ),
      },
    );

  it('returns a correct data payload corresponding to mocked response', async () => {
    server.use(
      graphql.query<GetMetricHistoryBulkInterval>('GetMetricHistoryBulkInterval', (req, res, ctx) =>
        res(
          ctx.data({
            mlflowGetMetricHistoryBulkInterval: {
              __typename: 'MlflowGetMetricHistoryBulkIntervalResponse',
              apiError: null,
              metrics: createMetrics(100),
            },
          }),
        ),
      ),
    );

    const { result } = renderTestHook(['test-run-uuid-2', 'test-run-uuid-5']);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Known run uuids should have 10 metric entries each
    expect(result.current.resultsByRunUuid['test-run-uuid-2']?.['test-metric-key'].metricsHistory).toHaveLength(10);
    expect(result.current.resultsByRunUuid['test-run-uuid-5']?.['test-metric-key'].metricsHistory).toHaveLength(10);

    // Unknown run uuid should not have any metric entries
    expect(result.current.resultsByRunUuid['test-run-uuid-118']?.['test-metric-key'].metricsHistory).toBeUndefined();
  });

  it('displays "resource not found" error if relevant message comes from the backend', async () => {
    jest.spyOn(Utils, 'displayGlobalErrorNotification');

    server.use(
      graphql.query<GetMetricHistoryBulkInterval>('GetMetricHistoryBulkInterval', (req, res, ctx) =>
        res(
          ctx.data({
            mlflowGetMetricHistoryBulkInterval: {
              __typename: 'MlflowGetMetricHistoryBulkIntervalResponse',
              apiError: {
                __typename: 'ApiError',
                code: 'RESOURCE_DOES_NOT_EXIST',
                message: 'Requested resource does not exist',
              },
              metrics: null,
            },
          }),
        ),
      ),
    );

    const { result } = renderTestHook(['test-run-uuid-1']);

    await waitFor(() => {
      expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledWith('Requested resource does not exist');
    });

    expect(result.current.apiError).toEqual(expect.objectContaining({ code: 'RESOURCE_DOES_NOT_EXIST' }));
  });

  it('returns apollo-level error if occurred', async () => {
    jest.spyOn(Utils, 'displayGlobalErrorNotification');

    server.use(
      graphql.query<GetMetricHistoryBulkInterval>('GetMetricHistoryBulkInterval', (req, res, ctx) =>
        res(
          ctx.status(400),
          ctx.data({
            mlflowGetMetricHistoryBulkInterval: {
              __typename: 'MlflowGetMetricHistoryBulkIntervalResponse',
              apiError: null,
              metrics: null,
            },
          }),
        ),
      ),
    );

    const { result } = renderTestHook(['test-run-uuid-1']);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
  });
});
