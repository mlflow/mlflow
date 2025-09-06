import { renderHook, waitFor } from '@testing-library/react';
import { useSampledMetricHistory } from './useSampledMetricHistory';
import { MockedReduxStoreProvider } from '../../../../common/utils/TestUtils';
import { fetchEndpoint } from '../../../../common/utils/FetchUtils';
import { getSampledMetricHistoryBulkAction } from '../../../sdk/SampledMetricHistoryService';
import React from 'react';
import { shouldEnableGraphQLSampledMetrics } from '../../../../common/utils/FeatureUtils';
import { IntlProvider } from 'react-intl';
import { TestApolloProvider } from '../../../../common/utils/TestApolloProvider';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldEnableGraphQLSampledMetrics: jest.fn(),
}));

jest.mock('../../../sdk/SampledMetricHistoryService', () => ({
  getSampledMetricHistoryBulkAction: jest.fn(),
}));

const hookWrapper: React.FC<React.PropsWithChildren<unknown>> = ({ children }) => (
  <IntlProvider locale="en">
    <TestApolloProvider>
      <MockedReduxStoreProvider
        state={{
          entities: {
            sampledMetricsByRunUuid: {},
          },
        }}
      >
        {children}
      </MockedReduxStoreProvider>
    </TestApolloProvider>
  </IntlProvider>
);

describe('useSampledMetricHistory (REST)', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableGraphQLSampledMetrics).mockImplementation(() => false);
    jest.mocked(getSampledMetricHistoryBulkAction).mockClear();
    jest.mocked(getSampledMetricHistoryBulkAction).mockImplementation(
      () =>
        ({
          payload: Promise.resolve({}),
          type: 'GET_SAMPLED_METRIC_HISTORY_API_BULK',
        } as any),
    );
  });

  test('should create service calling action when run UUIDs and metric keys are provided', async () => {
    renderHook(
      () =>
        useSampledMetricHistory({
          runUuids: ['run-uuid-1'],
          metricKeys: ['metric-a'],
          enabled: true,
          autoRefreshEnabled: true,
        }),
      {
        wrapper: hookWrapper,
      },
    );

    await waitFor(() => {
      expect(getSampledMetricHistoryBulkAction).toHaveBeenCalledWith(
        ['run-uuid-1'],
        'metric-a',
        undefined,
        undefined,
        undefined,
      );
    });
  });

  test('not call service action when run UUIDs are not provided', async () => {
    renderHook(
      () =>
        useSampledMetricHistory({
          runUuids: [],
          metricKeys: ['metric-a'],
          enabled: true,
          autoRefreshEnabled: true,
        }),
      {
        wrapper: hookWrapper,
      },
    );

    expect(getSampledMetricHistoryBulkAction).not.toHaveBeenCalled();
  });
});
