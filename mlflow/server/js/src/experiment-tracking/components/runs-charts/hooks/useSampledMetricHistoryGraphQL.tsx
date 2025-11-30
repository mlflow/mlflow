import { gql, NetworkStatus } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { useQuery } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL } from '../../../utils/MetricsUtils';
import { groupBy, keyBy } from 'lodash';
import { useEffect, useMemo } from 'react';
import type { SampledMetricsByRun } from './useSampledMetricHistory';
import type { GetMetricHistoryBulkInterval } from '../../../../graphql/__generated__/graphql';
import Utils from '../../../../common/utils/Utils';
import { useIntl } from 'react-intl';

const GET_METRIC_HISTORY_BULK_INTERVAL = gql`
  query GetMetricHistoryBulkInterval($data: MlflowGetMetricHistoryBulkIntervalInput!)
  @component(name: "MLflow.ExperimentRunTracking") {
    mlflowGetMetricHistoryBulkInterval(input: $data) {
      __typename
      metrics {
        timestamp
        step
        runId
        key
        value
      }
      apiError {
        code
        message
      }
    }
  }
`;

export const useSampledMetricHistoryGraphQL = ({
  metricKey,
  runUuids,
  autoRefreshEnabled,
  enabled,
  maxResults = 320,
  range,
}: {
  runUuids: string[];
  metricKey: string;
  maxResults?: number;
  range?: [number, number];
  enabled?: boolean;
  autoRefreshEnabled?: boolean;
}) => {
  const intl = useIntl();
  const { data, refetch, startPolling, stopPolling, networkStatus, error } = useQuery<GetMetricHistoryBulkInterval>(
    GET_METRIC_HISTORY_BULK_INTERVAL,
    {
      skip: !enabled,
      notifyOnNetworkStatusChange: true,
      pollInterval: autoRefreshEnabled ? EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL : undefined,
      onCompleted(data) {
        if (data.mlflowGetMetricHistoryBulkInterval?.apiError?.code === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.displayGlobalErrorNotification(
            intl.formatMessage({
              defaultMessage: 'Requested resource does not exist',
              description:
                'Error message displayed when a requested run does not exist while fetching sampled metric data',
            }),
          );
        } else if (data.mlflowGetMetricHistoryBulkInterval?.apiError?.message) {
          Utils.logErrorAndNotifyUser(new Error(data.mlflowGetMetricHistoryBulkInterval.apiError.message));
        }
      },
      variables: {
        data: {
          runIds: runUuids,
          metricKey,
          startStep: range?.[0] ?? null,
          endStep: range?.[1] ?? null,
          maxResults,
        },
      },
    },
  );

  useEffect(() => {
    if (autoRefreshEnabled) {
      startPolling(EXPERIMENT_RUNS_SAMPLE_METRIC_AUTO_REFRESH_INTERVAL);
    } else {
      stopPolling();
    }
  }, [autoRefreshEnabled, startPolling, stopPolling]);

  const resultsByRunUuid = useMemo<Record<string, SampledMetricsByRun>>(() => {
    if (data) {
      const metrics = data?.mlflowGetMetricHistoryBulkInterval?.metrics;
      const metricsByRunId = groupBy(metrics, 'runId');

      // Transform the data into the already existing format
      return keyBy(
        runUuids.map(
          (runId) =>
            ({
              runUuid: runId,
              [metricKey]: {
                metricsHistory: metricsByRunId[runId]?.map(({ key, step, timestamp, value }) => ({
                  key: key ?? undefined,
                  step: Number(step),
                  timestamp: Number(timestamp),
                  value: value ?? undefined,
                })),
              },
            } as SampledMetricsByRun),
        ),
        'runUuid',
      );
    }

    return {};
  }, [data, metricKey, runUuids]);

  const isLoading = networkStatus === NetworkStatus.loading || networkStatus === NetworkStatus.setVariables;
  const isRefreshing = networkStatus === NetworkStatus.poll;
  return {
    resultsByRunUuid,
    isLoading,
    isRefreshing,
    refresh: refetch,
    error,
    apiError: data?.mlflowGetMetricHistoryBulkInterval?.apiError,
  };
};
