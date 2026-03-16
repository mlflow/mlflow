import { LegacySelect } from '@databricks/design-system';
import { useCallback, useEffect } from 'react';
import { isEmpty, isUndefined } from 'lodash';
import type {
  RunsChartsCardConfig,
  RunsChartsBarCardConfig,
  RunsChartsMetricByDatasetEntry,
} from '../../runs-charts.types';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';
import { RunsChartsConfigureMetricWithDatasetSelect } from './RunsChartsConfigureMetricWithDatasetSelect';

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsChartsConfigureBarChart = ({
  state,
  onStateChange,
  metricKeyList,
  metricKeysByDataset,
}: {
  metricKeyList: string[];
  metricKeysByDataset?: RunsChartsMetricByDatasetEntry[];
  state: Partial<RunsChartsBarCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsBarCardConfig) => void;
}) => {
  /**
   * Callback for updating metric key (used for dataset-aware metrics which remain single-select)
   */
  const updateMetric = useCallback(
    (metricKey: string, datasetName?: string, dataAccessKey?: string) => {
      onStateChange((current) => ({
        ...(current as RunsChartsBarCardConfig),
        metricKey,
        selectedMetricKeys: [dataAccessKey ?? metricKey],
        datasetName,
        dataAccessKey,
      }));
    },
    [onStateChange],
  );

  /**
   * Callback for updating selected metrics (multi-select)
   */
  const updateSelectedMetrics = useCallback(
    (metricKeys: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsChartsBarCardConfig),
        metricKey: metricKeys[0] ?? '',
        selectedMetricKeys: metricKeys,
      }));
    },
    [onStateChange],
  );

  /**
   * For backwards compatibility, if selectedMetricKeys is not present,
   * set it using metricKey
   */
  useEffect(() => {
    if (isUndefined(state.selectedMetricKeys) && !isUndefined(state.metricKey) && state.metricKey !== '') {
      updateSelectedMetrics([state.dataAccessKey ?? state.metricKey]);
    }
  }, [state.selectedMetricKeys, state.metricKey, state.dataAccessKey, updateSelectedMetrics]);

  /**
   * If somehow metric key is not predetermined, automatically
   * select the first one so it's not empty
   */
  useEffect(() => {
    if (!state.metricKey && metricKeysByDataset?.[0]) {
      updateMetric(
        metricKeysByDataset[0].metricKey,
        metricKeysByDataset[0].datasetName,
        metricKeysByDataset[0].dataAccessKey,
      );
      return;
    }

    if (!state.metricKey && metricKeyList?.[0]) {
      updateSelectedMetrics([metricKeyList[0]]);
    }
  }, [state.metricKey, updateMetric, updateSelectedMetrics, metricKeyList, metricKeysByDataset]);

  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <>
      <RunsChartsConfigureField title="Metric">
        {!isEmpty(metricKeysByDataset) ? (
          <RunsChartsConfigureMetricWithDatasetSelect
            metricKeysByDataset={metricKeysByDataset}
            onChange={({ metricKey, datasetName, dataAccessKey }) =>
              updateMetric(metricKey, datasetName, dataAccessKey)
            }
            value={state.dataAccessKey ?? state.metricKey}
          />
        ) : (
          <LegacySelect
            css={styles.selectFull}
            mode="multiple"
            placeholder={emptyMetricsList ? 'No metrics available' : 'Select metrics'}
            value={emptyMetricsList ? [] : (state.selectedMetricKeys ?? (state.metricKey ? [state.metricKey] : []))}
            onChange={updateSelectedMetrics}
            disabled={emptyMetricsList}
            dangerouslySetAntdProps={{ showSearch: true }}
          >
            {metricKeyList.map((metric) => (
              <LegacySelect.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
                {metric}
              </LegacySelect.Option>
            ))}
          </LegacySelect>
        )}
      </RunsChartsConfigureField>
    </>
  );
};

const styles = { selectFull: { width: '100%' } };
