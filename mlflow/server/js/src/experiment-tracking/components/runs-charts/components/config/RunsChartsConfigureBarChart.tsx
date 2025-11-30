import { LegacySelect } from '@databricks/design-system';
import { useCallback, useEffect } from 'react';
import type {
  RunsChartsCardConfig,
  RunsChartsBarCardConfig,
  RunsChartsMetricByDatasetEntry,
} from '../../runs-charts.types';
import { RunsChartsConfigureField, runsChartsRunCountDefaultOptions } from './RunsChartsConfigure.common';
import { isEmpty } from 'lodash';
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
   * Callback for updating metric key
   */
  const updateMetric = useCallback(
    (metricKey: string, datasetName?: string, dataAccessKey?: string) => {
      onStateChange((current) => ({ ...(current as RunsChartsBarCardConfig), metricKey, datasetName, dataAccessKey }));
    },
    [onStateChange],
  );

  /**
   * Callback for updating run count
   */
  const updateVisibleRunCount = useCallback(
    (runsCountToCompare: number) => {
      onStateChange((current) => ({
        ...(current as RunsChartsBarCardConfig),
        runsCountToCompare,
      }));
    },
    [onStateChange],
  );

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
      updateMetric(metricKeyList[0]);
    }
  }, [state.metricKey, updateMetric, metricKeyList, metricKeysByDataset]);

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
            value={emptyMetricsList ? 'No metrics available' : state.metricKey}
            onChange={(metricKey) => updateMetric(metricKey)}
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
