import { LegacySelect } from '@databricks/design-system';
import { useCallback, useEffect } from 'react';
import type { RunsChartsCardConfig, RunsChartsBarCardConfig } from '../../runs-charts.types';
import {
  RunsChartsConfigureField,
  runsChartsRunCountDefaultOptions,
  RunsChartsRunNumberSelect,
} from './RunsChartsConfigure.common';

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsChartsConfigureBarChart = ({
  state,
  onStateChange,
  metricKeyList,
}: {
  metricKeyList: string[];
  state: Partial<RunsChartsBarCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsBarCardConfig) => void;
}) => {
  /**
   * Callback for updating metric key
   */
  const updateMetric = useCallback(
    (metricKey: string) => {
      onStateChange((current) => ({ ...(current as RunsChartsBarCardConfig), metricKey }));
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
    if (!state.metricKey && metricKeyList?.[0]) {
      updateMetric(metricKeyList[0]);
    }
  }, [state.metricKey, updateMetric, metricKeyList]);

  const emptyMetricsList = metricKeyList.length === 0;

  return (
    <>
      <RunsChartsConfigureField title="Metric">
        <LegacySelect
          css={styles.selectFull}
          value={emptyMetricsList ? 'No metrics available' : state.metricKey}
          onChange={updateMetric}
          disabled={emptyMetricsList}
          dangerouslySetAntdProps={{ showSearch: true }}
        >
          {metricKeyList.map((metric) => (
            <LegacySelect.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
              {metric}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
      </RunsChartsConfigureField>
      <RunsChartsRunNumberSelect
        value={state.runsCountToCompare}
        onChange={updateVisibleRunCount}
        options={runsChartsRunCountDefaultOptions}
      />
    </>
  );
};

const styles = { selectFull: { width: '100%' } };
