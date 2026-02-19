import { LegacySelect } from '@databricks/design-system';
import { useCallback } from 'react';
import type { RunsChartsCardConfig, RunsChartsParallelCardConfig } from '../../runs-charts.types';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsChartsConfigureParallelChart = ({
  state,
  onStateChange,
  metricKeyList,
  paramKeyList,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  state: Partial<RunsChartsParallelCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsParallelCardConfig) => void;
}) => {
  /**
   * Callback for updating selected metrics and params
   */

  const updateSelectedParams = useCallback(
    (selectedParams: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsChartsParallelCardConfig),
        selectedParams,
      }));
    },
    [onStateChange],
  );

  const updateSelectedMetrics = useCallback(
    (selectedMetrics: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsChartsParallelCardConfig),
        selectedMetrics,
      }));
    },
    [onStateChange],
  );

  const emptyMetricsList = metricKeyList.length === 0;
  const emptyParamsList = paramKeyList.length === 0;

  return (
    <>
      <RunsChartsConfigureField title="Params">
        <LegacySelect
          mode={emptyParamsList ? undefined : 'multiple'}
          onChange={updateSelectedParams}
          style={{
            width: 275,
          }}
          value={emptyParamsList ? ('No parameters available' as any) : state.selectedParams}
          disabled={emptyParamsList}
        >
          {paramKeyList.map((param) => (
            <LegacySelect.Option value={param} key={param}>
              {param}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
      </RunsChartsConfigureField>
      <RunsChartsConfigureField title="Metrics">
        <LegacySelect
          mode={emptyMetricsList ? undefined : 'multiple'}
          onChange={updateSelectedMetrics}
          style={{
            width: 275,
          }}
          value={emptyMetricsList ? ('No metrics available' as any) : state.selectedMetrics}
          disabled={emptyMetricsList}
        >
          {metricKeyList.map((metric) => (
            <LegacySelect.Option value={metric} key={metric}>
              {metric}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
      </RunsChartsConfigureField>
    </>
  );
};
