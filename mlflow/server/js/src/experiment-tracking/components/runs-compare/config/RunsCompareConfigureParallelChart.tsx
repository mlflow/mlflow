import { Select } from '@databricks/design-system';
import { useCallback } from 'react';
import type { RunsCompareCardConfig, RunsCompareParallelCardConfig } from '../runs-compare.types';
import { RunsCompareConfigureField } from './RunsCompareConfigure.common';

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsCompareConfigureParallelChart = ({
  state,
  onStateChange,
  metricKeyList,
  paramKeyList,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  state: Partial<RunsCompareParallelCardConfig>;
  onStateChange: (
    setter: (current: RunsCompareCardConfig) => RunsCompareParallelCardConfig,
  ) => void;
}) => {
  /**
   * Callback for updating selected metrics and params
   */

  const updateSelectedParams = useCallback(
    (selectedParams: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsCompareParallelCardConfig),
        selectedParams,
      }));
    },
    [onStateChange],
  );

  const updateSelectedMetrics = useCallback(
    (selectedMetrics: string[]) => {
      onStateChange((current) => ({
        ...(current as RunsCompareParallelCardConfig),
        selectedMetrics,
      }));
    },
    [onStateChange],
  );

  const emptyMetricsList = metricKeyList.length === 0;
  const emptyParamsList = paramKeyList.length === 0;

  return (
    <>
      <RunsCompareConfigureField title='Params'>
        <Select
          mode={emptyParamsList ? undefined : 'multiple'}
          onChange={updateSelectedParams}
          style={{
            width: 300,
          }}
          value={emptyParamsList ? ('No parameters available' as any) : state.selectedParams}
          disabled={emptyParamsList}
        >
          {paramKeyList.map((param) => (
            <Select.Option value={param}>{param}</Select.Option>
          ))}
        </Select>
      </RunsCompareConfigureField>
      <RunsCompareConfigureField title='Metrics'>
        <Select
          mode={emptyMetricsList ? undefined : 'multiple'}
          onChange={updateSelectedMetrics}
          style={{
            width: 300,
          }}
          value={emptyMetricsList ? ('No metrics available' as any) : state.selectedMetrics}
          disabled={emptyMetricsList}
        >
          {metricKeyList.map((metric) => (
            <Select.Option value={metric}>{metric}</Select.Option>
          ))}
        </Select>
      </RunsCompareConfigureField>
    </>
  );
};
