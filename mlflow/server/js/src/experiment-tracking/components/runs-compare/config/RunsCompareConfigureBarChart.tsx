import { Select } from '@databricks/design-system';
import { useCallback, useEffect } from 'react';
import type { RunsCompareCardConfig, RunsCompareBarCardConfig } from '../runs-compare.types';
import {
  RunsCompareConfigureField,
  runsCompareRunCountDefaultOptions,
  RunsCompareRunNumberSelect,
} from './RunsCompareConfigure.common';

/**
 * Form containing configuration controls for runs compare charts.
 */
export const RunsCompareConfigureBarChart = ({
  state,
  onStateChange,
  metricKeyList,
}: {
  metricKeyList: string[];
  state: Partial<RunsCompareBarCardConfig>;
  onStateChange: (setter: (current: RunsCompareCardConfig) => RunsCompareBarCardConfig) => void;
}) => {
  /**
   * Callback for updating metric key
   */
  const updateMetric = useCallback(
    (metricKey: string) => {
      onStateChange((current) => ({ ...(current as RunsCompareBarCardConfig), metricKey }));
    },
    [onStateChange],
  );

  /**
   * Callback for updating run count
   */
  const updateVisibleRunCount = useCallback(
    (runsCountToCompare: number) => {
      onStateChange((current) => ({
        ...(current as RunsCompareBarCardConfig),
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
      <RunsCompareConfigureField title='Metric'>
        <Select
          css={styles.selectFull}
          value={emptyMetricsList ? 'No metrics available' : state.metricKey}
          onChange={updateMetric}
          disabled={emptyMetricsList}
          dangerouslySetAntdProps={{ showSearch: true }}
        >
          {metricKeyList.map((metric) => (
            <Select.Option key={metric} value={metric} data-testid={`metric-${metric}`}>
              {metric}
            </Select.Option>
          ))}
        </Select>
      </RunsCompareConfigureField>
      <RunsCompareRunNumberSelect
        value={state.runsCountToCompare}
        onChange={updateVisibleRunCount}
        options={runsCompareRunCountDefaultOptions}
      />
    </>
  );
};

const styles = { selectFull: { width: '100%' } };
