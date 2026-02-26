import { LegacySelect, type LegacySelectProps } from '@databricks/design-system';
import { useMemo } from 'react';
import type { RunsChartsHistogramCardConfig } from '../../runs-charts.types';
import { RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';
import { FormattedMessage } from 'react-intl';

export const RunsChartsConfigureHistogramChart = ({
  state,
  onStateChange,
  histogramKeys,
}: {
  state: Partial<RunsChartsHistogramCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsHistogramCardConfig) => void;
  histogramKeys: string[];
}) => {
  const updateHistogramKey = (histogramKey: string) => {
    onStateChange((current) => ({
      ...current,
      histogramKeys: [histogramKey],
      selectedRunUuids: (current as RunsChartsHistogramCardConfig).selectedRunUuids || [],
    }));
  };

  const selectedHistogramKey = state.histogramKeys?.[0] || '';

  const selectOptions: LegacySelectProps['options'] = useMemo(
    () =>
      histogramKeys.map((key) => ({
        value: key,
        label: key,
      })),
    [histogramKeys],
  );

  return (
    <>
      <RunsChartsConfigureField title="Histogram">
        <LegacySelect
          className="histogram-key-select"
          css={styles.selectFull}
          value={selectedHistogramKey}
          onChange={updateHistogramKey}
          options={selectOptions}
          placeholder="Select histogram..."
          dangerouslySetAntdProps={{
            showSearch: true,
          }}
        />
      </RunsChartsConfigureField>
    </>
  );
};

const styles = {
  selectFull: {
    width: '100%',
  },
};
