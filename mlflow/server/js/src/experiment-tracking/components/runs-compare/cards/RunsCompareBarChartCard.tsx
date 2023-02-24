import { useMemo } from 'react';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsMetricsBarPlot } from '../charts/CompareRunsMetricsBarPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import type { RunsCompareBarCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';

export interface RunsCompareBarChartCardProps {
  config: RunsCompareBarCardConfig;
  chartRunData: CompareChartRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const barChartCardDefaultMargin = {
  t: 24,
  b: 48,
  r: 0,
  l: 4,
  pad: 0,
};

export const RunsCompareBarChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
}: RunsCompareBarChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );
  const { setTooltip, resetTooltip, selectedRunUuid } = useCompareRunsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={config.metricKey}
      // TODO: add i18n after making decision on the final wording
      subtitle={<>Comparing first {slicedRuns.length} runs</>}
    >
      <div css={styles.barChartCardWrapper}>
        <CompareRunsMetricsBarPlot
          runsData={slicedRuns}
          metricKey={config.metricKey}
          displayRunNames={false}
          displayMetricKey={false}
          useDefaultHoverBox={false}
          margin={barChartCardDefaultMargin}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          selectedRunUuid={selectedRunUuid}
        />
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  barChartCardWrapper: {
    overflow: 'hidden',
  },
};
