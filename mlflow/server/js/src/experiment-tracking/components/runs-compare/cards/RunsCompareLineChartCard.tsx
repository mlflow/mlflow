import { LegacySkeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsMetricsLinePlot } from '../charts/CompareRunsMetricsLinePlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import type { RunsCompareLineCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';

export interface RunsCompareLineChartCardProps {
  config: RunsCompareLineCardConfig;
  chartRunData: CompareChartRunData[];

  isMetricHistoryLoading?: boolean;

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareLineChartCard = ({
  config,
  chartRunData,
  isMetricHistoryLoading,
  onDelete,
  onEdit,
}: RunsCompareLineChartCardProps) => {
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
      <div css={styles.lineChartCardWrapper}>
        {isMetricHistoryLoading ? (
          <LegacySkeleton />
        ) : (
          <CompareRunsMetricsLinePlot
            runsData={slicedRuns}
            metricKey={config.metricKey}
            scaleType={config.scaleType}
            xAxisKey={config.xAxisKey}
            lineSmoothness={config.lineSmoothness}
            useDefaultHoverBox={false}
            onHover={setTooltip}
            onUnhover={resetTooltip}
            selectedRunUuid={selectedRunUuid}
          />
        )}
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  lineChartCardWrapper: {
    overflow: 'hidden',
  },
};
