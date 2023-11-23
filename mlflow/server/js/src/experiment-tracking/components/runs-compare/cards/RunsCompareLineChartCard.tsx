import { LegacySkeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsLinePlot } from '../../runs-charts/components/RunsMetricsLinePlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import type { RunsCompareLineCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';

export interface RunsCompareLineChartCardProps {
  config: RunsCompareLineCardConfig;
  chartRunData: RunsChartsRunData[];

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

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

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
          <RunsMetricsLinePlot
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
