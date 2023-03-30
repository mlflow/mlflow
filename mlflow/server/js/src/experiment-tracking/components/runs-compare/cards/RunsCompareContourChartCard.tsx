import { useMemo } from 'react';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import type { RunsCompareContourCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';
import { CompareRunsContourPlot } from '../charts/CompareRunsContourPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';

export interface RunsCompareContourChartCardProps {
  config: RunsCompareContourCardConfig;
  chartRunData: CompareChartRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareContourChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
}: RunsCompareContourChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );

  const { setTooltip, resetTooltip, selectedRunUuid } = useCompareRunsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={`${config.xaxis.key} vs. ${config.yaxis.key} vs. ${config.zaxis.key}`}
      // TODO: add i18n after making decision on the final wording
      subtitle={<>Comparing first {slicedRuns.length} runs</>}
    >
      <div css={styles.contourChartCardWrapper}>
        <CompareRunsContourPlot
          runsData={slicedRuns}
          xAxis={config.xaxis}
          yAxis={config.yaxis}
          zAxis={config.zaxis}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          selectedRunUuid={selectedRunUuid}
        />
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  contourChartCardWrapper: {
    overflow: 'hidden',
  },
};
