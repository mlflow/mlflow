import { useMemo } from 'react';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import type { RunsCompareScatterCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';
import { CompareRunsScatterPlot } from '../charts/CompareRunsScatterPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';

export interface RunsCompareScatterChartCardProps {
  config: RunsCompareScatterCardConfig;
  chartRunData: CompareChartRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareScatterChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
}: RunsCompareScatterChartCardProps) => {
  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );

  const { setTooltip, resetTooltip, selectedRunUuid } = useCompareRunsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={`${config.xaxis.key} vs. ${config.yaxis.key}`}
      // TODO: add i18n after making decision on the final wording
      subtitle={<>Comparing first {slicedRuns.length} runs</>}
    >
      <div css={styles.scatterChartCardWrapper}>
        <CompareRunsScatterPlot
          runsData={slicedRuns}
          xAxis={config.xaxis}
          yAxis={config.yaxis}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          useDefaultHoverBox={false}
          selectedRunUuid={selectedRunUuid}
        />
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  scatterChartCardWrapper: {
    overflow: 'hidden',
  },
};
