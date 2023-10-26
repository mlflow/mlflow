import { useMemo } from 'react';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import type { RunsCompareContourCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';
import { RunsContourPlot } from '../../runs-charts/components/RunsContourPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';

export interface RunsCompareContourChartCardProps {
  config: RunsCompareContourCardConfig;
  chartRunData: RunsChartsRunData[];

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

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={`${config.xaxis.key} vs. ${config.yaxis.key} vs. ${config.zaxis.key}`}
      // TODO: add i18n after making decision on the final wording
      subtitle={<>Comparing first {slicedRuns.length} runs</>}
    >
      <div css={styles.contourChartCardWrapper}>
        <RunsContourPlot
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
