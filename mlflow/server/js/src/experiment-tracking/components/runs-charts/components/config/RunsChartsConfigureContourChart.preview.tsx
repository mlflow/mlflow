import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsContourPlot } from '../RunsContourPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsContourCardConfig } from '../../runs-charts.types';

export const RunsChartsConfigureContourChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsContourCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  // We need to re-render the chart when any axis config changes.
  // Plotly tries to determine axis format based on values and is not capable
  // of dynamic switching between different axis types, so we need to make sure
  // that we re-mount the chart when config changes.
  const key = useMemo(() => {
    const { xaxis, yaxis, zaxis } = cardConfig;
    return JSON.stringify({ xaxis, yaxis, zaxis });
  }, [cardConfig]);

  return (
    <RunsContourPlot
      xAxis={cardConfig.xaxis}
      yAxis={cardConfig.yaxis}
      zAxis={cardConfig.zaxis}
      runsData={previewData}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      key={key}
    />
  );
};
