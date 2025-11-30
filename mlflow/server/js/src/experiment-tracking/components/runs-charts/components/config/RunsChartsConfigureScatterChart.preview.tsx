import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsScatterPlot } from '../RunsScatterPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsScatterCardConfig } from '../../runs-charts.types';

export const RunsChartsConfigureScatterChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsScatterCardConfig;
}) => {
  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  // We need to re-render the chart when the x or y axis changes.
  // Plotly tries to determine axis format based on values and is not capable
  // of dynamic switching between different axis types, so we need to make sure
  // that we re-mount the chart when config changes.
  const key = useMemo(() => {
    const { xaxis, yaxis } = cardConfig;
    return JSON.stringify({ xaxis, yaxis });
  }, [cardConfig]);

  return (
    <RunsScatterPlot
      xAxis={cardConfig.xaxis}
      yAxis={cardConfig.yaxis}
      runsData={previewData}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      key={key}
    />
  );
};
