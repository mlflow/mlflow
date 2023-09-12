import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import LazyParallelCoordinatesPlot, {
  processParallelCoordinateData,
} from '../charts/LazyParallelCoordinatesPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import { RunsCompareParallelCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureParallelChartPreview = ({
  previewData,
  cardConfig,
}: {
  previewData: CompareChartRunData[];
  cardConfig: RunsCompareParallelCardConfig;
}) => {
  const selectedParamsCount = cardConfig.selectedParams?.length || 0;
  const selectedMetricsCount = cardConfig.selectedMetrics?.length || 0;

  const isConfigured = selectedParamsCount + selectedMetricsCount >= 2;

  const { setTooltip, resetTooltip } = useCompareRunsTooltip(cardConfig);

  const filteredData = isConfigured
    ? processParallelCoordinateData(
        previewData,
        cardConfig.selectedParams,
        cardConfig.selectedMetrics,
      )
    : [];

  if (!isConfigured) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        Select at least two metrics and params first
      </div>
    );
  }

  return filteredData.length ? (
    /* Avoid displaying empty set, otherwise parcoord-es crashes */
    <LazyParallelCoordinatesPlot
      selectedMetrics={cardConfig.selectedMetrics}
      selectedParams={cardConfig.selectedParams}
      data={filteredData}
      axesRotateThreshold={6}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  ) : null;
};
