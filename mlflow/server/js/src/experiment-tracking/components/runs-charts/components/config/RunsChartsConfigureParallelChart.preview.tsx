import { RunsChartsRunData } from '../RunsCharts.common';
import LazyParallelCoordinatesPlot from '../charts/LazyParallelCoordinatesPlot';
import { processParallelCoordinateData } from '../../utils/parallelCoordinatesPlot.utils';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { RunsChartsParallelCardConfig } from '../../runs-charts.types';
import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { Empty, NoIcon } from '@databricks/design-system';

export const RunsChartsConfigureParallelChartPreview = ({
  previewData,
  cardConfig,
  groupBy,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsParallelCardConfig;
  groupBy: RunsGroupByConfig | null;
}) => {
  const selectedParamsCount = cardConfig.selectedParams?.length || 0;
  const selectedMetricsCount = cardConfig.selectedMetrics?.length || 0;

  const isConfigured = selectedParamsCount + selectedMetricsCount >= 2;

  const { setTooltip, resetTooltip } = useRunsChartsTooltip(cardConfig);

  const containsStringValues = useMemo(
    () =>
      cardConfig.selectedParams?.some(
        (paramKey) => previewData?.some((dataTrace) => isNaN(Number(dataTrace.params[paramKey]?.value))),
        [cardConfig.selectedParams, previewData],
      ),
    [cardConfig.selectedParams, previewData],
  );

  if (containsStringValues && groupBy) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', textAlign: 'center' }}>
        <FormattedMessage
          defaultMessage="Parallel coordinates chart does not support aggregated string values. Use other parameters or disable run grouping to continue."
          description="Experiment page > compare runs > parallel coordinates chart configuration modal > unsupported string values warning"
        />
      </div>
    );
  }

  const filteredData = isConfigured
    ? processParallelCoordinateData(previewData, cardConfig.selectedParams, cardConfig.selectedMetrics)
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
  ) : (
    <Empty
      description={
        <FormattedMessage
          defaultMessage="No matching data found for the available runs."
          description="Experiment tracking > runs charts > parallel coordinates chart preview > no data found description"
        />
      }
      image={<NoIcon />}
    />
  );
};
