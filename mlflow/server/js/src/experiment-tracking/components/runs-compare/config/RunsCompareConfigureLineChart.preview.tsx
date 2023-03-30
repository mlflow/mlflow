import { Skeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { connect } from 'react-redux';
import { StateWithEntities } from '../../../../redux-types';
import { MetricHistoryByName } from '../../../types';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import { CompareRunsMetricsLinePlot } from '../charts/CompareRunsMetricsLinePlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import { useFetchCompareRunsMetricHistory } from '../hooks/useFetchCompareRunsMetricHistory';
import { RunsCompareLineCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureLineChartPreviewImpl = ({
  previewData,
  cardConfig,
  metricsByRunUuid,
}: {
  previewData: CompareChartRunData[];
  cardConfig: RunsCompareLineCardConfig;

  metricsByRunUuid: Record<string, MetricHistoryByName>;
}) => {
  const metricKeysToFetch = useMemo(() => [cardConfig.metricKey], [cardConfig.metricKey]);
  const { isLoading, error } = useFetchCompareRunsMetricHistory(
    metricKeysToFetch,
    previewData,
    metricsByRunUuid,
  );

  const previewDataWithHistory = useMemo<CompareChartRunData[]>(
    () =>
      previewData.map((previewRun) => ({
        ...previewRun,
        metricsHistory: metricsByRunUuid[previewRun.runInfo.run_uuid],
      })),
    [previewData, metricsByRunUuid],
  );

  const { resetTooltip, setTooltip } = useCompareRunsTooltip(cardConfig);

  if (isLoading) {
    return <Skeleton />;
  }

  if (error) {
    return <>Error occured!</>;
  }

  return (
    <CompareRunsMetricsLinePlot
      runsData={previewDataWithHistory}
      metricKey={cardConfig.metricKey}
      scaleType={cardConfig.scaleType}
      xAxisKey={cardConfig.xAxisKey}
      useDefaultHoverBox={false}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  );
};

const mapStateToProps = ({ entities: { metricsByRunUuid } }: StateWithEntities) => ({
  metricsByRunUuid,
});

/**
 * Preview of line chart used in compare runs configuration modal
 */
export const RunsCompareConfigureLineChartPreview = connect(mapStateToProps, undefined, undefined, {
  areStatesEqual: (nextState, prevState) =>
    nextState.entities.metricsByRunUuid === prevState.entities.metricsByRunUuid,
})(RunsCompareConfigureLineChartPreviewImpl);
