import { LegacySkeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { connect } from 'react-redux';
import { ReduxState } from '../../../../redux-types';
import { MetricHistoryByName } from '../../../types';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsLinePlot } from '../../runs-charts/components/RunsMetricsLinePlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { useFetchCompareRunsMetricHistory } from '../hooks/useFetchCompareRunsMetricHistory';
import { RunsCompareLineCardConfig } from '../runs-compare.types';

export const RunsCompareConfigureLineChartPreviewImpl = ({
  previewData,
  cardConfig,
  metricsByRunUuid,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsCompareLineCardConfig;

  metricsByRunUuid: Record<string, MetricHistoryByName>;
}) => {
  const metricKeysToFetch = useMemo(() => [cardConfig.metricKey], [cardConfig.metricKey]);
  const { isLoading, error } = useFetchCompareRunsMetricHistory(metricKeysToFetch, previewData);

  const previewDataWithHistory = useMemo<RunsChartsRunData[]>(
    () =>
      previewData.map((previewRun) => ({
        ...previewRun,
        metricsHistory: metricsByRunUuid[previewRun.runInfo.run_uuid],
      })),
    [previewData, metricsByRunUuid],
  );

  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

  if (isLoading) {
    return <LegacySkeleton />;
  }

  if (error) {
    return <>Error occured!</>;
  }

  return (
    <RunsMetricsLinePlot
      runsData={previewDataWithHistory}
      metricKey={cardConfig.metricKey}
      scaleType={cardConfig.scaleType}
      lineSmoothness={cardConfig.lineSmoothness}
      xAxisKey={cardConfig.xAxisKey}
      useDefaultHoverBox={false}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  );
};

const mapStateToProps = ({ entities: { metricsByRunUuid } }: ReduxState) => ({
  metricsByRunUuid,
});

/**
 * Preview of line chart used in compare runs configuration modal
 */
export const RunsCompareConfigureLineChartPreview = connect(mapStateToProps, undefined, undefined, {
  areStatesEqual: (nextState, prevState) =>
    nextState.entities.metricsByRunUuid === prevState.entities.metricsByRunUuid,
})(RunsCompareConfigureLineChartPreviewImpl);
