import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsMetricsBarPlot } from '../RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsBarCardConfig } from '../../runs-charts.types';
import type { RunsChartCardFullScreenProps, RunsChartCardVisibilityProps } from './ChartCard.common';
import { RunsChartCardWrapper, type RunsChartCardReorderProps, RunsChartsChartsDragGroup } from './ChartCard.common';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartDataCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { customMetricBehaviorDefs } from '../../../experiment-page/utils/customMetricBehaviorUtils';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';
import { Tag, Typography } from '@databricks/design-system';

export interface RunsChartsBarChartCardProps
  extends RunsChartCardReorderProps,
    RunsChartCardFullScreenProps,
    RunsChartCardVisibilityProps {
  config: RunsChartsBarCardConfig;
  chartRunData: RunsChartsRunData[];

  hideEmptyCharts?: boolean;

  onDelete: () => void;
  onEdit: () => void;
}

export const barChartCardDefaultMargin = {
  t: 24,
  b: 48,
  r: 0,
  l: 4,
  pad: 0,
};

export const RunsChartsBarChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  fullScreen,
  setFullScreenChart,
  hideEmptyCharts,
  isInViewport: isInViewportProp,
  ...reorderProps
}: RunsChartsBarChartCardProps) => {
  const dataKey = config.dataAccessKey ?? config.metricKey;

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: customMetricBehaviorDefs[config.metricKey]?.displayName ?? config.metricKey,
      subtitle: null,
    });
  };

  const slicedRuns = useMemo(
    () => chartRunData.filter(({ hidden, metrics }) => !hidden && metrics[dataKey]),
    [chartRunData, dataKey],
  );

  const isEmptyDataset = useMemo(() => {
    const metricsInRuns = slicedRuns.flatMap(({ metrics }) => Object.keys(metrics));
    return !metricsInRuns.includes(dataKey);
  }, [dataKey, slicedRuns]);

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  // If the chart is in fullscreen mode, we always render its body.
  // Otherwise, we only render the chart if it is in the viewport.
  const isInViewport = fullScreen || isInViewportProp;

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  const chartBody = (
    <div
      css={[
        styles.barChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
    >
      {isInViewport ? (
        <RunsMetricsBarPlot
          runsData={slicedRuns}
          metricKey={dataKey}
          displayRunNames={false}
          displayMetricKey={false}
          useDefaultHoverBox={false}
          margin={barChartCardDefaultMargin}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          selectedRunUuid={selectedRunUuid}
          onSetDownloadHandler={setImageDownloadHandler}
        />
      ) : null}
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  // Do not render the card if the chart is empty and the user has enabled hiding empty charts
  if (hideEmptyCharts && isEmptyDataset) {
    return null;
  }

  const chartTitle = (() => {
    if (config.datasetName) {
      return (
        <div css={{ flex: 1, display: 'flex', alignItems: 'center', overflow: 'hidden' }}>
          <Typography.Text title={config.metricKey} ellipsis bold>
            <Tag componentId="mlflow.charts.bar_card_title.dataset_tag" css={{ marginRight: 0 }}>
              {config.datasetName}
            </Tag>{' '}
            {config.metricKey}
          </Typography.Text>
        </div>
      );
    }
    return customMetricBehaviorDefs[config.metricKey]?.displayName ?? config.displayName ?? config.metricKey;
  })();

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={chartTitle}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      // Disable fullscreen button if the chart is empty
      toggleFullScreenChart={isEmptyDataset ? undefined : toggleFullScreenChart}
      supportedDownloadFormats={['png', 'svg', 'csv']}
      onClickDownload={(format) => {
        if (format === 'csv' || format === 'csv-full') {
          const runsToExport = [...slicedRuns].reverse();
          downloadChartDataCsv(runsToExport, [config.metricKey], [], config.metricKey);
          return;
        }
        imageDownloadHandler?.(format, config.metricKey);
      }}
      {...reorderProps}
    >
      {isEmptyDataset ? <RunsChartsNoDataFoundIndicator /> : chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  barChartCardWrapper: {
    overflow: 'hidden',
  },
};
