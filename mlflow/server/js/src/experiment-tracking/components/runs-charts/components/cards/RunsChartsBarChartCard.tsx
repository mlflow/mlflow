import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsMetricsBarPlot } from '../RunsMetricsBarPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsBarCardConfig } from '../../runs-charts.types';
import { useIsInViewport } from '../../hooks/useIsInViewport';
import {
  shouldEnableDraggableChartsGridLayout,
  shouldEnableHidingChartsWithNoData,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import {
  RunsChartCardWrapper,
  type RunsChartCardReorderProps,
  RunsChartsChartsDragGroup,
  ChartRunsCountIndicator,
  RunsChartCardFullScreenProps,
  RunsChartCardVisibilityProps,
} from './ChartCard.common';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartDataCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { customMetricBehaviorDefs } from '../../../experiment-page/utils/customMetricBehaviorUtils';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';

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
  const usingDraggableChartsGridLayout = shouldEnableDraggableChartsGridLayout();

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: customMetricBehaviorDefs[config.metricKey]?.displayName ?? config.metricKey,
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      // If hiding empty charts is supported, we additionally filter out bars without recorded metric of interest
      if (shouldEnableHidingChartsWithNoData()) {
        return chartRunData.filter(({ hidden, metrics }) => !hidden && metrics[config.metricKey]);
      }
      return chartRunData.filter(({ hidden }) => !hidden);
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const isEmptyDataset = useMemo(() => {
    if (!shouldEnableHidingChartsWithNoData()) {
      return false;
    }
    const metricsInRuns = slicedRuns.flatMap(({ metrics }) => Object.keys(metrics));
    return !metricsInRuns.includes(config.metricKey);
  }, [config, slicedRuns]);

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  const { elementRef, isInViewport: isInViewportInternal } = useIsInViewport({
    enabled: !usingDraggableChartsGridLayout,
  });

  // If the chart is in fullscreen mode, we always render its body.
  // Otherwise, we only render the chart if it is in the viewport.
  // Viewport flag is either consumed from the prop (new approach) or calculated internally (legacy).
  const isInViewport = fullScreen || (isInViewportProp ?? isInViewportInternal);

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  const chartBody = (
    <div
      css={[
        styles.barChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
      ref={elementRef}
    >
      {isInViewport ? (
        <RunsMetricsBarPlot
          runsData={slicedRuns}
          metricKey={config.metricKey}
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

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={customMetricBehaviorDefs[config.metricKey]?.displayName ?? config.metricKey}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
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
