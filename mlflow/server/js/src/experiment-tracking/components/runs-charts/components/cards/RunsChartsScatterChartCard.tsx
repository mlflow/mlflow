import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsScatterCardConfig } from '../../runs-charts.types';
import type {
  RunsChartCardFullScreenProps,
  RunsChartCardReorderProps,
  RunsChartCardVisibilityProps,
} from './ChartCard.common';
import { RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';
import { RunsScatterPlot } from '../RunsScatterPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartDataCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { intersection, uniq } from 'lodash';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';
import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';

export interface RunsChartsScatterChartCardProps
  extends RunsChartCardReorderProps,
    RunsChartCardVisibilityProps,
    RunsChartCardFullScreenProps {
  config: RunsChartsScatterCardConfig;
  chartRunData: RunsChartsRunData[];

  hideEmptyCharts?: boolean;

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsChartsScatterChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  fullScreen,
  setFullScreenChart,
  hideEmptyCharts,
  isInViewport: isInViewportProp,
  ...reorderProps
}: RunsChartsScatterChartCardProps) => {
  const { theme } = useDesignSystemTheme();
  const title = (() => {
    if (config.xaxis.datasetName || config.yaxis.datasetName) {
      return (
        <div css={{ flex: 1, display: 'flex', alignItems: 'center', overflow: 'hidden', gap: theme.spacing.xs }}>
          <Typography.Text title={config.xaxis.key} ellipsis bold>
            {config.xaxis.datasetName && (
              <>
                <Tag componentId="mlflow.charts.scatter_card_title.dataset_tag" css={{ marginRight: 0 }}>
                  {config.xaxis.datasetName}
                </Tag>{' '}
              </>
            )}
            {config.xaxis.key}
          </Typography.Text>
          <Typography.Text>vs</Typography.Text>
          <Typography.Text title={config.xaxis.key} ellipsis bold>
            {config.yaxis.datasetName && (
              <>
                <Tag componentId="mlflow.charts.scatter_card_title.dataset_tag" css={{ marginRight: 0 }}>
                  {config.yaxis.datasetName}
                </Tag>{' '}
              </>
            )}
            {config.yaxis.key}
          </Typography.Text>
        </div>
      );
    }
    return `${config.xaxis.key} vs. ${config.yaxis.key}`;
  })();

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title,
      subtitle: null,
    });
  };

  const slicedRuns = useMemo(() => chartRunData.filter(({ hidden }) => !hidden), [chartRunData]);

  const isEmptyDataset = useMemo(() => {
    const metricKeys = [config.xaxis.dataAccessKey ?? config.xaxis.key, config.yaxis.dataAccessKey ?? config.yaxis.key];
    const metricsInRuns = slicedRuns.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }, [config, slicedRuns]);

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  // If the chart is in fullscreen mode, we always render its body.
  // Otherwise, we only render the chart if it is in the viewport.
  const isInViewport = fullScreen || isInViewportProp;

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  const chartBody = (
    <div
      css={[
        styles.scatterChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
    >
      {isInViewport ? (
        <RunsScatterPlot
          runsData={slicedRuns}
          xAxis={config.xaxis}
          yAxis={config.yaxis}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          useDefaultHoverBox={false}
          selectedRunUuid={selectedRunUuid}
          onSetDownloadHandler={setImageDownloadHandler}
        />
      ) : null}
    </div>
  );

  // Do not render the card if the chart is empty and the user has enabled hiding empty charts
  if (hideEmptyCharts && isEmptyDataset) {
    return null;
  }

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={title}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      // Disable fullscreen button if the chart is empty
      toggleFullScreenChart={isEmptyDataset ? undefined : toggleFullScreenChart}
      supportedDownloadFormats={['png', 'svg', 'csv']}
      onClickDownload={(format) => {
        const savedChartTitle = [config.xaxis.key, config.yaxis.key].join('-');
        if (format === 'csv' || format === 'csv-full') {
          const paramsToExport = [];
          const metricsToExport = [];
          for (const axis of ['xaxis' as const, 'yaxis' as const]) {
            if (config[axis].type === 'PARAM') {
              paramsToExport.push(config[axis].key);
            } else {
              metricsToExport.push(config[axis].key);
            }
          }
          downloadChartDataCsv(slicedRuns, metricsToExport, paramsToExport, savedChartTitle);
          return;
        }
        imageDownloadHandler?.(format, savedChartTitle);
      }}
      {...reorderProps}
    >
      {isEmptyDataset ? <RunsChartsNoDataFoundIndicator /> : chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  scatterChartCardWrapper: {
    overflow: 'hidden',
  },
};
