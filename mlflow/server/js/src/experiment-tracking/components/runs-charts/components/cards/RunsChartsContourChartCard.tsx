import { useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsContourCardConfig } from '../../runs-charts.types';
import {
  ChartRunsCountIndicator,
  RunsChartCardFullScreenProps,
  RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
} from './ChartCard.common';
import { RunsContourPlot } from '../RunsContourPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import {
  shouldEnableHidingChartsWithNoData,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartDataCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { intersection, uniq } from 'lodash';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';

export interface RunsChartsContourChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsContourCardConfig;
  chartRunData: RunsChartsRunData[];

  hideEmptyCharts?: boolean;

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsChartsContourChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  fullScreen,
  setFullScreenChart,
  hideEmptyCharts,
  ...reorderProps
}: RunsChartsContourChartCardProps) => {
  const title = `${config.xaxis.key} vs. ${config.yaxis.key} vs. ${config.zaxis.key}`;

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title,
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden);
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const isEmptyDataset = useMemo(() => {
    if (!shouldEnableHidingChartsWithNoData()) {
      return false;
    }
    const metricKeys = [config.xaxis.key, config.yaxis.key, config.zaxis.key];
    const metricsInRuns = slicedRuns.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }, [config, slicedRuns]);

  const { setTooltip, resetTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  const chartBody = (
    <div
      css={[
        styles.contourChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
    >
      <RunsContourPlot
        runsData={slicedRuns}
        xAxis={config.xaxis}
        yAxis={config.yaxis}
        zAxis={config.zaxis}
        useDefaultHoverBox={false}
        onHover={setTooltip}
        onUnhover={resetTooltip}
        selectedRunUuid={selectedRunUuid}
        onSetDownloadHandler={setImageDownloadHandler}
      />
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
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      // Disable fullscreen button if the chart is empty
      toggleFullScreenChart={isEmptyDataset ? undefined : toggleFullScreenChart}
      supportedDownloadFormats={['png', 'svg', 'csv']}
      onClickDownload={(format) => {
        const savedChartTitle = [config.xaxis.key, config.yaxis.key, config.zaxis.key].join('-');
        if (format === 'csv' || format === 'csv-full') {
          const paramsToExport = [];
          const metricsToExport = [];
          for (const axis of ['xaxis' as const, 'yaxis' as const, 'zaxis' as const]) {
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
  contourChartCardWrapper: {
    overflow: 'hidden',
  },
};
