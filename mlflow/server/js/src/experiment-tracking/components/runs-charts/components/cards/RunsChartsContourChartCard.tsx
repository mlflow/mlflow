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
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartDataCsv } from '../../../experiment-page/utils/experimentPage.common-utils';

export interface RunsChartsContourChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsContourCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsChartsContourChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  fullScreen,
  setFullScreenChart,
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
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

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
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
      toggleFullScreenChart={toggleFullScreenChart}
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
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  contourChartCardWrapper: {
    overflow: 'hidden',
  },
};
