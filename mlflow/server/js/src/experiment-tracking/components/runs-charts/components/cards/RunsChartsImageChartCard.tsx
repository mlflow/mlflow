import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import {
  type RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
  RunsChartCardFullScreenProps,
  ChartRunsCountIndicator,
} from './ChartCard.common';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';
import { DifferenceViewPlot } from '../charts/DifferenceViewPlot';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import { useIntl, FormattedMessage } from 'react-intl';
import { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import { ImageGridPlot } from '../charts/ImageGridPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useImageSliderStepMarks } from '../../hooks/useImageSliderStepMarks';
import {
  DEFAULT_IMAGE_GRID_CHART_NAME,
  LOG_IMAGE_TAG_INDICATOR,
  NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';

export interface RunsChartsImageChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsImageCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
  groupBy: RunsGroupByConfig | null;
}

export const RunsChartsImageChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  groupBy,
  fullScreen,
  setFullScreenChart,
  ...reorderProps
}: RunsChartsImageChartCardProps) => {
  const { theme } = useDesignSystemTheme();
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }
    return () => {
      resizeObserver.disconnect();
    };
  }, [containerRef]);

  // Optimizations for smoother slider experience. Maintain a local copy of config, and update
  // the global state only after the user has finished dragging the slider.
  const [tmpConfig, setTmpConfig] = useState(config);
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();
  const updateStep = useCallback(
    (step: number) => {
      confirmChartCardConfiguration({ ...config, step } as RunsChartsImageCardConfig);
    },
    [config, confirmChartCardConfiguration],
  );
  const tmpStepChange = (step: number) => {
    setTmpConfig((conf) => ({ ...conf, step }));
  };

  const chartName = config.imageKeys.length === 1 ? config.imageKeys[0] : DEFAULT_IMAGE_GRID_CHART_NAME;

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: chartName,
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const setCardConfig = (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => {
    confirmChartCardConfiguration(setter(config));
  };

  const { stepMarks, maxMark, minMark } = useImageSliderStepMarks({
    data: slicedRuns,
    selectedImageKeys: config.imageKeys || [],
  });

  const stepMarkLength = Object.keys(stepMarks).length;

  useEffect(() => {
    // If there is only one step mark, set the step to the min mark
    if (stepMarkLength === 1 && tmpConfig.step !== minMark) {
      updateStep(minMark);
    }
  }, [minMark, stepMarkLength, tmpConfig.step, updateStep]);

  const shouldDisplayImageLimitIndicator =
    slicedRuns.filter((run) => {
      return run.tags[LOG_IMAGE_TAG_INDICATOR];
    }).length > NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE;

  const chartBody = (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: fullScreen ? '100%' : undefined,
        width: '100%',
        overflow: 'auto',
      }}
    >
      <div
        ref={containerRef}
        css={{
          cursor: 'pointer',
          height: `calc(100% - ${theme.spacing.md * 2}px)`,
          overflow: 'auto',
        }}
      >
        <ImageGridPlot
          previewData={slicedRuns}
          groupBy={groupBy}
          cardConfig={tmpConfig}
          setCardConfig={setCardConfig}
          containerWidth={containerWidth}
        />
      </div>
      <div
        css={{
          justifyContent: 'center',
          alignItems: 'center',
          display: 'inline-flex',
          gap: theme.spacing.md,
        }}
      >
        <div css={{ width: '350px' }}>
          <LineSmoothSlider
            defaultValue={tmpConfig.step}
            onChange={tmpStepChange}
            max={maxMark}
            min={minMark}
            marks={stepMarks}
            step={null}
            disabled={Object.keys(stepMarks).length <= 1}
            onAfterChange={updateStep}
          />
        </div>
      </div>
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={chartName}
      subtitle={
        shouldDisplayImageLimitIndicator && `Displaying images from first ${NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE} runs`
      }
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      toggleFullScreenChart={toggleFullScreenChart}
      {...reorderProps}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};
