import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartCardFullScreenProps } from './ChartCard.common';
import { type RunsChartCardReorderProps, RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import type { RunsChartsCardConfig, RunsChartsImageCardConfig } from '../../runs-charts.types';
import { ImageGridPlot } from '../charts/ImageGridPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useImageSliderStepMarks } from '../../hooks/useImageSliderStepMarks';
import {
  DEFAULT_IMAGE_GRID_CHART_NAME,
  LOG_IMAGE_TAG_INDICATOR,
  NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import type { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';

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

  // Optimizations for smoother slider experience. Maintain a local copy of config, and update
  // the global state only after the user has finished dragging the slider.
  const [tmpConfig, setTmpConfig] = useState(config);
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();
  const updateStep = useCallback(
    (newStep: number) => {
      // Skip updating base chart config if step is the same as current step.
      if (config.step === newStep) {
        return;
      }
      confirmChartCardConfiguration({ ...config, step: newStep } as RunsChartsImageCardConfig);
    },
    [config, confirmChartCardConfiguration],
  );
  const tmpStepChange = useCallback((newStep: number) => {
    setTmpConfig((currentConfig) => {
      // Skip updating temporary config if step is the same as current step.
      if (currentConfig.step === newStep) {
        return currentConfig;
      }
      return { ...currentConfig, step: newStep };
    });
  }, []);

  const chartName = config.imageKeys.length === 1 ? config.imageKeys[0] : DEFAULT_IMAGE_GRID_CHART_NAME;

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: chartName,
      subtitle: null,
    });
  };

  const slicedRuns = useMemo(() => chartRunData.filter(({ hidden }) => !hidden).reverse(), [chartRunData]);

  const setCardConfig = useCallback(
    (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => {
      confirmChartCardConfiguration(setter(config));
    },
    [config, confirmChartCardConfiguration],
  );

  const { stepMarks, maxMark, minMark } = useImageSliderStepMarks({
    data: slicedRuns,
    selectedImageKeys: config.imageKeys || [],
  });

  const stepMarkLength = Object.keys(stepMarks).length;

  useEffect(() => {
    // If there is only one step mark, set the step to the min mark
    if (stepMarkLength === 1 && tmpConfig.step !== minMark) {
      updateStep(minMark);
      tmpStepChange(minMark);
    }
  }, [minMark, stepMarkLength, tmpConfig.step, updateStep, tmpStepChange]);

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
        overflow: 'hidden',
        marginTop: theme.spacing.sm,
        gap: theme.spacing.md,
      }}
    >
      <div
        ref={containerRef}
        css={{
          flex: 1,
          overflow: 'auto',
        }}
      >
        <ImageGridPlot
          previewData={slicedRuns}
          groupBy={groupBy}
          cardConfig={tmpConfig}
          setCardConfig={setCardConfig}
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
        <div css={{ flex: 1 }}>
          <LineSmoothSlider
            value={tmpConfig.step}
            onChange={tmpStepChange}
            max={maxMark}
            min={minMark}
            marks={stepMarks}
            disabled={Object.keys(stepMarks).length <= 1}
            onAfterChange={updateStep}
            css={{
              '&[data-orientation="horizontal"]': { width: 'auto' },
            }}
          />
        </div>
      </div>
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  const cardBodyToRender = chartBody;

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
      {cardBodyToRender}
    </RunsChartCardWrapper>
  );
};
