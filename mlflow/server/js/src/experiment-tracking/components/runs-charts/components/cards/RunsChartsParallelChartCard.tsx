import { Button, DropdownMenu, Typography } from '@databricks/design-system';
import { useCallback, useMemo } from 'react';
import { ReactComponent as ParallelChartSvg } from '../../../../../common/static/parallel-chart-placeholder.svg';
import type { RunsChartsRunData } from '../RunsCharts.common';
import LazyParallelCoordinatesPlot, { processParallelCoordinateData } from '../charts/LazyParallelCoordinatesPlot';
import { useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { RunsChartsParallelCardConfig } from '../../runs-charts.types';
import {
  type RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
  RunsChartCardFullScreenProps,
} from './ChartCard.common';
import { useIsInViewport } from '../../hooks/useIsInViewport';
import {
  shouldEnableDeepLearningUI,
  shouldEnableRunGrouping,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import { FormattedMessage } from 'react-intl';
import { useUpdateExperimentViewUIState } from '../../../experiment-page/contexts/ExperimentPageUIStateContext';

export interface RunsChartsParallelChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsParallelCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
  groupBy?: string;
}

/**
 * A placeholder component displayed before parallel coords chart is being configured by user
 */
const EmptyParallelCoordsPlaceholder = ({ onEdit }: { onEdit: () => void }) => {
  return (
    <div css={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 360 }}>
        <ParallelChartSvg />
        <Typography.Title css={{ marginTop: 16 }} color="secondary" level={3}>
          Compare parameter importance
        </Typography.Title>
        <Typography.Text css={{ marginBottom: 16 }} color="secondary">
          Use the parallel coordinates chart to compare how various parameters in model affect your model metrics.
        </Typography.Text>
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_cards_runscompareparallelchartcard.tsx_51"
          type="primary"
          onClick={onEdit}
        >
          Configure chart
        </Button>
      </div>
    </div>
  );
};

/**
 * A placeholder component displayed before parallel coords chart is being configured by user
 */
const UnsupportedDataPlaceholder = () => (
  <div css={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center' }}>
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 360 }}>
      <ParallelChartSvg />
      <Typography.Title css={{ marginTop: 16, textAlign: 'center' }} color="secondary" level={3}>
        <FormattedMessage
          defaultMessage="Parallel coordinates chart does not support aggregated string values."
          description="Experiment page > compare runs > parallel coordinates chart > unsupported string values warning > title"
        />
      </Typography.Title>
      <Typography.Text css={{ marginBottom: 16 }} color="secondary">
        <FormattedMessage
          defaultMessage="Use other parameters or disable run grouping to continue."
          description="Experiment page > compare runs > parallel coordinates chart > unsupported string values warning > description"
        />
      </Typography.Text>
    </div>
  </div>
);

export const RunsChartsParallelChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  groupBy,
  fullScreen,
  setFullScreenChart,
}: RunsChartsParallelChartCardProps) => {
  const updateUIState = useUpdateExperimentViewUIState();

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: 'Parallel Coordinates',
      subtitle: displaySubtitle ? subtitle : null,
    });
  };

  const configuredChartRunData = useMemo(() => {
    if (!shouldUseNewRunRowsVisibilityModel() || config?.showAllRuns) {
      return chartRunData;
    }
    return chartRunData?.filter(({ hidden }) => !hidden);
  }, [chartRunData, config?.showAllRuns]);

  const containsStringValues = useMemo(
    () =>
      config.selectedParams?.some(
        (paramKey) => configuredChartRunData?.some((dataTrace) => isNaN(Number(dataTrace.params[paramKey]?.value))),
        [config.selectedParams, configuredChartRunData],
      ),
    [config.selectedParams, configuredChartRunData],
  );

  const updateVisibleOnlySetting = useCallback(
    (showAllRuns: boolean) => {
      updateUIState((state) => {
        const newCompareRunCharts = state.compareRunCharts?.map((existingChartConfig) => {
          if (existingChartConfig.uuid === config.uuid) {
            const parallelChartConfig = existingChartConfig as RunsChartsParallelCardConfig;
            return { ...parallelChartConfig, showAllRuns };
          }
          return existingChartConfig;
        });

        return { ...state, compareRunCharts: newCompareRunCharts };
      });
    },
    [config.uuid, updateUIState],
  );

  const [isConfigured, parallelCoordsData] = useMemo(() => {
    const selectedParamsCount = config.selectedParams?.length || 0;
    const selectedMetricsCount = config.selectedMetrics?.length || 0;

    const configured = selectedParamsCount + selectedMetricsCount >= 2;

    // Prepare the data in the parcoord-es format
    const data = configured
      ? processParallelCoordinateData(configuredChartRunData, config.selectedParams, config.selectedMetrics)
      : [];

    return [configured, data];
  }, [config, configuredChartRunData]);

  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const { elementRef, isInViewport } = useIsInViewport({ enabled: usingV2ChartImprovements });

  const { setTooltip, resetTooltip, selectedRunUuid, closeContextMenu } = useRunsChartsTooltip(config);

  const containsUnsupportedValues = containsStringValues && groupBy && shouldEnableRunGrouping();
  const displaySubtitle = isConfigured && !containsUnsupportedValues;

  const subtitle = shouldUseNewRunRowsVisibilityModel() ? (
    <>
      {config.showAllRuns ? (
        <FormattedMessage
          defaultMessage="Showing all runs"
          description="Experiment page > compare runs > parallel chart > header > indicator for all runs shown"
        />
      ) : (
        <FormattedMessage
          defaultMessage="Showing only visible runs"
          description="Experiment page > compare runs > parallel chart > header > indicator for only visible runs shown"
        />
      )}
    </>
  ) : (
    <>Comparing {parallelCoordsData.length} runs</>
  );

  const chartBody = (
    <>
      {!isConfigured ? (
        <EmptyParallelCoordsPlaceholder onEdit={onEdit} />
      ) : containsUnsupportedValues ? (
        <UnsupportedDataPlaceholder />
      ) : parallelCoordsData.length ? (
        // Avoid displaying empty set, otherwise parcoord-es goes crazy
        <div
          css={[
            styles.parallelChartCardWrapper,
            {
              height: fullScreen ? '100%' : undefined,
            },
          ]}
          ref={elementRef}
        >
          {isInViewport ? (
            <LazyParallelCoordinatesPlot
              data={parallelCoordsData}
              selectedParams={config.selectedParams}
              selectedMetrics={config.selectedMetrics}
              onHover={setTooltip}
              onUnhover={resetTooltip}
              axesRotateThreshold={8}
              selectedRunUuid={selectedRunUuid}
              closeContextMenu={closeContextMenu}
            />
          ) : null}
        </div>
      ) : null}
    </>
  );

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title="Parallel Coordinates"
      subtitle={displaySubtitle ? subtitle : null}
      uuid={config.uuid}
      tooltip="The Parallel Coordinates Chart now only shows runs with columns that are either numbers or strings. If a column has string entries, the runs corresponding to the 30 most recent unique values will be shown."
      dragGroupKey={RunsChartsChartsDragGroup.PARALLEL_CHARTS_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
      toggleFullScreenChart={isConfigured && !containsUnsupportedValues ? toggleFullScreenChart : undefined}
      additionalMenuContent={
        shouldUseNewRunRowsVisibilityModel() ? (
          <>
            <DropdownMenu.Separator />
            <DropdownMenu.CheckboxItem checked={!config.showAllRuns} onClick={() => updateVisibleOnlySetting(false)}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show only visible"
                description="Experiment page > compare runs tab > chart header > move down option"
              />
            </DropdownMenu.CheckboxItem>
            <DropdownMenu.CheckboxItem checked={config.showAllRuns} onClick={() => updateVisibleOnlySetting(true)}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show all runs"
                description="Experiment page > compare runs tab > chart header > move down option"
              />
            </DropdownMenu.CheckboxItem>
          </>
        ) : null
      }
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  parallelChartCardWrapper: {
    // Set "display: flex" here (and "flex: 1" in the child element)
    // so the chart will grow in width and height
    display: 'flex',
    overflow: 'hidden',
    cursor: 'pointer',
  },
};
