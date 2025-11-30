import { useCallback, useMemo, useState } from 'react';
import type { RunEntity } from '../../../types';
import { ChartSectionConfig } from '../../../types';
import type { RunsChartsRunData } from '../../../components/runs-charts/components/RunsCharts.common';
import { RunsChartsTooltipWrapper } from '../../../components/runs-charts/hooks/useRunsChartsTooltip';
import { RunsChartsDraggableCardsGridContextProvider } from '../../../components/runs-charts/components/RunsChartsDraggableCardsGridContext';
import { RunsChartsSectionAccordion } from '../../../components/runs-charts/components/sections/RunsChartsSectionAccordion';
import {
  RunsChartsUIConfigurationContextProvider,
  useConfirmChartCardConfigurationFn,
  useInsertRunsChartsFn,
  useRemoveRunsChartFn,
  useReorderRunsChartsFn,
} from '../../../components/runs-charts/hooks/useRunsChartsUIConfiguration';
import {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartType,
} from '../../../components/runs-charts/runs-charts.types';
import { keyBy, uniq } from 'lodash';
import { MLFLOW_RUN_COLOR_TAG } from '../../../constants';
import { getStableColorForRun } from '../../../utils/RunNameUtils';
import type { ExperimentEvaluationRunsChartsUIConfiguration } from '../hooks/useExperimentEvaluationRunsChartsUIState';
import { useExperimentEvaluationRunsChartsUIState } from '../hooks/useExperimentEvaluationRunsChartsUIState';
import { useMemoDeep } from '../../../../common/hooks/useMemoDeep';
import { RunsChartsConfigureModal } from '../../../components/runs-charts/components/RunsChartsConfigureModal';
import { RunsChartsTooltipBody } from '../../../components/runs-charts/components/RunsChartsTooltipBody';
import { TableSkeleton } from '@databricks/design-system';
import { useGetExperimentRunColor } from '../../../components/experiment-page/hooks/useExperimentRunColor';
import { useExperimentEvaluationRunsRowVisibility } from '../hooks/useExperimentEvaluationRunsRowVisibility';

const SUPPORTED_CHART_TYPES = [
  RunsChartType.LINE,
  RunsChartType.BAR,
  RunsChartType.CONTOUR,
  RunsChartType.DIFFERENCE,
  RunsChartType.PARALLEL,
  RunsChartType.SCATTER,
];

const ExperimentEvaluationRunsPageChartsImpl = ({
  runs = [],
  chartUIState,
}: {
  runs?: RunEntity[];
  chartUIState: ExperimentEvaluationRunsChartsUIConfiguration;
}) => {
  const getRunColor = useGetExperimentRunColor();
  const { isRowHidden } = useExperimentEvaluationRunsRowVisibility();

  const chartData: RunsChartsRunData[] = useMemo(() => {
    return runs
      .filter((run) => run.info)
      .map<RunsChartsRunData>((run) => {
        const metricsByKey = keyBy(run.data?.metrics, 'key');
        const paramsByKey = keyBy(run.data?.params, 'key');
        const tagsByKey = keyBy(run.data?.tags, 'key');

        return {
          displayName: run.info.runName,
          images: {},
          metrics: metricsByKey,
          params: paramsByKey,
          tags: tagsByKey,
          uuid: run.info.runUuid,
          color: getRunColor(run.info.runUuid),
          runInfo: run.info,
          hidden: isRowHidden(run.info.runUuid),
        };
      });
  }, [runs, isRowHidden, getRunColor]);

  const availableMetricKeys = useMemo(() => uniq(chartData.flatMap((run) => Object.keys(run.metrics))), [chartData]);
  const availableParamKeys = useMemo(() => uniq(chartData.flatMap((run) => Object.keys(run.params))), [chartData]);

  const removeChart = useRemoveRunsChartFn();
  const reorderCharts = useReorderRunsChartsFn();
  const insertCharts = useInsertRunsChartsFn();
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();

  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsChartsCardConfig | null>(null);

  const addNewChartCard = useCallback(
    (metricSectionId: string) => (type: RunsChartType) =>
      setConfiguredCardConfig(RunsChartsCardConfig.getEmptyChartCardByType(type, false, undefined, metricSectionId)),
    [],
  );

  const contextValue = useMemo(() => ({ runs: chartData }), [chartData]);

  return (
    <div>
      <RunsChartsTooltipWrapper contextData={contextValue} component={RunsChartsTooltipBody}>
        <RunsChartsDraggableCardsGridContextProvider visibleChartCards={chartUIState.compareRunCharts}>
          <RunsChartsSectionAccordion
            supportedChartTypes={SUPPORTED_CHART_TYPES}
            compareRunSections={chartUIState.compareRunSections}
            compareRunCharts={chartUIState.compareRunCharts}
            reorderCharts={reorderCharts}
            insertCharts={insertCharts}
            chartData={chartData}
            startEditChart={setConfiguredCardConfig}
            removeChart={removeChart}
            addNewChartCard={addNewChartCard}
            search={chartUIState.chartsSearchFilter ?? ''}
            // TODO: get group by to work for line charts. simply passing
            // groupBy from the parent component does not work, as the line
            // chart requires the chart data to contain the
            // groupParentInfo key.
            groupBy={null}
            setFullScreenChart={() => {}}
            autoRefreshEnabled={chartUIState.autoRefreshEnabled}
            hideEmptyCharts={false}
            globalLineChartConfig={chartUIState.globalLineChartConfig}
          />
          {configuredCardConfig && (
            <RunsChartsConfigureModal
              chartRunData={chartData}
              metricKeyList={availableMetricKeys}
              paramKeyList={availableParamKeys}
              config={configuredCardConfig}
              onSubmit={(configuredCardConfig) => {
                confirmChartCardConfiguration({ ...configuredCardConfig, displayName: undefined });
                setConfiguredCardConfig(null);
              }}
              onCancel={() => setConfiguredCardConfig(null)}
              groupBy={null}
              supportedChartTypes={SUPPORTED_CHART_TYPES}
            />
          )}
        </RunsChartsDraggableCardsGridContextProvider>
      </RunsChartsTooltipWrapper>
    </div>
  );
};

export const ExperimentEvaluationRunsPageCharts = ({
  runs = [],
  experimentId,
}: {
  runs?: RunEntity[];
  experimentId: string;
}) => {
  // Get all unique metric keys from the runs
  const uniqueMetricKeys = useMemo(() => {
    const metricKeys = runs.flatMap((run) => run.data?.metrics?.map((metric) => metric.key) || []);
    return Array.from(new Set(metricKeys));
  }, [runs]);

  // The list of updated metrics is used to regenerate list of charts,
  // so it's memoized deeply (using deep equality) to avoid unnecessary re-calculation.
  const memoizedMetricKeys = useMemoDeep(() => uniqueMetricKeys, [uniqueMetricKeys]);

  const { chartUIState, loading, updateUIState } = useExperimentEvaluationRunsChartsUIState(
    memoizedMetricKeys,
    experimentId,
  );

  if (loading) {
    return <TableSkeleton lines={5} />;
  }

  return (
    <RunsChartsUIConfigurationContextProvider updateChartsUIState={updateUIState}>
      <ExperimentEvaluationRunsPageChartsImpl runs={runs} chartUIState={chartUIState} />
    </RunsChartsUIConfigurationContextProvider>
  );
};
