import { Button, Typography } from '@databricks/design-system';
import { useMemo } from 'react';
import { ReactComponent as ParallelChartSvg } from '../../../../common/static/parallel-chart-placeholder.svg';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import LazyParallelCoordinatesPlot, { processParallelCoordinateData } from '../charts/LazyParallelCoordinatesPlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import type { RunsCompareParallelCardConfig } from '../runs-compare.types';
import {
  type RunsCompareChartCardReorderProps,
  RunsCompareChartCardWrapper,
  RunsCompareChartsDragGroup,
} from './ChartCard.common';
import { useIsInViewport } from '../../runs-charts/hooks/useIsInViewport';
import { shouldEnableDeepLearningUI, shouldEnableDeepLearningUIPhase2 } from '../../../../common/utils/FeatureUtils';
import { FormattedMessage } from 'react-intl';

export interface RunsCompareParallelChartCardProps extends RunsCompareChartCardReorderProps {
  config: RunsCompareParallelCardConfig;
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
        <Button type="primary" onClick={onEdit}>
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

export const RunsCompareParallelChartCard = ({
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
}: RunsCompareParallelChartCardProps) => {
  const runGroupingEnabled = shouldEnableDeepLearningUIPhase2();

  const containsStringValues = useMemo(
    () =>
      config.selectedParams?.some(
        (paramKey) => chartRunData.some((dataTrace) => isNaN(Number(dataTrace.params[paramKey]?.value))),
        [config.selectedParams, chartRunData],
      ),
    [config.selectedParams, chartRunData],
  );

  const [isConfigured, parallelCoordsData] = useMemo(() => {
    const selectedParamsCount = config.selectedParams?.length || 0;
    const selectedMetricsCount = config.selectedMetrics?.length || 0;

    const configured = selectedParamsCount + selectedMetricsCount >= 2;

    // Prepare the data in the parcoord-es format
    const data = configured
      ? processParallelCoordinateData(chartRunData, config.selectedParams, config.selectedMetrics)
      : [];

    return [configured, data];
  }, [config, chartRunData]);

  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const { elementRef, isInViewport } = useIsInViewport({ enabled: usingV2ChartImprovements });

  const { setTooltip, resetTooltip, selectedRunUuid, closeContextMenu } = useRunsChartsTooltip(config);

  const containsUnsupportedValues = containsStringValues && groupBy && runGroupingEnabled;
  const displaySubtitle = isConfigured && !containsUnsupportedValues;

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title="Parallel Coordinates"
      subtitle={displaySubtitle ? <>Comparing {parallelCoordsData.length} runs</> : null}
      fullWidth
      uuid={config.uuid}
      tooltip="The Parallel Coordinates Chart now only shows runs with columns that are either numbers or strings. If a column has string entries, the runs corresponding to the 30 most recent unique values will be shown."
      dragGroupKey={RunsCompareChartsDragGroup.PARALLEL_CHARTS_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
    >
      {!isConfigured ? (
        <EmptyParallelCoordsPlaceholder onEdit={onEdit} />
      ) : containsUnsupportedValues ? (
        <UnsupportedDataPlaceholder />
      ) : parallelCoordsData.length ? (
        // Avoid displaying empty set, otherwise parcoord-es goes crazy
        <div css={styles.parallelChartCardWrapper} ref={elementRef}>
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
    </RunsCompareChartCardWrapper>
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
