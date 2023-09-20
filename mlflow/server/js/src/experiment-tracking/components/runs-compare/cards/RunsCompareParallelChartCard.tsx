import { Button, Typography } from '@databricks/design-system';
import { useMemo } from 'react';
import { ReactComponent as ParallelChartSvg } from '../../../../common/static/parallel-chart-placeholder.svg';
import type { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import LazyParallelCoordinatesPlot, {
  processParallelCoordinateData,
} from '../charts/LazyParallelCoordinatesPlot';
import { useCompareRunsTooltip } from '../hooks/useCompareRunsTooltip';
import type { RunsCompareParallelCardConfig } from '../runs-compare.types';
import { RunsCompareChartCardWrapper } from './ChartCard.common';

export interface RunsCompareParallelChartCardProps {
  config: RunsCompareParallelCardConfig;
  chartRunData: CompareChartRunData[];

  onDelete: () => void;
  onEdit: () => void;
}

/**
 * A placeholder component displayed before parallel coords chart is being configured by user
 */
const EmptyParallelCoordsPlaceholder = ({ onEdit }: { onEdit: () => void }) => {
  return (
    <div css={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 360 }}>
        <ParallelChartSvg />
        <Typography.Title css={{ marginTop: 16 }} color='secondary' level={3}>
          Compare parameter importance
        </Typography.Title>
        <Typography.Text css={{ marginBottom: 16 }} color='secondary'>
          Use the parallel coordinates chart to compare how various parameters in model affect your
          model metrics.
        </Typography.Text>
        <Button type='primary' onClick={onEdit}>
          Configure chart
        </Button>
      </div>
    </div>
  );
};

export const RunsCompareParallelChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
}: RunsCompareParallelChartCardProps) => {
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

  const { setTooltip, resetTooltip, selectedRunUuid, closeContextMenu } =
    useCompareRunsTooltip(config);

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={'Parallel Coordinates'}
      subtitle={isConfigured ? <>Comparing {parallelCoordsData.length} runs</> : null}
      fullWidth
      tooltip='The Parallel Coordinates Chart now only shows runs with columns that are either numbers or strings. If a column has string entries, the runs corresponding to the 30 most recent unique values will be shown.'
    >
      {!isConfigured ? (
        <EmptyParallelCoordsPlaceholder onEdit={onEdit} />
      ) : parallelCoordsData.length ? (
        // Avoid displaying empty set, otherwise parcoord-es goes crazy
        <div css={styles.parallelChartCardWrapper}>
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
