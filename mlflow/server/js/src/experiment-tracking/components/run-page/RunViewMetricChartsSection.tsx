import { RunInfoEntity, RunViewMetricConfig } from '../../types';
import { ChartRefreshManager } from './useChartRefreshManager';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { getGridColumnSetup } from '../../../common/utils/CssGrid.utils';
import { useDragAndDropElement } from '../../../common/hooks/useDragAndDropElement';
import { RunViewMetricChart } from './RunViewMetricChart';

export const RunViewMetricChartsSection = ({
  section,
  metricConfigs,
  onInsertChart,
  onReorderChart,
  canMoveUp,
  canMoveDown,
  moveChartDown,
  moveChartUp,
  runInfo,
  chartRefreshManager,
}: {
  section: string;
  metricConfigs: RunViewMetricConfig[];
  onInsertChart: (fromMetricKey: string, toGroup: string) => void;
  onReorderChart: (fromMetricKey: string, toMetricKey: string) => void;
  canMoveUp: (config: RunViewMetricConfig) => boolean;
  canMoveDown: (config: RunViewMetricConfig) => boolean;
  moveChartDown: (config: RunViewMetricConfig) => void;
  moveChartUp: (config: RunViewMetricConfig) => void;
  runInfo: RunInfoEntity;
  chartRefreshManager: ChartRefreshManager;
}) => {
  const { theme } = useDesignSystemTheme();

  const gridSetup = useMemo(
    () => ({
      ...getGridColumnSetup({
        maxColumns: 3,
        gap: theme.spacing.lg,
        additionalBreakpoints: [{ breakpointWidth: 3 * 720, minColumnWidthForBreakpoint: 720 }],
      }),
      overflow: 'hidden',
    }),
    [theme],
  );

  const { dragPreviewRef, dropTargetRef, isOver } = useDragAndDropElement({
    dragGroupKey: 'metricCharts',
    dragKey: section,
    onDrop: onInsertChart,
  });

  return (
    <div
      role="figure"
      ref={(element) => {
        // Use this element for both drag preview and drop target
        dragPreviewRef?.(element);
        dropTargetRef?.(element);
      }}
      css={{
        padding: metricConfigs.length === 0 ? theme.spacing.lg : 0,
        background: theme.colors.backgroundPrimary,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {isOver && (
        // Visual overlay for target drop element
        <div
          css={{
            position: 'absolute',
            inset: 0,
            backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
            border: `2px dashed ${theme.colors.blue400}`,
            opacity: 0.75,
          }}
        />
      )}
      <div css={gridSetup}>
        {metricConfigs.map((metricConfig, index) => (
          <RunViewMetricChart
            // Use both metric name and index as a key,
            // charts needs to be rerendered when order is changed
            key={`${metricConfig.metricKey}-${index}`}
            dragGroupKey="metricCharts"
            metricKey={metricConfig.metricKey}
            runInfo={runInfo}
            onReorderWith={onReorderChart}
            canMoveUp={canMoveUp(metricConfig)}
            canMoveDown={canMoveDown(metricConfig)}
            onMoveDown={() => moveChartDown(metricConfig)}
            onMoveUp={() => moveChartUp(metricConfig)}
            chartRefreshManager={chartRefreshManager}
          />
        ))}
      </div>
    </div>
  );
};
