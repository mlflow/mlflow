import { useMemo } from 'react';
import { RunsChartsBarChartCard } from '../runs-charts/components/cards/RunsChartsBarChartCard';
import { RunsChartsParallelChartCard } from '../runs-charts/components/cards/RunsChartsParallelChartCard';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsScatterCardConfig,
} from '../runs-charts/runs-charts.types';

import { RunsChartType } from '../runs-charts/runs-charts.types';
import { RunsChartsScatterChartCard } from '../runs-charts/components/cards/RunsChartsScatterChartCard';
import { RunsChartsContourChartCard } from '../runs-charts/components/cards/RunsChartsContourChartCard';
import { RunsChartsLineChartCard } from '../runs-charts/components/cards/RunsChartsLineChartCard';
import { useDesignSystemTheme } from '@databricks/design-system';
import { getGridColumnSetup } from '../../../common/utils/CssGrid.utils';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

export interface RunsCompareChartsProps {
  chartRunData: RunsChartsRunData[];
  cardsConfig: RunsChartsCardConfig[];
  isMetricHistoryLoading?: boolean;
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  onReorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  groupBy: string;
}

export const RunsCompareCharts = ({
  chartRunData,
  cardsConfig,
  isMetricHistoryLoading,
  onRemoveChart,
  onStartEditChart,
  onReorderCharts,
  groupBy,
}: RunsCompareChartsProps) => {
  const { theme } = useDesignSystemTheme();

  const [parallelChartCards, remainingChartCards] = useMemo(() => {
    // Play it safe in case that cards config somehow failed to load
    if (!Array.isArray(cardsConfig)) {
      return [[], []];
    }
    return [
      cardsConfig.filter((c) => c.type === RunsChartType.PARALLEL),
      cardsConfig.filter((c) => c.type !== RunsChartType.PARALLEL),
    ];
  }, [cardsConfig]);

  const gridSetup = useMemo(
    () =>
      getGridColumnSetup({
        maxColumns: 3,
        gap: theme.spacing.md,
        additionalBreakpoints: [{ breakpointWidth: 3 * 720, minColumnWidthForBreakpoint: 600 }],
      }),
    [theme],
  );

  return (
    <DndProvider backend={HTML5Backend}>
      {parallelChartCards.length ? (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            marginBottom: theme.spacing.md,
          }}
        >
          {parallelChartCards.map((cardConfig, index) => (
            <RunsChartsParallelChartCard
              key={`${cardConfig.uuid}-${index}`}
              config={cardConfig as RunsChartsParallelCardConfig}
              chartRunData={chartRunData}
              onEdit={() => onStartEditChart(cardConfig)}
              onDelete={() => onRemoveChart(cardConfig)}
              onReorderWith={onReorderCharts}
              canMoveDown={index < parallelChartCards.length - 1}
              canMoveUp={index > 0}
              onMoveDown={() => onReorderCharts(cardConfig.uuid || '', parallelChartCards[index + 1]?.uuid || '')}
              onMoveUp={() => onReorderCharts(cardConfig.uuid || '', parallelChartCards[index - 1]?.uuid || '')}
            />
          ))}
        </div>
      ) : null}
      <div css={gridSetup}>
        {remainingChartCards.map((cardConfig, index) => {
          const reorderProps = {
            onReorderWith: onReorderCharts,
            canMoveDown: index < remainingChartCards.length - 1,
            canMoveUp: index > 0,
            onMoveDown: () => onReorderCharts(cardConfig.uuid || '', remainingChartCards[index + 1]?.uuid || ''),
            onMoveUp: () => onReorderCharts(cardConfig.uuid || '', remainingChartCards[index - 1]?.uuid || ''),
          };
          if (cardConfig.type === RunsChartType.BAR) {
            return (
              <RunsChartsBarChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsChartsBarCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsChartType.LINE) {
            return (
              <RunsChartsLineChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsChartsLineCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                isMetricHistoryLoading={isMetricHistoryLoading}
                groupBy={groupBy}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsChartType.SCATTER) {
            return (
              <RunsChartsScatterChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsChartsScatterCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsChartType.CONTOUR) {
            return (
              <RunsChartsContourChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsChartsContourCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                {...reorderProps}
              />
            );
          }

          return null;
        })}
      </div>
    </DndProvider>
  );
};
