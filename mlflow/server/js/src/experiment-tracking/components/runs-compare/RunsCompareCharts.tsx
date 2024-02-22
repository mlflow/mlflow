import { useMemo } from 'react';
import { RunsCompareBarChartCard } from './cards/RunsCompareBarChartCard';
import { RunsCompareParallelChartCard } from './cards/RunsCompareParallelChartCard';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import type {
  RunsCompareBarCardConfig,
  RunsCompareCardConfig,
  RunsCompareContourCardConfig,
  RunsCompareLineCardConfig,
  RunsCompareParallelCardConfig,
  RunsCompareScatterCardConfig,
} from './runs-compare.types';

import { RunsCompareChartType } from './runs-compare.types';
import { RunsCompareScatterChartCard } from './cards/RunsCompareScatterChartCard';
import { RunsCompareContourChartCard } from './cards/RunsCompareContourChartCard';
import { RunsCompareLineChartCard } from './cards/RunsCompareLineChartCard';
import { useDesignSystemTheme } from '@databricks/design-system';
import { getGridColumnSetup } from '../../../common/utils/CssGrid.utils';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

export interface RunsCompareChartsProps {
  chartRunData: RunsChartsRunData[];
  cardsConfig: RunsCompareCardConfig[];
  isMetricHistoryLoading?: boolean;
  onRemoveChart: (chart: RunsCompareCardConfig) => void;
  onStartEditChart: (chart: RunsCompareCardConfig) => void;
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
      cardsConfig.filter((c) => c.type === RunsCompareChartType.PARALLEL),
      cardsConfig.filter((c) => c.type !== RunsCompareChartType.PARALLEL),
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
            <RunsCompareParallelChartCard
              key={`${cardConfig.uuid}-${index}`}
              config={cardConfig as RunsCompareParallelCardConfig}
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
          if (cardConfig.type === RunsCompareChartType.BAR) {
            return (
              <RunsCompareBarChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsCompareBarCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.LINE) {
            return (
              <RunsCompareLineChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsCompareLineCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                isMetricHistoryLoading={isMetricHistoryLoading}
                groupBy={groupBy}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.SCATTER) {
            return (
              <RunsCompareScatterChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsCompareScatterCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                {...reorderProps}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.CONTOUR) {
            return (
              <RunsCompareContourChartCard
                key={`${cardConfig.uuid}-${index}`}
                config={cardConfig as RunsCompareContourCardConfig}
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
