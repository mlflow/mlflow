import { memo } from 'react';
import { useIsInViewport } from '../hooks/useIsInViewport';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import type { RunsChartsRunData } from './RunsCharts.common';
import { RunsChartsDraggableCard } from './RunsChartsDraggableCard';
import type { RunsChartCardSetFullscreenFn } from './cards/ChartCard.common';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsGlobalLineChartConfig } from '../../experiment-page/models/ExperimentPageUIState';

const VIRTUAL_GROUP_OVERSCAN_MARGIN = '400px';

export interface RunsChartsVirtualizedGroupProps {
  cards: RunsChartsCardConfig[];
  allCards: RunsChartsCardConfig[];
  groupStartIndex: number;
  columns: number;
  cardHeight: number;
  gap: number;
  chartRunData: RunsChartsRunData[];
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  groupBy: RunsGroupByConfig | null;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
  onResizeStart: (rect: DOMRect) => void;
  onResizeStop: () => void;
  onResize: (width: number, height: number) => void;
  onReorderWith: (draggedKey: string, targetDropKey: string) => void;
  getTranslateBy: (uuid?: string) => { x: number; y: number; overflowing: boolean } | undefined;
}

/**
 * Renders a group of chart cards with viewport-based virtualization.
 *
 * When the group is outside the viewport (with overscan), only a single
 * placeholder div is rendered to preserve the grid layout and scroll height.
 * When inside the viewport, the actual draggable chart cards are mounted.
 */
export const RunsChartsVirtualizedGroup = memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    cards,
    allCards,
    groupStartIndex,
    columns,
    cardHeight,
    gap,
    chartRunData,
    onRemoveChart,
    onStartEditChart,
    setFullScreenChart,
    groupBy,
    autoRefreshEnabled,
    hideEmptyCharts,
    globalLineChartConfig,
    onResizeStart,
    onResizeStop,
    onResize,
    onReorderWith,
    getTranslateBy,
  }: RunsChartsVirtualizedGroupProps) => {
    const { setElementRef, isInViewport } = useIsInViewport<HTMLDivElement>({
      enabled: true,
      rootMargin: VIRTUAL_GROUP_OVERSCAN_MARGIN,
    });

    const groupRowCount = Math.ceil(cards.length / columns);
    const groupHeight = groupRowCount * cardHeight + (groupRowCount - 1) * gap;

    // Render a placeholder that preserves the group's layout space.
    if (!isInViewport) {
      return (
        <div
          ref={setElementRef}
          data-testid="virtualized-chart-group-placeholder"
          css={{ gridColumn: '1 / -1' }}
          style={{ height: groupHeight }}
        />
      );
    }

    return (
      <>
        {/**
         * A zero-height sentinel that stays mounted while the group is visible so
         * the IntersectionObserver can keep tracking it. It spans the full grid width
         * and creates a negligible extra grid gap.
         */}
        <div
          ref={setElementRef}
          data-testid="virtualized-chart-group-sentinel"
          css={{ gridColumn: '1 / -1' }}
          style={{ height: 0 }}
        />
        {cards.map((cardConfig, localIndex) => {
          const index = groupStartIndex + localIndex;
          return (
            <RunsChartsDraggableCard
              key={cardConfig.uuid}
              uuid={cardConfig.uuid ?? ''}
              onResizeStart={onResizeStart}
              onResizeStop={onResizeStop}
              onResize={onResize}
              cardConfig={cardConfig}
              chartRunData={chartRunData}
              onReorderWith={onReorderWith}
              translateBy={getTranslateBy(cardConfig.uuid)}
              index={index}
              height={cardHeight}
              canMoveDown={Boolean(allCards[index + 1])}
              canMoveUp={Boolean(allCards[index - 1])}
              canMoveToTop={index > 0}
              canMoveToBottom={index < allCards.length - 1}
              previousChartUuid={allCards[index - 1]?.uuid}
              nextChartUuid={allCards[index + 1]?.uuid}
              hideEmptyCharts={hideEmptyCharts}
              firstChartUuid={allCards[0]?.uuid}
              lastChartUuid={allCards[allCards.length - 1]?.uuid}
              onRemoveChart={onRemoveChart}
              onStartEditChart={onStartEditChart}
              setFullScreenChart={setFullScreenChart}
              groupBy={groupBy}
              autoRefreshEnabled={autoRefreshEnabled}
              globalLineChartConfig={globalLineChartConfig}
            />
          );
        })}
      </>
    );
  },
);
