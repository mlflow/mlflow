import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useUpdateRunsChartsUIConfiguration } from '../hooks/useRunsChartsUIConfiguration';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import type { RunsChartsRunData } from './RunsCharts.common';
import { createEmptyChartCardPredicate } from './RunsCharts.common';
import { useMediaQuery } from '@databricks/web-shared/hooks';
import { Global } from '@emotion/react';
import { FormattedMessage } from 'react-intl';
import type { ChartSectionConfig } from '../../../types';
import { RunsChartsDraggableCard } from './RunsChartsDraggableCard';
import {
  useRunsChartsDraggableGridActionsContext,
  useRunsChartsDraggableGridStateContext,
} from './RunsChartsDraggableCardsGridContext';
import { RunsChartsDraggablePreview } from './RunsChartsDraggablePreview';
import { DRAGGABLE_CARD_TRANSITION_NAME, type RunsChartCardSetFullscreenFn } from './cards/ChartCard.common';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsGlobalLineChartConfig } from '../../experiment-page/models/ExperimentPageUIState';
import { useVirtualizer } from '@tanstack/react-virtual';

const rowHeightSuggestions = [300, 330, 360, 400, 500];

const getColumnSuggestions = (containerWidth: number, gapSize = 8) =>
  [1, 2, 3, 4, 5].map((n) => ({
    cols: n,
    width: (containerWidth - (n - 1) * gapSize) / n,
  }));

const PlaceholderSymbol = Symbol('placeholder');

/**
 * Threshold for the number of chart cards above which:
 * 1. Virtualization is enabled (only visible rows are rendered to the DOM)
 * 2. The "hide empty charts" filter is skipped to avoid expensive per-card predicate scans
 */
const VIRTUALIZATION_THRESHOLD = 50;

/**
 * Maximum height for the virtualized scroll container. When the grid contains
 * more cards than VIRTUALIZATION_THRESHOLD, the grid is rendered inside a
 * scrollable container capped at this height so that the browser never has to
 * mount thousands of DOM nodes simultaneously.
 */
const VIRTUALIZED_CONTAINER_MAX_HEIGHT = '80vh';

interface RunsChartsDraggableCardsGridProps {
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  sectionConfig: ChartSectionConfig;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  sectionId: string;
  groupBy: RunsGroupByConfig | null;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
  chartRunData: RunsChartsRunData[];
  cardsConfig: RunsChartsCardConfig[];
}

// Renders draggable cards grid in a single chart section
export const RunsChartsDraggableCardsGridSection = memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    cardsConfig,
    sectionConfig,
    chartRunData,
    sectionId,
    hideEmptyCharts,
    ...cardProps
  }: RunsChartsDraggableCardsGridProps) => {
    const { theme } = useDesignSystemTheme();

    // If below medium breakpoint, display only 1 card per row.
    // Otherwise, use section configuration or fall back to 3 columns.
    const isCompactMode = useMediaQuery(`(max-width: ${theme.responsive.breakpoints.md}px)`);
    const columns = isCompactMode ? 1 : (sectionConfig.columns ?? 3);

    // Use card height from the section configuration or fall back to 360 pixels.
    const cardHeight = sectionConfig.cardHeight ?? 360;

    const gridBoxRef = useRef<HTMLDivElement | null>(null);

    const { draggedCardUuid, isDragging } = useRunsChartsDraggableGridStateContext();
    const { setTargetSection, setTargetPosition, onSwapCards } = useRunsChartsDraggableGridActionsContext();

    const updateUIState = useUpdateRunsChartsUIConfiguration();

    const setColumns = useCallback(
      (columnCount: number) => {
        updateUIState((current) => {
          const section = current.compareRunSections?.find((section) => section.uuid === sectionId);
          if (!section) {
            return current;
          }

          return {
            ...current,
            compareRunSections: current.compareRunSections?.map((s) => {
              if (s.uuid === sectionId) {
                return {
                  ...s,
                  columns: columnCount,
                };
              }
              return s;
            }),
          };
        });
      },
      [sectionId, updateUIState],
    );

    const setCardHeight = useCallback(
      (height: number) => {
        updateUIState((current) => {
          const section = current.compareRunSections?.find((section) => section.uuid === sectionId);
          if (!section) {
            return current;
          }

          return {
            ...current,
            compareRunSections: current.compareRunSections?.map((s) => {
              if (s.uuid === sectionId) {
                return {
                  ...s,
                  cardHeight: height,
                };
              }
              return s;
            }),
          };
        });
      },
      [sectionId, updateUIState],
    );

    const lastElementCount = useRef<number>(0);

    const [positionInSection, setPositionInSection] = useState<number | null>(null);
    const [resizePreview, setResizePreview] = useState<null | Partial<DOMRect>>(null);

    lastElementCount.current = cardsConfig.length;
    const position = !draggedCardUuid ? null : positionInSection;

    // Helper function that calculates the x, y coordinates of a card based on its position in the grid
    const findCoords = useCallback(
      (position) => {
        const gap = theme.spacing.sm;
        const rowCount = Math.ceil(cardsConfig.length / columns);

        const row = Math.floor(position / columns);
        const col = position % columns;

        const colGapsCount = columns - 1;

        const passedColGaps = col;
        const passedRowGaps = row;
        const rect = gridBoxRef.current?.getBoundingClientRect();

        const singleWidth = ((rect?.width ?? 0) - colGapsCount * gap) / columns;

        const overflowing = row >= rowCount;

        return {
          overflowing,
          row,
          col,
          x: col * singleWidth + passedColGaps * gap,
          y: row * cardHeight + passedRowGaps * gap,
        };
      },
      [columns, cardHeight, theme, cardsConfig.length],
    );

    const cardsToRender = useMemo(() => {
      // Skip the expensive empty-chart filtering when the card count exceeds
      // the virtualization threshold. At that scale the per-card predicate scan
      // (which walks every visible run's metric data) becomes a major bottleneck
      // and compounds the rendering cost.
      if (!hideEmptyCharts || cardsConfig.length > VIRTUALIZATION_THRESHOLD) {
        return cardsConfig;
      }
      const isEmptyChartCard = createEmptyChartCardPredicate(chartRunData);
      return cardsConfig.filter((cardConfig) => !isEmptyChartCard(cardConfig));
    }, [cardsConfig, chartRunData, hideEmptyCharts]);

    // Calculate the transforms for each card based on the dragged card and its position.
    const cardTransforms = useMemo(() => {
      if (!draggedCardUuid || position === null) {
        return {};
      }

      const result: Record<string, { x: number; y: number; overflowing: boolean }> = {};

      const newArray: (RunsChartsCardConfig | typeof PlaceholderSymbol)[] = cardsToRender.slice();
      const fromIndex = cardsToRender.findIndex((x) => x.uuid === draggedCardUuid);
      const toIndex = position;

      if (fromIndex !== -1) {
        // If the card is dragged within same section, just rearrange the cards
        newArray.splice(fromIndex, 1);
        newArray.splice(toIndex, 0, cardsToRender[fromIndex]);
      } else {
        // If the card is dragged from another section, insert empty placeholder element
        newArray.splice(toIndex, 0, PlaceholderSymbol);
      }

      for (const cardConfig of cardsToRender) {
        const newIndex = newArray.indexOf(cardConfig);
        const oldIndex = cardsToRender.indexOf(cardConfig);

        const oldCoords = findCoords(oldIndex);
        const newCoords = findCoords(newIndex);

        // If the card is not moving, skip it
        if (newCoords.x === oldCoords.x && newCoords.y === oldCoords.y) {
          continue;
        }

        // Calculate the delta between the old and new positions
        const deltaX = newCoords.x - oldCoords.x;
        const deltaY = newCoords.y - oldCoords.y;

        if (cardConfig.uuid) {
          result[cardConfig.uuid] = {
            x: deltaX,
            y: deltaY,
            overflowing: newCoords.overflowing,
          };
        }
      }

      return result;
    }, [draggedCardUuid, position, cardsToRender, findCoords]);

    // Calculate the preview (placeholder) for the dragged card based on its new position
    const dragPreview = useMemo(() => {
      if (position === null) {
        return null;
      }
      if (cardsToRender.length === 0) {
        return { x: 0, y: 0, width: '100%', height: '100%' };
      }
      const { x, y } = findCoords(position);
      const colGapsCount = columns - 1;
      const rect = gridBoxRef.current?.getBoundingClientRect();

      const singleWidth = ((rect?.width ?? 0) - colGapsCount * theme.spacing.sm) / columns;

      const height = cardHeight;
      return { x, y, width: singleWidth, height };
    }, [position, findCoords, columns, cardHeight, theme, cardsToRender.length]);

    const mouseMove = useCallback(
      (e: React.MouseEvent) => {
        if (!isDragging() || !gridBoxRef.current) {
          return;
        }

        const rect = gridBoxRef.current.getBoundingClientRect();
        const rowCount = Math.ceil(lastElementCount.current / columns);

        setTargetSection(sectionId);
        const pos =
          Math.floor(((e.clientY - rect.top) / rect.height) * rowCount) * columns +
          Math.floor(((e.clientX - rect.left) / rect.width) * columns);

        setPositionInSection(pos);
        setTargetPosition(pos);
      },

      [columns, isDragging, sectionId, setTargetSection, setTargetPosition],
    );

    const [columnSuggestions, setColumnSuggestions] = useState<{ cols: number; width: number }[]>([]);

    const immediateColSuggestion = useRef<number | null>(null);
    const immediateRowSuggestion = useRef<number | null>(null);

    const onResizeStart = useCallback((rect: DOMRect) => {
      const gridBoxRefSize = gridBoxRef.current?.getBoundingClientRect();
      if (!gridBoxRefSize) {
        return;
      }

      setResizePreview({
        x: rect.left - gridBoxRefSize.left,
        y: rect.top - gridBoxRefSize.top,
        width: rect.width,
        height: rect.height,
      });

      setColumnSuggestions(getColumnSuggestions(gridBoxRefSize.width));
    }, []);

    const onResizeStop = useCallback(() => {
      setColumns(immediateColSuggestion.current ?? columns);
      setCardHeight(immediateRowSuggestion.current ?? cardHeight);
      setResizePreview(null);
    }, [cardHeight, columns, setCardHeight, setColumns]);

    const onResize = useCallback(
      (width: number, height: number) => {
        const columnSuggestion = columnSuggestions.reduce((prev, curr) =>
          Math.abs(curr.width - width) < Math.abs(prev.width - width) ? curr : prev,
        );

        const rowHeightSuggestion = rowHeightSuggestions.reduce((prev, curr) =>
          Math.abs(curr - height) < Math.abs(prev - height) ? curr : prev,
        );

        immediateColSuggestion.current = columnSuggestion.cols;
        immediateRowSuggestion.current = rowHeightSuggestion;

        setResizePreview((current) => {
          if (!current) {
            return null;
          }
          if (current.width !== columnSuggestion.width || current.height !== rowHeightSuggestion) {
            return { ...current, width: columnSuggestion.width, height: rowHeightSuggestion };
          }
          return current;
        });
      },
      [columnSuggestions],
    );

    // Determine whether virtualization should be active. Virtualization is
    // enabled when the card count exceeds the threshold AND no drag/resize
    // operation is in progress (drag-and-drop requires all cards to be in the
    // DOM so that position calculations and reorder references remain correct).
    const isLargeGrid = cardsToRender.length > VIRTUALIZATION_THRESHOLD;
    const shouldVirtualize = isLargeGrid && !draggedCardUuid && !resizePreview;

    // Group the flat cards list into rows for the virtualizer.
    // Only perform the O(N) partitioning if we are in the large grid path.
    const rows = useMemo(() => {
      if (!isLargeGrid) {
        return [];
      }
      const result: RunsChartsCardConfig[][] = [];
      for (let i = 0; i < cardsToRender.length; i += columns) {
        result.push(cardsToRender.slice(i, i + columns));
      }
      return result;
    }, [cardsToRender, columns, isLargeGrid]);

    const gapSize = theme.spacing.sm;

    // Ref for the scroll container that wraps the virtualized grid.
    const scrollContainerRef = useRef<HTMLDivElement | null>(null);

    const rowVirtualizer = useVirtualizer({
      count: isLargeGrid ? rows.length : 0,
      getScrollElement: () => scrollContainerRef.current,
      estimateSize: () => cardHeight + gapSize,
      overscan: 3,
      // Only enable when virtualization is active
      enabled: shouldVirtualize,
    });

    // Shared card rendering function used by both virtualized and
    // non-virtualized paths. `globalIndex` is the card's position in the
    // flat `cardsToRender` array so that drag-and-drop neighbour references
    // remain correct.
    const renderCard = useCallback(
      (cardConfig: RunsChartsCardConfig, globalIndex: number) => (
        <RunsChartsDraggableCard
          key={cardConfig.uuid}
          uuid={cardConfig.uuid ?? ''}
          translateBy={cardTransforms[cardConfig.uuid ?? '']}
          onResizeStart={onResizeStart}
          onResizeStop={onResizeStop}
          onResize={onResize}
          cardConfig={cardConfig}
          chartRunData={chartRunData}
          onReorderWith={onSwapCards}
          index={globalIndex}
          height={cardHeight}
          canMoveDown={Boolean(cardsToRender[globalIndex + 1])}
          canMoveUp={Boolean(cardsToRender[globalIndex - 1])}
          canMoveToTop={globalIndex > 0}
          canMoveToBottom={globalIndex < cardsToRender.length - 1}
          previousChartUuid={cardsToRender[globalIndex - 1]?.uuid}
          nextChartUuid={cardsToRender[globalIndex + 1]?.uuid}
          hideEmptyCharts={hideEmptyCharts}
          firstChartUuid={cardsToRender[0]?.uuid}
          lastChartUuid={cardsToRender[cardsToRender.length - 1]?.uuid}
          {...cardProps}
        />
      ),
      [
        cardTransforms,
        onResizeStart,
        onResizeStop,
        onResize,
        chartRunData,
        onSwapCards,
        cardHeight,
        cardsToRender,
        hideEmptyCharts,
        cardProps,
      ],
    );

    // ── Large grid rendering path (with optional virtualization) ────────
    if (isLargeGrid) {
      return (
        <div
          ref={scrollContainerRef}
          css={{
            maxHeight: VIRTUALIZED_CONTAINER_MAX_HEIGHT,
            overflow: 'auto',
          }}
          data-testid="virtualized-chart-cards-scroll-container"
        >
          <div
            ref={gridBoxRef}
            css={{ position: 'relative' }}
            style={{
              // Total height of all rows so the scrollbar reflects the full grid size
              height: shouldVirtualize ? rowVirtualizer.getTotalSize() : rows.length * (cardHeight + gapSize),
              ...(draggedCardUuid && {
                [DRAGGABLE_CARD_TRANSITION_NAME]: 'transform 0.1s',
              }),
            }}
            data-testid="draggable-chart-cards-grid"
            onMouseMove={mouseMove}
            onMouseLeave={() => {
              setPositionInSection(null);
            }}
          >
            {(draggedCardUuid || resizePreview) && (
              <Global
                styles={{
                  'body, :host': {
                    userSelect: 'none',
                  },
                }}
              />
            )}
            {shouldVirtualize
              ? rowVirtualizer.getVirtualItems().map((virtualRow) => {
                  const rowCards = rows[virtualRow.index];
                  const rowStartIndex = virtualRow.index * columns;
                  return (
                    <div
                      key={virtualRow.index}
                      data-testid="virtualized-chart-row"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: virtualRow.size,
                        transform: `translateY(${virtualRow.start}px)`,
                      }}
                      css={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${columns}, 1fr)`,
                        gap: gapSize,
                      }}
                    >
                      {rowCards.map((cardConfig, colIndex) => renderCard(cardConfig, rowStartIndex + colIndex))}
                    </div>
                  );
                })
              : rows.map((rowCards, rowIndex) => {
                  const rowStartIndex = rowIndex * columns;
                  return (
                    <div
                      key={rowIndex}
                      data-testid="virtualized-chart-row"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: cardHeight + gapSize,
                        transform: `translateY(${rowIndex * (cardHeight + gapSize)}px)`,
                      }}
                      css={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${columns}, 1fr)`,
                        gap: gapSize,
                      }}
                    >
                      {rowCards.map((cardConfig, colIndex) => renderCard(cardConfig, rowStartIndex + colIndex))}
                    </div>
                  );
                })}
            {dragPreview && <RunsChartsDraggablePreview {...dragPreview} />}
            {resizePreview && <RunsChartsDraggablePreview {...resizePreview} />}
          </div>
        </div>
      );
    }

    // ── Small grid rendering path (original behavior) ───────────────────
    return (
      <div
        ref={gridBoxRef}
        css={[
          { position: 'relative' },
          cardsToRender.length > 0 && {
            display: 'grid',
            gap: theme.spacing.sm,
          },
        ]}
        style={{
          gridTemplateColumns: 'repeat(' + columns + ', 1fr)',
          ...(draggedCardUuid && {
            [DRAGGABLE_CARD_TRANSITION_NAME]: 'transform 0.1s',
          }),
        }}
        data-testid="draggable-chart-cards-grid"
        onMouseMove={mouseMove}
        onMouseLeave={() => {
          setPositionInSection(null);
        }}
      >
        {(draggedCardUuid || resizePreview) && (
          <Global
            styles={{
              'body, :host': {
                userSelect: 'none',
              },
            }}
          />
        )}
        {cardsToRender.length === 0 && (
          <div css={{ display: 'flex', justifyContent: 'center', minHeight: 160 }}>
            <Empty
              title={
                <FormattedMessage
                  defaultMessage="No charts in this section"
                  description="Runs compare page > Charts tab > No charts placeholder title"
                />
              }
              description={
                <FormattedMessage
                  defaultMessage="Click 'Add chart' or drag and drop to add charts here."
                  description="Runs compare page > Charts tab > No charts placeholder description"
                />
              }
            />
          </div>
        )}
        {cardsToRender.map((cardConfig, index) => renderCard(cardConfig, index))}
        {dragPreview && <RunsChartsDraggablePreview {...dragPreview} />}
        {resizePreview && <RunsChartsDraggablePreview {...resizePreview} />}
      </div>
    );
  },
);
