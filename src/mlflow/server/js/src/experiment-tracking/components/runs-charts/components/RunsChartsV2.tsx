import { useMemo } from 'react';
import type { RunsChartsRunData } from './RunsCharts.common';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import { RunsChartType } from '../runs-charts.types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { getGridColumnSetup } from '../../../../common/utils/CssGrid.utils';
import { RunsChartCardSetFullscreenFn, RunsChartsChartsDragGroup } from './cards/ChartCard.common';
import { useDragAndDropElement } from 'common/hooks/useDragAndDropElement';
import { FormattedMessage } from 'react-intl';
import { Empty } from '@databricks/design-system';
import { RunsChartsCard } from './cards/RunsChartsCard';

export interface RunsChartsV2Props {
  sectionId: string;
  chartRunData: RunsChartsRunData[];
  cardsConfig: RunsChartsCardConfig[];
  isMetricHistoryLoading?: boolean;
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  onReorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  onInsertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  groupBy: string;
  sectionIndex: number;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
}

export const RunsChartsV2 = ({
  sectionId,
  chartRunData,
  cardsConfig,
  isMetricHistoryLoading,
  onRemoveChart,
  onStartEditChart,
  onReorderCharts,
  onInsertCharts,
  groupBy,
  sectionIndex,
  setFullScreenChart,
}: RunsChartsV2Props) => {
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

  const { dropTargetRef: dropTargetGeneralRef, isOver: isOverGeneral } = useDragAndDropElement({
    dragGroupKey: RunsChartsChartsDragGroup.GENERAL_AREA,
    dragKey: sectionId,
    onDrop: onInsertCharts,
  });

  const { dropTargetRef: dropTargetParallelRef, isOver: isOverParallel } = useDragAndDropElement({
    dragGroupKey: RunsChartsChartsDragGroup.PARALLEL_CHARTS_AREA,
    dragKey: sectionId,
    onDrop: onInsertCharts,
  });

  return (
    <div
      role="figure"
      ref={(element) => {
        // Use this element for drop target
        dropTargetGeneralRef?.(element);
        dropTargetParallelRef?.(element);
      }}
      css={{
        padding: cardsConfig.length === 0 ? theme.spacing.lg : 0,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {(isOverGeneral || isOverParallel) && cardsConfig.length === 0 && (
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
      {!parallelChartCards.length && !remainingChartCards.length && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
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
      {parallelChartCards.length ? (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            marginBottom: theme.spacing.md,
          }}
        >
          {parallelChartCards.map((cardConfig, index) => {
            const reorderProps = {
              onReorderWith: onReorderCharts,
              canMoveDown: index < parallelChartCards.length - 1,
              canMoveUp: index > 0,
              onMoveDown: () => onReorderCharts(cardConfig.uuid || '', parallelChartCards[index + 1]?.uuid || ''),
              onMoveUp: () => onReorderCharts(cardConfig.uuid || '', parallelChartCards[index - 1]?.uuid || ''),
            };
            return (
              <RunsChartsCard
                cardConfig={cardConfig}
                chartRunData={chartRunData}
                onStartEditChart={onStartEditChart}
                onRemoveChart={onRemoveChart}
                setFullScreenChart={setFullScreenChart}
                onReorderCharts={onReorderCharts}
                index={index}
                sectionIndex={sectionIndex}
                isMetricHistoryLoading={isMetricHistoryLoading}
                groupBy={groupBy}
                {...reorderProps}
              />
            );
          })}
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

          return (
            <RunsChartsCard
              cardConfig={cardConfig}
              chartRunData={chartRunData}
              onStartEditChart={onStartEditChart}
              onRemoveChart={onRemoveChart}
              setFullScreenChart={setFullScreenChart}
              onReorderCharts={onReorderCharts}
              index={index}
              sectionIndex={sectionIndex}
              isMetricHistoryLoading={isMetricHistoryLoading}
              groupBy={groupBy}
              {...reorderProps}
            />
          );
        })}
      </div>
    </div>
  );
};
