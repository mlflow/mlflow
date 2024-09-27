import { useMemo, useState } from 'react';
import type { RunsChartsRunData } from './RunsCharts.common';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import { RunsChartType } from '../runs-charts.types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { getGridColumnSetup } from '../../../../common/utils/CssGrid.utils';
import { RunsChartCardSetFullscreenFn, RunsChartsChartsDragGroup } from './cards/ChartCard.common';
import { useDragAndDropElement } from '@mlflow/mlflow/src/common/hooks/useDragAndDropElement';
import { FormattedMessage } from 'react-intl';
import { Empty } from '@databricks/design-system';
import { RunsChartsCard } from './cards/RunsChartsCard';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsGlobalLineChartConfig } from '../../experiment-page/models/ExperimentPageUIState';

export interface RunsChartsProps {
  sectionId: string;
  chartRunData: RunsChartsRunData[];
  cardsConfig: RunsChartsCardConfig[];
  isMetricHistoryLoading?: boolean;
  onRemoveChart: (chart: RunsChartsCardConfig) => void;
  onStartEditChart: (chart: RunsChartsCardConfig) => void;
  onReorderCharts: (sourceChartUuid: string, targetChartUuid: string) => void;
  onInsertCharts: (sourceChartUuid: string, targetSectionId: string) => void;
  groupBy: RunsGroupByConfig | null;
  sectionIndex: number;
  setFullScreenChart: RunsChartCardSetFullscreenFn;
  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
}

export const RunsCharts = ({
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
  autoRefreshEnabled,
  hideEmptyCharts,
  globalLineChartConfig,
}: RunsChartsProps) => {
  const { theme } = useDesignSystemTheme();

  const [parallelChartCards, differenceChartCards, imageChartCards, remainingChartCards] = useMemo(() => {
    // Play it safe in case that cards config somehow failed to load
    if (!Array.isArray(cardsConfig)) {
      return [[], [], [], []];
    }
    return [
      cardsConfig.filter((c) => c.type === RunsChartType.PARALLEL),
      cardsConfig.filter((c) => c.type === RunsChartType.DIFFERENCE),
      cardsConfig.filter((c) => c.type === RunsChartType.IMAGE),
      cardsConfig.filter(
        (c) =>
          c.type !== RunsChartType.PARALLEL && c.type !== RunsChartType.DIFFERENCE && c.type !== RunsChartType.IMAGE,
      ),
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

  const isChartsEmpty =
    !parallelChartCards.length &&
    !differenceChartCards.length &&
    !imageChartCards.length &&
    !remainingChartCards.length;

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
      {isChartsEmpty && (
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
      {[differenceChartCards, parallelChartCards, imageChartCards].map((chartCards, index) => {
        if (chartCards.length) {
          return (
            <div
              key={`chart-cards-${index}`}
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.md,
                marginBottom: theme.spacing.md,
              }}
            >
              {chartCards.map((cardConfig, index) => {
                const reorderProps = {
                  onReorderWith: onReorderCharts,
                  canMoveDown: index < chartCards.length - 1,
                  canMoveUp: index > 0,
                  onMoveDown: () => onReorderCharts(cardConfig.uuid || '', chartCards[index + 1]?.uuid || ''),
                  onMoveUp: () => onReorderCharts(cardConfig.uuid || '', chartCards[index - 1]?.uuid || ''),
                };
                return (
                  <RunsChartsCard
                    cardConfig={cardConfig}
                    chartRunData={chartRunData}
                    onStartEditChart={onStartEditChart}
                    onRemoveChart={onRemoveChart}
                    setFullScreenChart={setFullScreenChart}
                    index={index}
                    sectionIndex={sectionIndex}
                    groupBy={groupBy}
                    autoRefreshEnabled={autoRefreshEnabled}
                    hideEmptyCharts={hideEmptyCharts}
                    key={`${cardConfig.uuid}-${index}-${sectionIndex}`}
                    {...reorderProps}
                  />
                );
              })}
            </div>
          );
        }
        return null;
      })}
      <div css={gridSetup}>
        {remainingChartCards.map((cardConfig, index) => {
          const reorderProps = {
            onReorderWith: onReorderCharts,
            canMoveDown: index < remainingChartCards.length - 1,
            canMoveUp: index > 0,
            previousChartUuid: remainingChartCards[index - 1]?.uuid,
            nextChartUuid: remainingChartCards[index + 1]?.uuid,
          };

          return (
            <RunsChartsCard
              cardConfig={cardConfig}
              chartRunData={chartRunData}
              onStartEditChart={onStartEditChart}
              onRemoveChart={onRemoveChart}
              setFullScreenChart={setFullScreenChart}
              index={index}
              sectionIndex={sectionIndex}
              groupBy={groupBy}
              autoRefreshEnabled={autoRefreshEnabled}
              hideEmptyCharts={hideEmptyCharts}
              key={`${cardConfig.uuid}-${index}-${sectionIndex}`}
              globalLineChartConfig={globalLineChartConfig}
              {...reorderProps}
            />
          );
        })}
      </div>
    </div>
  );
};
