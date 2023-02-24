import { useMemo } from 'react';
import { Theme } from '@emotion/react';
import { RunsCompareBarChartCard } from './cards/RunsCompareBarChartCard';
import { RunsCompareParallelChartCard } from './cards/RunsCompareParallelChartCard';
import type { CompareChartRunData } from './charts/CompareRunsCharts.common';
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

export interface RunsCompareChartsProps {
  chartRunData: CompareChartRunData[];
  cardsConfig: RunsCompareCardConfig[];
  isMetricHistoryLoading?: boolean;
  onRemoveChart: (chart: RunsCompareCardConfig) => void;
  onStartEditChart: (chart: RunsCompareCardConfig) => void;
}

export const RunsCompareCharts = ({
  chartRunData,
  cardsConfig,
  isMetricHistoryLoading,
  onRemoveChart,
  onStartEditChart,
}: RunsCompareChartsProps) => {
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
  return (
    <>
      {parallelChartCards.length ? (
        <div css={styles.parallelChartsWrapper}>
          {parallelChartCards.map((cardConfig) => (
            <RunsCompareParallelChartCard
              config={cardConfig as RunsCompareParallelCardConfig}
              chartRunData={chartRunData}
              onEdit={() => onStartEditChart(cardConfig)}
              onDelete={() => onRemoveChart(cardConfig)}
            />
          ))}
        </div>
      ) : null}
      <div css={styles.chartsWrapper}>
        {remainingChartCards.map((cardConfig) => {
          if (cardConfig.type === RunsCompareChartType.BAR) {
            return (
              <RunsCompareBarChartCard
                config={cardConfig as RunsCompareBarCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.LINE) {
            return (
              <RunsCompareLineChartCard
                config={cardConfig as RunsCompareLineCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
                isMetricHistoryLoading={isMetricHistoryLoading}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.SCATTER) {
            return (
              <RunsCompareScatterChartCard
                config={cardConfig as RunsCompareScatterCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
              />
            );
          } else if (cardConfig.type === RunsCompareChartType.CONTOUR) {
            return (
              <RunsCompareContourChartCard
                config={cardConfig as RunsCompareContourCardConfig}
                chartRunData={chartRunData}
                onEdit={() => onStartEditChart(cardConfig)}
                onDelete={() => onRemoveChart(cardConfig)}
              />
            );
          }

          return null;
        })}
      </div>
    </>
  );
};

const styles = {
  chartGroupWrapper: (theme: Theme) => ({
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.md,
  }),
  parallelChartsWrapper: (theme: Theme) => ({
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.md,
    marginBottom: theme.spacing.md,
  }),
  chartsWrapper: (theme: Theme) => ({
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(520px, 1fr))',
    gap: theme.spacing.md,
  }),
};
