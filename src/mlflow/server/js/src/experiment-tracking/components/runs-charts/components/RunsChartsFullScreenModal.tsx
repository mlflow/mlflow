import { ReactNode } from 'react';
import { RunsChartsCardConfig } from '../runs-charts.types';
import { RunsChartsRunData } from './RunsCharts.common';
import { Modal, useDesignSystemTheme } from '@databricks/design-system';
import { RunsChartsTooltipBodyProps, RunsChartsTooltipWrapper } from '../hooks/useRunsChartsTooltip';
import { RunsChartsCard } from './cards/RunsChartsCard';

export const RunsChartsFullScreenModal = <TContext,>({
  chartData,
  isMetricHistoryLoading = false,
  groupBy = '',
  fullScreenChart,
  onCancel,
  tooltipContextValue,
  tooltipComponent,
}: {
  chartData: RunsChartsRunData[];
  isMetricHistoryLoading?: boolean;
  groupBy?: string;
  fullScreenChart: { config: RunsChartsCardConfig; title: string; subtitle: ReactNode } | undefined;
  onCancel: () => void;
  tooltipContextValue: TContext;
  tooltipComponent: React.ComponentType<RunsChartsTooltipBodyProps<TContext>>;
}) => {
  const { theme, getPrefixedClassName } = useDesignSystemTheme();

  const emptyReorderProps = {
    canMoveDown: false,
    canMoveUp: false,
    onMoveDown: () => {},
    onMoveUp: () => {},
    onReorderWith: () => {},
  };

  const emptyConfigureProps = {
    onRemoveChart: () => {},
    onReorderCharts: () => {},
    onStartEditChart: () => {},
    setFullScreenChart: () => {},
  };

  if (!fullScreenChart) {
    return null;
  }

  return (
    <Modal
      visible
      onCancel={onCancel}
      title={
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          {fullScreenChart.title}
          <span
            css={{
              color: theme.colors.textSecondary,
              fontSize: theme.typography.fontSizeSm,
              marginRight: theme.spacing.xs,
            }}
          >
            {fullScreenChart.subtitle}
          </span>
        </div>
      }
      footer={null}
      verticalSizing="maxed_out"
      dangerouslySetAntdProps={{ width: '95%' }}
      css={{
        [`.${getPrefixedClassName('modal-body')}`]: {
          flex: 1,
        },
      }}
    >
      <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={tooltipComponent}>
        <RunsChartsCard
          cardConfig={fullScreenChart.config}
          chartRunData={chartData}
          isMetricHistoryLoading={isMetricHistoryLoading}
          groupBy={groupBy}
          index={0}
          sectionIndex={0}
          fullScreen
          {...emptyConfigureProps}
          {...emptyReorderProps}
        />
      </RunsChartsTooltipWrapper>
    </Modal>
  );
};
