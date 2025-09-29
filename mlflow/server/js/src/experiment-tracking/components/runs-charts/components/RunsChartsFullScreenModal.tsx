import type { ReactNode } from 'react';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import type { RunsChartsRunData } from './RunsCharts.common';
import { Modal, useDesignSystemTheme } from '@databricks/design-system';
import type { RunsChartsTooltipBodyProps } from '../hooks/useRunsChartsTooltip';
import { RunsChartsTooltipWrapper } from '../hooks/useRunsChartsTooltip';
import { RunsChartsCard } from './cards/RunsChartsCard';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsGlobalLineChartConfig } from '../../experiment-page/models/ExperimentPageUIState';

export const RunsChartsFullScreenModal = <TContext,>({
  chartData,
  isMetricHistoryLoading = false,
  groupBy,
  fullScreenChart,
  onCancel,
  tooltipContextValue,
  tooltipComponent,
  autoRefreshEnabled,
  globalLineChartConfig,
}: {
  chartData: RunsChartsRunData[];
  isMetricHistoryLoading?: boolean;
  groupBy: RunsGroupByConfig | null;
  autoRefreshEnabled?: boolean;
  fullScreenChart:
    | {
        config: RunsChartsCardConfig;
        title: string | ReactNode;
        subtitle: ReactNode;
      }
    | undefined;
  onCancel: () => void;
  tooltipContextValue: TContext;
  tooltipComponent: React.ComponentType<React.PropsWithChildren<RunsChartsTooltipBodyProps<TContext>>>;
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
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
      componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsfullscreenmodal.tsx_53"
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
          canMoveToTop={false}
          canMoveToBottom={false}
          firstChartUuid={undefined}
          lastChartUuid={undefined}
          cardConfig={fullScreenChart.config}
          chartRunData={chartData}
          groupBy={groupBy}
          index={0}
          sectionIndex={0}
          fullScreen
          autoRefreshEnabled={autoRefreshEnabled}
          globalLineChartConfig={globalLineChartConfig}
          {...emptyConfigureProps}
          {...emptyReorderProps}
        />
      </RunsChartsTooltipWrapper>
    </Modal>
  );
};
