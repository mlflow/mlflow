import { Empty, Input, SearchIcon, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { noop, uniq } from 'lodash';
import type { ReactNode } from 'react';
import { memo, useMemo, useCallback, useState } from 'react';
import type { LoggedModelProto } from '../../types';
import type { ExperimentRunsChartsUIConfiguration } from '../experiment-page/models/ExperimentPageUIState';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { RunsChartsDraggableCardsGridContextProvider } from '../runs-charts/components/RunsChartsDraggableCardsGridContext';
import { RunsChartsFullScreenModal } from '../runs-charts/components/RunsChartsFullScreenModal';
import { RunsChartsTooltipBody } from '../runs-charts/components/RunsChartsTooltipBody';
import { RunsChartsSectionAccordion } from '../runs-charts/components/sections/RunsChartsSectionAccordion';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import {
  RunsChartsUIConfigurationContextProvider,
  useConfirmChartCardConfigurationFn,
  useRemoveRunsChartFn,
  useUpdateRunsChartsUIConfiguration,
} from '../runs-charts/hooks/useRunsChartsUIConfiguration';
import type { RunsChartsMetricByDatasetEntry } from '../runs-charts/runs-charts.types';
import { RunsChartsCardConfig, RunsChartType } from '../runs-charts/runs-charts.types';
import { useExperimentLoggedModelsChartsData } from './hooks/useExperimentLoggedModelsChartsData';
import { useExperimentLoggedModelsChartsUIState } from './hooks/useExperimentLoggedModelsChartsUIState';
import { useExperimentLoggedModelAllMetricsByDataset } from './hooks/useExperimentLoggedModelAllMetricsByDataset';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMemoDeep } from '../../../common/hooks/useMemoDeep';
import { RunsChartsConfigureModal } from '../runs-charts/components/RunsChartsConfigureModal';
import Routes from '../../routes';

const ExperimentLoggedModelListChartsImpl = memo(
  ({
    chartData,
    uiState,
    metricKeysByDataset,
    minWidth,
  }: {
    chartData: RunsChartsRunData[];
    uiState: ExperimentRunsChartsUIConfiguration;
    metricKeysByDataset: RunsChartsMetricByDatasetEntry[];
    minWidth: number;
  }) => {
    const { theme } = useDesignSystemTheme();
    const { formatMessage } = useIntl();

    const availableMetricKeys = useMemo(() => uniq(chartData.flatMap((run) => Object.keys(run.metrics))), [chartData]);
    const availableParamKeys = useMemo(() => uniq(chartData.flatMap((run) => Object.keys(run.params))), [chartData]);

    const updateChartsUIState = useUpdateRunsChartsUIConfiguration();

    const setSearch = useCallback(
      (search: string) => {
        updateChartsUIState((state) => ({ ...state, chartsSearchFilter: search }));
      },
      [updateChartsUIState],
    );

    const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsChartsCardConfig | null>(null);

    const addNewChartCard = useCallback(
      (metricSectionId: string) => (type: RunsChartType) =>
        setConfiguredCardConfig(RunsChartsCardConfig.getEmptyChartCardByType(type, false, undefined, metricSectionId)),
      [],
    );

    const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();
    const removeChart = useRemoveRunsChartFn();

    const [fullScreenChart, setFullScreenChart] = useState<
      | {
          config: RunsChartsCardConfig;
          title: string | ReactNode;
          subtitle: ReactNode;
        }
      | undefined
    >(undefined);

    const fullscreenTooltipContextValue = useMemo(() => ({ runs: chartData }), [chartData]);

    const tooltipContextValue = useMemo(
      () => ({ runs: chartData, getDataTraceLink: Routes.getExperimentLoggedModelDetailsPageRoute }),
      [chartData],
    );

    const emptyState = (
      <div css={{ marginTop: theme.spacing.lg }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No models found in experiment or all models are hidden. Select at least one model to view charts."
              description="Label displayed in logged models chart view when no models are visible or selected"
            />
          }
        />
      </div>
    );

    return (
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          paddingLeft: theme.spacing.md,
          paddingRight: theme.spacing.md,
          paddingBottom: theme.spacing.md,

          borderTop: `1px solid ${theme.colors.border}`,
          borderLeft: `1px solid ${theme.colors.border}`,

          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          minWidth,
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            paddingTop: theme.spacing.sm,
            overflow: 'hidden',
            flex: 1,
          }}
        >
          <Input
            componentId="mlflow.logged_model.list.charts.search"
            role="searchbox"
            prefix={<SearchIcon />}
            value={uiState.chartsSearchFilter ?? ''}
            allowClear
            onChange={({ target }) => setSearch(target.value)}
            placeholder={formatMessage({
              defaultMessage: 'Search metric charts',
              description: 'Placeholder for chart search input on the logged model chart view',
            })}
          />
          <div css={{ overflow: 'auto' }}>
            <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={RunsChartsTooltipBody}>
              <RunsChartsDraggableCardsGridContextProvider visibleChartCards={uiState.compareRunCharts}>
                <RunsChartsSectionAccordion
                  compareRunSections={uiState.compareRunSections}
                  compareRunCharts={uiState.compareRunCharts}
                  reorderCharts={noop}
                  insertCharts={noop}
                  chartData={chartData}
                  startEditChart={setConfiguredCardConfig}
                  removeChart={removeChart}
                  addNewChartCard={addNewChartCard}
                  search={uiState.chartsSearchFilter ?? ''}
                  groupBy={null}
                  setFullScreenChart={setFullScreenChart}
                  autoRefreshEnabled={false}
                  hideEmptyCharts={false}
                  globalLineChartConfig={undefined}
                  supportedChartTypes={[RunsChartType.BAR, RunsChartType.SCATTER]}
                  noRunsSelectedEmptyState={emptyState}
                />
              </RunsChartsDraggableCardsGridContextProvider>
            </RunsChartsTooltipWrapper>
            <RunsChartsFullScreenModal
              fullScreenChart={fullScreenChart}
              onCancel={() => setFullScreenChart(undefined)}
              chartData={chartData}
              groupBy={null}
              tooltipContextValue={fullscreenTooltipContextValue}
              tooltipComponent={RunsChartsTooltipBody}
              autoRefreshEnabled={false}
              globalLineChartConfig={undefined}
            />
            {configuredCardConfig && (
              <RunsChartsConfigureModal
                chartRunData={chartData}
                metricKeyList={availableMetricKeys}
                metricKeysByDataset={metricKeysByDataset}
                paramKeyList={availableParamKeys}
                config={configuredCardConfig}
                onSubmit={(configuredCardConfig) => {
                  confirmChartCardConfiguration({ ...configuredCardConfig, displayName: undefined });
                  setConfiguredCardConfig(null);
                }}
                onCancel={() => setConfiguredCardConfig(null)}
                groupBy={null}
                supportedChartTypes={[RunsChartType.BAR, RunsChartType.SCATTER]}
              />
            )}
          </div>
        </div>
      </div>
    );
  },
);

export const ExperimentLoggedModelListCharts = memo(
  ({
    loggedModels,
    experimentId,
    minWidth,
  }: {
    loggedModels: LoggedModelProto[];
    experimentId: string;
    minWidth: number;
  }) => {
    const { theme } = useDesignSystemTheme();

    // Perform deep comparison on the logged models to avoid re-rendering the charts when the logged models change.
    // Deep comparison should still be cheaper than rerendering all charts.
    const cachedLoggedModels = useMemoDeep(() => loggedModels, [loggedModels]);

    const metricsByDataset = useExperimentLoggedModelAllMetricsByDataset(cachedLoggedModels);

    const {
      chartUIState,
      updateUIState,
      loading: loadingState,
    } = useExperimentLoggedModelsChartsUIState(metricsByDataset, experimentId);
    const chartData = useExperimentLoggedModelsChartsData(cachedLoggedModels);

    if (loadingState) {
      return (
        <div
          css={{
            backgroundColor: theme.colors.backgroundPrimary,
            paddingTop: theme.spacing.lg,
            borderTop: `1px solid ${theme.colors.border}`,
            borderLeft: `1px solid ${theme.colors.border}`,
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
            display: 'flex',
          }}
        >
          <Spinner />
        </div>
      );
    }
    return (
      <RunsChartsUIConfigurationContextProvider updateChartsUIState={updateUIState}>
        <ExperimentLoggedModelListChartsImpl
          chartData={chartData}
          uiState={chartUIState}
          metricKeysByDataset={metricsByDataset}
          minWidth={minWidth}
        />
      </RunsChartsUIConfigurationContextProvider>
    );
  },
);
