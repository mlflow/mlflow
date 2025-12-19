import React, { useCallback, useMemo } from 'react';
import type { UpdateExperimentViewStateFn } from '../../../../types';
import { useRunSortOptions } from '../../hooks/useRunSortOptions';
import type { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import { ExperimentViewRunsControlsFilters } from './ExperimentViewRunsControlsFilters';
import type { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { ToggleButton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import Utils from '../../../../../common/utils/Utils';
import { downloadRunsCsv } from '../../utils/experimentPage.common-utils';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentViewRunsGroupBySelector } from './ExperimentViewRunsGroupBySelector';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { ExperimentViewRunsSortSelectorV2 } from './ExperimentViewRunsSortSelectorV2';

type ExperimentViewRunsControlsProps = {
  viewState: ExperimentPageViewState;
  updateViewState: UpdateExperimentViewStateFn;

  searchFacetsState: ExperimentPageSearchFacetsState;

  experimentId: string;

  runsData: ExperimentRunsSelectorResult;

  expandRows: boolean;
  updateExpandRows: (expandRows: boolean) => void;

  requestError: ErrorWrapper | Error | null;

  refreshRuns: () => void;
  uiState: ExperimentPageUIState;
  isLoading: boolean;
  isComparingExperiments: boolean;
};

/**
 * This component houses all controls related to searching runs: sort controls,
 * filters and run related actions (delete, restore, download CSV).
 */
export const ExperimentViewRunsControls = React.memo(
  ({
    runsData,
    viewState,
    updateViewState,
    searchFacetsState,
    experimentId,
    requestError,
    expandRows,
    updateExpandRows,
    refreshRuns,
    uiState,
    isLoading,
    isComparingExperiments,
  }: ExperimentViewRunsControlsProps) => {
    const [compareRunsMode, setCompareRunsMode] = useExperimentPageViewMode();

    const { paramKeyList, metricKeyList, tagsList } = runsData;
    const { orderByAsc, orderByKey } = searchFacetsState;

    const updateUIState = useUpdateExperimentViewUIState();

    const isComparingRuns = compareRunsMode !== 'TABLE';
    const isEvaluationMode = compareRunsMode === 'ARTIFACT';

    const { theme } = useDesignSystemTheme();

    const filteredParamKeys = paramKeyList;
    const filteredMetricKeys = metricKeyList;
    const filteredTagKeys = Utils.getVisibleTagKeyList(tagsList);

    const onDownloadCsv = useCallback(
      () => downloadRunsCsv(runsData, filteredTagKeys, filteredParamKeys, filteredMetricKeys),
      [filteredMetricKeys, filteredParamKeys, filteredTagKeys, runsData],
    );

    const sortOptions = useRunSortOptions(filteredMetricKeys, filteredParamKeys);

    const selectedRunsCount = Object.values(viewState.runsSelected).filter(Boolean).length;
    const canRestoreRuns = selectedRunsCount > 0;
    const canRenameRuns = selectedRunsCount === 1;
    const canCompareRuns = selectedRunsCount > 1;
    const showActionButtons = canCompareRuns || canRenameRuns || canRestoreRuns;

    const showGroupBySelector = !isEvaluationMode;

    // Shows or hides the column selector
    const changeColumnSelectorVisible = useCallback(
      (value: boolean) => updateViewState({ columnSelectorVisible: value }),
      [updateViewState],
    );

    const toggleExpandedRows = useCallback(() => updateExpandRows(!expandRows), [expandRows, updateExpandRows]);

    const multipleDatasetsArePresent = useMemo(
      () => runsData.datasetsList.some((datasetsInRun) => datasetsInRun?.length > 1),
      [runsData],
    );

    return (
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          flexDirection: 'column' as const,
          marginBottom: theme.spacing.sm,
        }}
      >
        {showActionButtons && (
          <ExperimentViewRunsControlsActions
            runsData={runsData}
            searchFacetsState={searchFacetsState}
            viewState={viewState}
            refreshRuns={refreshRuns}
          />
        )}

        {!showActionButtons && (
          <ExperimentViewRunsControlsFilters
            onDownloadCsv={onDownloadCsv}
            searchFacetsState={searchFacetsState}
            experimentId={experimentId}
            viewState={viewState}
            updateViewState={updateViewState}
            runsData={runsData}
            requestError={requestError}
            refreshRuns={refreshRuns}
            viewMaximized={uiState.viewMaximized}
            autoRefreshEnabled={uiState.autoRefreshEnabled}
            hideEmptyCharts={uiState.hideEmptyCharts}
            areRunsGrouped={Boolean(uiState.groupBy)}
            additionalControls={
              <>
                <ExperimentViewRunsSortSelectorV2
                  orderByAsc={orderByAsc}
                  orderByKey={orderByKey}
                  metricKeys={filteredMetricKeys}
                  paramKeys={filteredParamKeys}
                />

                {!isComparingRuns && (
                  <ExperimentViewRunsColumnSelector
                    columnSelectorVisible={viewState.columnSelectorVisible}
                    onChangeColumnSelectorVisible={changeColumnSelectorVisible}
                    runsData={runsData}
                    selectedColumns={uiState.selectedColumns}
                  />
                )}

                {!isComparingRuns && multipleDatasetsArePresent && (
                  <ToggleButton
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrols.tsx_175"
                    onClick={toggleExpandedRows}
                  >
                    <FormattedMessage
                      defaultMessage="Expand rows"
                      description="Label for the expand rows button above the experiment runs table"
                    />
                  </ToggleButton>
                )}
                {showGroupBySelector && (
                  <ExperimentViewRunsGroupBySelector
                    groupBy={uiState.groupBy}
                    onChange={(groupBy) => {
                      updateUIState((state) => ({ ...state, groupBy }));
                    }}
                    runsData={runsData}
                    isLoading={isLoading}
                    useGroupedValuesInCharts={uiState.useGroupedValuesInCharts ?? true}
                    onUseGroupedValuesInChartsChange={(useGroupedValuesInCharts) => {
                      updateUIState((state) => ({ ...state, useGroupedValuesInCharts }));
                    }}
                  />
                )}
              </>
            }
          />
        )}
      </div>
    );
  },
);
