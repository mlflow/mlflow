import React, { useCallback, useMemo } from 'react';
import { UpdateExperimentViewStateFn } from '../../../../types';
import { useRunSortOptions } from '../../hooks/useRunSortOptions';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import { ExperimentViewRunsControlsFilters } from './ExperimentViewRunsControlsFilters';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { ToggleButton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ExperimentViewRunsSortSelector } from './ExperimentViewRunsSortSelector';
import { TAGS_TO_COLUMNS_MAP } from '../../utils/experimentPage.column-utils';
import { COLUMN_SORT_BY_ASC, SORT_DELIMITER_SYMBOL } from '../../../../constants';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { shouldEnableRunGrouping } from '../../../../../common/utils/FeatureUtils';
import { ExperimentViewRunsModeSwitch } from './ExperimentViewRunsModeSwitch';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import Utils from '../../../../../common/utils/Utils';
import { downloadRunsCsv } from '../../utils/experimentPage.common-utils';
import { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentViewRunsGroupBySelector } from './ExperimentViewRunsGroupBySelector';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

type ExperimentViewRunsControlsProps = {
  viewState: ExperimentPageViewState;
  updateViewState: UpdateExperimentViewStateFn;

  searchFacetsState: ExperimentPageSearchFacetsState;

  experimentId: string;

  runsData: ExperimentRunsSelectorResult;

  expandRows: boolean;
  updateExpandRows: (expandRows: boolean) => void;

  requestError: ErrorWrapper | null;

  refreshRuns: () => void;
  uiState: ExperimentPageUIState;
  isLoading: boolean;
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
  }: ExperimentViewRunsControlsProps) => {
    const [compareRunsMode, setCompareRunsMode] = useExperimentPageViewMode();

    const { paramKeyList, metricKeyList, tagsList } = runsData;
    const { orderByAsc, orderByKey } = searchFacetsState;

    const updateUIState = useUpdateExperimentViewUIState();

    const isComparingRuns = compareRunsMode !== undefined;

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
          marginTop: uiState.viewMaximized ? undefined : theme.spacing.md,
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
            additionalControls={
              <>
                <ExperimentViewRunsSortSelector
                  orderByAsc={orderByAsc}
                  orderByKey={orderByKey}
                  sortOptions={sortOptions}
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
                  <ToggleButton onClick={toggleExpandedRows}>
                    <FormattedMessage
                      defaultMessage="Expand rows"
                      description="Label for the expand rows button above the experiment runs table"
                    />
                  </ToggleButton>
                )}
                {shouldEnableRunGrouping() && (
                  <ExperimentViewRunsGroupBySelector
                    groupBy={uiState.groupBy}
                    onChange={(groupBy) => {
                      updateUIState((state) => ({ ...state, groupBy }));
                    }}
                    runsData={runsData}
                    isLoading={isLoading}
                  />
                )}
              </>
            }
          />
        )}
        <div>
          <ExperimentViewRunsModeSwitch
            // Use modernized view mode value and updater if flag is set
            compareRunsMode={compareRunsMode}
            setCompareRunsMode={setCompareRunsMode}
            viewState={viewState}
            runsAreGrouped={Boolean(uiState.groupBy)}
          />
        </div>
      </div>
    );
  },
);
