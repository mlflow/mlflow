import React, { useCallback, useMemo } from 'react';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { useRunSortOptions } from '../../hooks/useRunSortOptions';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
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
import {
  shouldEnableRunGrouping,
  shouldEnableShareExperimentViewByTags,
} from '../../../../../common/utils/FeatureUtils';
import { ExperimentViewRunsModeSwitch } from './ExperimentViewRunsModeSwitch';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import Utils from '../../../../../common/utils/Utils';
import { downloadRunsCsv } from '../../utils/experimentPage.common-utils';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';
import { ExperimentViewRunsGroupBySelector } from './ExperimentViewRunsGroupBySelector';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';

type ExperimentViewRunsControlsProps = {
  viewState: SearchExperimentRunsViewState;
  updateViewState: UpdateExperimentViewStateFn;

  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;

  experimentId: string;

  runsData: ExperimentRunsSelectorResult;

  expandRows: boolean;
  updateExpandRows: (expandRows: boolean) => void;

  requestError: ErrorWrapper | null;

  refreshRuns: () => void;
  uiState: ExperimentPageUIStateV2;
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
    updateSearchFacets,
    searchFacetsState,
    experimentId,
    requestError,
    expandRows,
    updateExpandRows,
    refreshRuns,
    uiState,
    isLoading,
  }: ExperimentViewRunsControlsProps) => {
    const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();

    const [pageViewMode, setPageViewMode] = useExperimentPageViewMode();

    const { paramKeyList, metricKeyList, tagsList } = runsData;
    const { orderByAsc, orderByKey } = searchFacetsState;

    // Use modernized view mode value getter if flag is set
    const compareRunsMode = usingNewViewStateModel ? pageViewMode : searchFacetsState.compareRunsMode;

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

    // Callback fired when the sort column value changes
    const sortKeyChanged = useCallback(
      ({ value: compiledOrderByKey }) => {
        const [newOrderBy, newOrderAscending] = compiledOrderByKey.split(SORT_DELIMITER_SYMBOL);

        const columnToAdd = TAGS_TO_COLUMNS_MAP[newOrderBy] || newOrderBy;
        const isOrderAscending = newOrderAscending === COLUMN_SORT_BY_ASC;

        updateSearchFacets((currentFacets) => {
          const { selectedColumns } = currentFacets;
          if (!selectedColumns.includes(columnToAdd)) {
            selectedColumns.push(columnToAdd);
          }
          return {
            ...currentFacets,
            selectedColumns,
            orderByKey: newOrderBy,
            orderByAsc: isOrderAscending,
          };
        });
      },
      [updateSearchFacets],
    );

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
          marginTop: viewState.viewMaximized ? undefined : theme.spacing.md,
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
            updateSearchFacets={updateSearchFacets}
            searchFacetsState={searchFacetsState}
            experimentId={experimentId}
            viewState={viewState}
            updateViewState={updateViewState}
            runsData={runsData}
            requestError={requestError}
            refreshRuns={refreshRuns}
            viewMaximized={usingNewViewStateModel ? uiState.viewMaximized : viewState.viewMaximized}
            additionalControls={
              <>
                <ExperimentViewRunsSortSelector
                  onSortKeyChanged={sortKeyChanged}
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
            compareRunsMode={usingNewViewStateModel ? pageViewMode : compareRunsMode}
            setCompareRunsMode={(newCompareRunsMode) => {
              if (usingNewViewStateModel) {
                setPageViewMode(newCompareRunsMode);
              } else {
                updateSearchFacets({ compareRunsMode: newCompareRunsMode });
              }
            }}
            viewState={viewState}
            runsAreGrouped={Boolean(uiState.groupBy)}
          />
        </div>
      </div>
    );
  },
);
