import { Button, Checkbox, Option, Select, SidebarIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate } from 'react-router-dom-v5-compat';
import { COLUMN_SORT_BY_ASC, LIFECYCLE_FILTER, SORT_DELIMITER_SYMBOL } from '../../../../constants';
import Routes from '../../../../routes';
import { UpdateExperimentSearchFacetsFn, UpdateExperimentViewStateFn } from '../../../../types';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunModals } from './ExperimentViewRunModals';
import { ExperimentViewRunsSortSelector } from './ExperimentViewRunsSortSelector';
import { ExperimentViewRunsColumnSelector } from './ExperimentViewRunsColumnSelector';
import { TAGS_TO_COLUMNS_MAP } from '../../utils/experimentPage.column-utils';
import type { ExperimentRunSortOption } from '../../hooks/useRunSortOptions';
import {
  shouldEnableArtifactBasedEvaluation,
  shouldEnableExperimentDatasetTracking,
} from '../../../../../common/utils/FeatureUtils';
import { ToggleIconButton } from '../../../../../common/components/ToggleIconButton';

export type ExperimentViewRunsControlsActionsProps = {
  viewState: SearchExperimentRunsViewState;
  sortOptions: ExperimentRunSortOption[];
  updateViewState: UpdateExperimentViewStateFn;
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  runsData: ExperimentRunsSelectorResult;
  expandRows: boolean;
  updateExpandRows: (expandRows: boolean) => void;
};

const CompareRunsButtonWrapper: React.FC = ({ children }) => <>{children}</>;

export const ExperimentViewRunsControlsActions = React.memo(
  ({
    viewState,
    runsData,
    sortOptions,
    searchFacetsState,
    updateSearchFacets,
    updateViewState,
    expandRows,
    updateExpandRows,
  }: ExperimentViewRunsControlsActionsProps) => {
    const { runsSelected } = viewState;
    const { runInfos } = runsData;
    const { lifecycleFilter, compareRunsMode } = searchFacetsState;

    const isComparingRuns = compareRunsMode !== undefined;

    const navigate = useNavigate();

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

    const [showDeleteRunModal, setShowDeleteRunModal] = useState(false);
    const [showRestoreRunModal, setShowRestoreRunModal] = useState(false);
    const [showRenameRunModal, setShowRenameRunModal] = useState(false);
    const [renamedRunName, setRenamedRunName] = useState('');

    const renameButtonClicked = useCallback(() => {
      const runsSelectedList = Object.keys(runsSelected);
      const selectedRun = runInfos.find((info) => info.run_uuid === runsSelectedList[0]);
      if (selectedRun) {
        setRenamedRunName(selectedRun.run_name);
        setShowRenameRunModal(true);
      }
    }, [runInfos, runsSelected]);

    const compareButtonClicked = useCallback(() => {
      const runsSelectedList = Object.keys(runsSelected);
      const experimentIds = runInfos
        .filter(({ run_uuid }: any) => runsSelectedList.includes(run_uuid))
        .map(({ experiment_id }: any) => experiment_id);

      navigate(Routes.getCompareRunPageRoute(runsSelectedList, [...new Set(experimentIds)].sort()));
    }, [navigate, runInfos, runsSelected]);

    const onDeleteRun = useCallback(() => setShowDeleteRunModal(true), []);
    const onRestoreRun = useCallback(() => setShowRestoreRunModal(true), []);
    const onCloseDeleteRunModal = useCallback(() => setShowDeleteRunModal(false), []);
    const onCloseRestoreRunModal = useCallback(() => setShowRestoreRunModal(false), []);
    const onCloseRenameRunModal = useCallback(() => setShowRenameRunModal(false), []);

    const selectedRunsCount = Object.values(viewState.runsSelected).filter(Boolean).length;
    const canRestoreRuns = selectedRunsCount > 0;
    const canRenameRuns = selectedRunsCount === 1;
    const canCompareRuns = selectedRunsCount > 1;
    const showActionButtons = canCompareRuns || canRenameRuns || canRestoreRuns;

    const toggleExpandedRows = useCallback(
      () => updateExpandRows(!expandRows),
      [expandRows, updateExpandRows],
    );

    // Show preview sidebar only on table view and artifact view
    const displaySidebarToggleButton =
      compareRunsMode === undefined || compareRunsMode === 'ARTIFACT';

    return (
      <div css={styles.controlBar}>
        <ExperimentViewRunModals
          runsSelected={runsSelected}
          onCloseRenameRunModal={onCloseRenameRunModal}
          onCloseDeleteRunModal={onCloseDeleteRunModal}
          onCloseRestoreRunModal={onCloseRestoreRunModal}
          showDeleteRunModal={showDeleteRunModal}
          showRestoreRunModal={showRestoreRunModal}
          showRenameRunModal={showRenameRunModal}
          renamedRunName={renamedRunName}
        />

        {!showActionButtons && (
          <>
            <ExperimentViewRunsSortSelector
              onSortKeyChanged={sortKeyChanged}
              searchFacetsState={searchFacetsState}
              sortOptions={sortOptions}
            />
            {!isComparingRuns && (
              <ExperimentViewRunsColumnSelector
                columnSelectorVisible={viewState.columnSelectorVisible}
                onChangeColumnSelectorVisible={changeColumnSelectorVisible}
                runsData={runsData}
              />
            )}

            {!isComparingRuns && shouldEnableExperimentDatasetTracking() && (
              <Button onClick={() => toggleExpandedRows()}>
                <div css={{ display: 'flex' }}>
                  <Checkbox isChecked={expandRows} />
                  <FormattedMessage
                    defaultMessage='Expand rows'
                    description='Label for the expand rows button above the experiment runs table'
                  />
                </div>
              </Button>
            )}
          </>
        )}

        {showActionButtons && (
          <>
            <Button
              data-testid='run-rename-button'
              onClick={renameButtonClicked}
              disabled={!canRenameRuns}
            >
              <FormattedMessage
                defaultMessage='Rename'
                description='Label for the rename run button above the experiment runs table'
              />
            </Button>
            {lifecycleFilter === LIFECYCLE_FILTER.ACTIVE ? (
              <Button
                data-testid='runs-delete-button'
                disabled={!canRestoreRuns}
                onClick={onDeleteRun}
                danger
              >
                <FormattedMessage
                  defaultMessage='Delete'
                  // eslint-disable-next-line max-len
                  description='String for the delete button to delete a particular experiment run'
                />
              </Button>
            ) : null}
            {lifecycleFilter === LIFECYCLE_FILTER.DELETED ? (
              <Button
                data-testid='runs-restore-button'
                disabled={!canRestoreRuns}
                onClick={onRestoreRun}
              >
                <FormattedMessage
                  defaultMessage='Restore'
                  // eslint-disable-next-line max-len
                  description='String for the restore button to undo the experiments that were deleted'
                />
              </Button>
            ) : null}
            <div css={styles.buttonSeparator} />
            <CompareRunsButtonWrapper>
              <Button
                data-testid='runs-compare-button'
                disabled={!canCompareRuns}
                onClick={compareButtonClicked}
              >
                <FormattedMessage
                  defaultMessage='Compare'
                  // eslint-disable-next-line max-len
                  description='String for the compare button to compare experiment runs to find an ideal model'
                />
              </Button>
            </CompareRunsButtonWrapper>
          </>
        )}
        <div css={{ flex: '1' }} />
        {shouldEnableArtifactBasedEvaluation() && displaySidebarToggleButton && (
          <ToggleIconButton
            pressed={viewState.previewPaneVisible}
            icon={<SidebarIcon />}
            onClick={() => updateViewState({ previewPaneVisible: !viewState.previewPaneVisible })}
          />
        )}
      </div>
    );
  },
);

const styles = {
  groupSeparator: () => ({ flex: 1 }),
  buttonSeparator: (theme: Theme) => ({
    borderLeft: `1px solid ${theme.colors.border}`,
    marginLeft: theme.spacing.xs,
    marginRight: theme.spacing.xs,
    height: '100%',
  }),
  controlBar: (theme: Theme) => ({
    display: 'flex',
    gap: theme.spacing.sm,
    alignItems: 'center',
    paddingTop: theme.spacing.sm,
    borderTop: `1px solid ${theme.colors.border}`,
  }),
};
