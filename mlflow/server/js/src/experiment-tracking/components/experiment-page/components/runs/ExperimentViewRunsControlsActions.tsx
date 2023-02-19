import { Button, Option, Select, Tooltip } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useHistory } from 'react-router-dom';
import { LIFECYCLE_FILTER } from '../../../../constants';
import Routes from '../../../../routes';
import { UpdateExperimentSearchFacetsFn } from '../../../../types';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { getStartTimeColumnDisplayName } from '../../utils/experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunModals } from './ExperimentViewRunModals';

export type ExperimentViewRunsControlsActionsProps = {
  viewState: SearchExperimentRunsViewState;

  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  runsData: ExperimentRunsSelectorResult;
};

const CompareRunsButtonWrapper: React.FC = ({ children }) => <>{children}</>;

export const ExperimentViewRunsControlsActions = React.memo(
  ({
    viewState,
    runsData,
    searchFacetsState,
    updateSearchFacets,
  }: ExperimentViewRunsControlsActionsProps) => {
    const { runsSelected } = viewState;
    const { runInfos } = runsData;
    const { lifecycleFilter, startTime } = searchFacetsState;

    const history = useHistory();
    const intl = useIntl();

    // List of labels for "start time" filter
    const startTimeColumnLabels = useMemo(() => getStartTimeColumnDisplayName(intl), [intl]);

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
      history.push(
        Routes.getCompareRunPageRoute(runsSelectedList, [...new Set(experimentIds)].sort()),
      );
    }, [history, runInfos, runsSelected]);

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

    const currentLifecycleFilterLabel = (
      <>
        <FormattedMessage
          defaultMessage='State:'
          description='Filtering label to filter experiments based on state of active or deleted'
        />{' '}
        {lifecycleFilter}
      </>
    );

    const currentStartTimeFilterLabel = (
      <>
        <FormattedMessage
          defaultMessage='Time created'
          description='Label for the start time select dropdown for experiment runs view'
        />
        : {startTimeColumnLabels[startTime as keyof typeof startTimeColumnLabels]}
      </>
    );

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
            <Select
              className='start-time-select'
              value={{ value: startTime, label: currentStartTimeFilterLabel }}
              labelInValue
              onChange={({ value: newStartTime }) => {
                updateSearchFacets({ startTime: newStartTime });
              }}
              data-test-id='start-time-select-dropdown'
              // Temporarily we're disabling virtualized list to maintain
              // backwards compatiblity. Functional unit tests rely heavily
              // on non-virtualized values.
              dangerouslySetAntdProps={{ virtual: false } as any}
            >
              {Object.keys(startTimeColumnLabels).map((startTimeKey) => (
                <Option
                  key={startTimeKey}
                  title={startTimeColumnLabels[startTimeKey as keyof typeof startTimeColumnLabels]}
                  data-test-id={`start-time-select-${startTimeKey}`}
                  value={startTimeKey}
                >
                  {startTimeColumnLabels[startTimeKey as keyof typeof startTimeColumnLabels]}
                </Option>
              ))}
            </Select>

            <Select
              value={{ value: lifecycleFilter, label: currentLifecycleFilterLabel }}
              labelInValue
              data-testid='lifecycle-filter'
              onChange={({ value }) => updateSearchFacets({ lifecycleFilter: value })}
            >
              <Select.Option data-testid='active-runs-menu-item' value={LIFECYCLE_FILTER.ACTIVE}>
                <FormattedMessage
                  defaultMessage='Active'
                  description='Linked model dropdown option to show active experiment runs'
                />
              </Select.Option>
              <Select.Option data-testid='deleted-runs-menu-item' value={LIFECYCLE_FILTER.DELETED}>
                <FormattedMessage
                  defaultMessage='Deleted'
                  description='Linked model dropdown option to show deleted experiment runs'
                />
              </Select.Option>
            </Select>
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
