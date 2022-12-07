import { Button } from '@databricks/design-system';
import React, { useCallback, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useHistory } from 'react-router-dom';
import { LIFECYCLE_FILTER } from '../../../../constants';
import Routes from '../../../../routes';
import { UpdateExperimentSearchFacetsFn } from '../../../../types';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunModals } from './ExperimentViewRunModals';

export type ExperimentViewRunsControlsActionsProps = {
  viewState: SearchExperimentRunsViewState;

  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  runsData: ExperimentRunsSelectorResult;
};

export const ExperimentViewRunsControlsActions = React.memo(
  ({ viewState, runsData, searchFacetsState }: ExperimentViewRunsControlsActionsProps) => {
    const { runsSelected } = viewState;
    const { runInfos } = runsData;
    const { lifecycleFilter } = searchFacetsState;

    const history = useHistory();

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
        <div css={styles.gapElement} />
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
      </div>
    );
  },
);

const styles = {
  controlBar: { display: 'flex', gap: 8, alignItems: 'center' },
  gapElement: { flex: 1 },
};
