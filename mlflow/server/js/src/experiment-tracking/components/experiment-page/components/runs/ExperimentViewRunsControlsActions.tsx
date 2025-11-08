import { Button } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../../../common/utils/RoutingUtils';
import { LegacyTooltip } from '@databricks/design-system';
import { LIFECYCLE_FILTER } from '../../../../constants';
import Routes from '../../../../routes';
import type { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunModals } from './ExperimentViewRunModals';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { RunInfoEntity } from '../../../../types';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentViewRunsControlsActionsSelectTags } from './ExperimentViewRunsControlsActionsSelectTags';

export type ExperimentViewRunsControlsActionsProps = {
  viewState: ExperimentPageViewState;
  searchFacetsState: ExperimentPageSearchFacetsState;
  runsData: ExperimentRunsSelectorResult;
  refreshRuns: () => void;
};

const CompareRunsButtonWrapper: React.FC<React.PropsWithChildren<unknown>> = ({ children }) => <>{children}</>;

export const ExperimentViewRunsControlsActions = React.memo(
  ({ viewState, runsData, searchFacetsState, refreshRuns }: ExperimentViewRunsControlsActionsProps) => {
    const { runsSelected } = viewState;
    const { runInfos, tagsList } = runsData;
    const { lifecycleFilter } = searchFacetsState;

    const navigate = useNavigate();
    const { theme } = useDesignSystemTheme();

    const [showDeleteRunModal, setShowDeleteRunModal] = useState(false);
    const [showRestoreRunModal, setShowRestoreRunModal] = useState(false);
    const [showRenameRunModal, setShowRenameRunModal] = useState(false);
    const [renamedRunName, setRenamedRunName] = useState('');

    const renameButtonClicked = useCallback(() => {
      const runsSelectedList = Object.keys(runsSelected);
      const selectedRun = runInfos.find((info) => info.runUuid === runsSelectedList[0]);
      if (selectedRun) {
        setRenamedRunName(selectedRun.runName);
        setShowRenameRunModal(true);
      }
    }, [runInfos, runsSelected]);

    const compareButtonClicked = useCallback(() => {
      const runsSelectedList = Object.keys(runsSelected);
      const experimentIds = runInfos
        .filter(({ runUuid }: RunInfoEntity) => runsSelectedList.includes(runUuid))
        .map(({ experimentId }: any) => experimentId);

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

    return (
      <>
        <div css={styles.controlBar}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_110"
            data-testid="run-rename-button"
            onClick={renameButtonClicked}
            disabled={!canRenameRuns}
          >
            <FormattedMessage
              defaultMessage="Rename"
              description="Label for the rename run button above the experiment runs table"
            />
          </Button>
          {lifecycleFilter === LIFECYCLE_FILTER.ACTIVE ? (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_117"
              data-testid="runs-delete-button"
              disabled={!canRestoreRuns}
              onClick={onDeleteRun}
              danger
            >
              <FormattedMessage
                defaultMessage="Delete"
                // eslint-disable-next-line max-len
                description="String for the delete button to delete a particular experiment run"
              />
            </Button>
          ) : null}
          {lifecycleFilter === LIFECYCLE_FILTER.DELETED ? (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_126"
              data-testid="runs-restore-button"
              disabled={!canRestoreRuns}
              onClick={onRestoreRun}
            >
              <FormattedMessage
                defaultMessage="Restore"
                // eslint-disable-next-line max-len
                description="String for the restore button to undo the experiments that were deleted"
              />
            </Button>
          ) : null}
          <div css={styles.buttonSeparator} />
          <CompareRunsButtonWrapper>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactions.tsx_136"
              data-testid="runs-compare-button"
              disabled={!canCompareRuns}
              onClick={compareButtonClicked}
            >
              <FormattedMessage
                defaultMessage="Compare"
                // eslint-disable-next-line max-len
                description="String for the compare button to compare experiment runs to find an ideal model"
              />
            </Button>
          </CompareRunsButtonWrapper>

          <div css={styles.buttonSeparator} />
          <ExperimentViewRunsControlsActionsSelectTags
            runsSelected={runsSelected}
            runInfos={runInfos}
            tagsList={tagsList}
            refreshRuns={refreshRuns}
          />
        </div>
        <ExperimentViewRunModals
          runsSelected={runsSelected}
          onCloseRenameRunModal={onCloseRenameRunModal}
          onCloseDeleteRunModal={onCloseDeleteRunModal}
          onCloseRestoreRunModal={onCloseRestoreRunModal}
          showDeleteRunModal={showDeleteRunModal}
          showRestoreRunModal={showRestoreRunModal}
          showRenameRunModal={showRenameRunModal}
          renamedRunName={renamedRunName}
          refreshRuns={refreshRuns}
        />
      </>
    );
  },
);

const styles = {
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
  }),
};
