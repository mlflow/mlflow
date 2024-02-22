import { useCallback } from 'react';
import DeleteRunModal from '../../../modals/DeleteRunModal';
import { RenameRunModal } from '../../../modals/RenameRunModal';
import RestoreRunModal from '../../../modals/RestoreRunModal';
import { useFetchExperimentRuns } from '../../hooks/useFetchExperimentRuns';
import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';

export interface ExperimentViewModalsProps {
  showDeleteRunModal: boolean;
  showRestoreRunModal: boolean;
  showRenameRunModal: boolean;
  runsSelected: Record<string, boolean>;
  onCloseDeleteRunModal: () => void;
  onCloseRestoreRunModal: () => void;
  onCloseRenameRunModal: () => void;
  renamedRunName: string;
  refreshRuns: () => void;
}

/**
 * A component that contains modals required for the run
 * management, i.e. delete and restore actions.
 */
export const ExperimentViewRunModals = ({
  showDeleteRunModal,
  showRestoreRunModal,
  showRenameRunModal,
  runsSelected,
  onCloseDeleteRunModal,
  onCloseRestoreRunModal,
  onCloseRenameRunModal,
  renamedRunName,
  refreshRuns: refreshRunsFromProps,
}: ExperimentViewModalsProps) => {
  // TODO(ML-35962): Use refreshRuns() from props, remove updateSearchFacets after migration to new view state model
  const { updateSearchFacets } = useFetchExperimentRuns();

  const selectedRunIds = Object.entries(runsSelected)
    .filter(([, selected]) => selected)
    .map(([key]) => key);

  /**
   * Function used to refresh the list after renaming the run
   */
  const refreshRuns = useCallback(
    () =>
      updateSearchFacets(
        {},
        {
          forceRefresh: true,
          preservePristine: true,
        },
      ),
    [updateSearchFacets],
  );

  return (
    <>
      <DeleteRunModal
        isOpen={showDeleteRunModal}
        onClose={onCloseDeleteRunModal}
        selectedRunIds={selectedRunIds}
        onSuccess={() => {
          if (shouldEnableShareExperimentViewByTags()) {
            refreshRunsFromProps();
          }
        }}
      />
      <RestoreRunModal
        isOpen={showRestoreRunModal}
        onClose={onCloseRestoreRunModal}
        selectedRunIds={selectedRunIds}
        onSuccess={() => {
          if (shouldEnableShareExperimentViewByTags()) {
            refreshRunsFromProps();
          }
        }}
      />
      <RenameRunModal
        runUuid={selectedRunIds[0]}
        onClose={onCloseRenameRunModal}
        runName={renamedRunName}
        isOpen={showRenameRunModal}
        onSuccess={() => {
          if (shouldEnableShareExperimentViewByTags()) {
            refreshRunsFromProps();
          } else {
            refreshRuns();
          }
        }}
      />
    </>
  );
};
