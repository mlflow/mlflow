import { useCallback } from 'react';
import DeleteRunModal from '../../../modals/DeleteRunModal';
import { RenameRunModal } from '../../../modals/RenameRunModal';
import RestoreRunModal from '../../../modals/RestoreRunModal';
import { useFetchExperimentRuns } from '../../hooks/useFetchExperimentRuns';

export interface ExperimentViewModalsProps {
  showDeleteRunModal: boolean;
  showRestoreRunModal: boolean;
  showRenameRunModal: boolean;
  runsSelected: Record<string, boolean>;
  onCloseDeleteRunModal: () => void;
  onCloseRestoreRunModal: () => void;
  onCloseRenameRunModal: () => void;
  renamedRunName: string;
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
}: ExperimentViewModalsProps) => {
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
      />
      <RestoreRunModal
        isOpen={showRestoreRunModal}
        onClose={onCloseRestoreRunModal}
        selectedRunIds={selectedRunIds}
      />
      <RenameRunModal
        runUuid={selectedRunIds[0]}
        onClose={onCloseRenameRunModal}
        runName={renamedRunName}
        isOpen={showRenameRunModal}
        onSuccess={() => refreshRuns()}
      />
    </>
  );
};
