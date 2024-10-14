import { useCallback } from 'react';
import DeleteRunModal from '../../../modals/DeleteRunModal';
import { RenameRunModal } from '../../../modals/RenameRunModal';
import RestoreRunModal from '../../../modals/RestoreRunModal';

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
  refreshRuns,
}: ExperimentViewModalsProps) => {
  const selectedRunIds = Object.entries(runsSelected)
    .filter(([, selected]) => selected)
    .map(([key]) => key);

  return (
    <>
      <DeleteRunModal
        isOpen={showDeleteRunModal}
        onClose={onCloseDeleteRunModal}
        selectedRunIds={selectedRunIds}
        onSuccess={() => {
          refreshRuns();
        }}
      />
      <RestoreRunModal
        isOpen={showRestoreRunModal}
        onClose={onCloseRestoreRunModal}
        selectedRunIds={selectedRunIds}
        onSuccess={() => {
          refreshRuns();
        }}
      />
      <RenameRunModal
        runUuid={selectedRunIds[0]}
        onClose={onCloseRenameRunModal}
        runName={renamedRunName}
        isOpen={showRenameRunModal}
        onSuccess={() => {
          refreshRuns();
        }}
      />
    </>
  );
};
