import { useCallback, useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import type { ExperimentEntity } from '../../../types';
import { DeleteExperimentModal } from '../../modals/DeleteExperimentModal';
import { MlflowService } from '../../../sdk/MlflowService';
import { useInvalidateExperimentList } from './useExperimentListQuery';

export interface ExperimentManagementActions {
  deleteExperiment: () => void;
  overlayBodies: ReactNode;
  saveExperimentName: (name: string) => Promise<void>;
}

/**
 * Experiment management actions for the Settings page. Like the experiment's ⋯ menu, they're offered to
 * everyone, and with authentication the server rejects one the user isn't allowed to make.
 */
export const useExperimentManagementActions = ({
  experiment,
  refetchExperiment,
}: {
  experiment: ExperimentEntity;
  refetchExperiment?: () => Promise<unknown>;
}): ExperimentManagementActions => {
  const navigate = useNavigate();
  const invalidateExperimentList = useInvalidateExperimentList();
  const [showDeleteExperimentModal, setShowDeleteExperimentModal] = useState(false);

  const deleteExperiment = useCallback(() => setShowDeleteExperimentModal(true), []);
  const saveExperimentName = useCallback(
    async (name: string) => {
      await MlflowService.updateExperiment({ experiment_id: experiment.experimentId, new_name: name });
      invalidateExperimentList();
      // The rename has saved; a failed refresh afterwards isn't a failed rename.
      await refetchExperiment?.()?.catch(() => undefined);
    },
    [experiment.experimentId, invalidateExperimentList, refetchExperiment],
  );

  const overlayBodies = (
    <DeleteExperimentModal
      experimentId={experiment.experimentId}
      experimentName={experiment.name}
      isOpen={showDeleteExperimentModal}
      onClose={() => setShowDeleteExperimentModal(false)}
      onExperimentDeleted={() => {
        invalidateExperimentList();
        navigate(Routes.experimentsObservatoryRoute);
      }}
    />
  );

  return { deleteExperiment, overlayBodies, saveExperimentName };
};
