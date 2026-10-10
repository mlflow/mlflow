import { useCallback, useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import type { ExperimentEntity } from '../../../types';
import { DeleteExperimentModal } from '../../modals/DeleteExperimentModal';
import { useInvalidateExperimentList } from './useExperimentListQuery';
import { MlflowService } from '../../../sdk/MlflowService';
import { canDeleteExperiment, canRenameExperiment } from '../utils/experimentPage.common-utils';

export interface ExperimentManagementActions {
  canDelete: boolean;
  canRename: boolean;
  deleteExperiment: () => void;
  overlayBodies: ReactNode;
  saveExperimentName: (name: string) => Promise<void>;
}

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
  const saveExperimentName = useCallback(
    async (name: string) => {
      const parentPath = experiment.name.split('/').slice(0, -1).join('/');
      await MlflowService.updateExperiment({
        experiment_id: experiment.experimentId,
        new_name: parentPath ? `${parentPath}/${name}` : name,
      });
      invalidateExperimentList();
      await refetchExperiment?.();
    },
    [experiment.experimentId, experiment.name, invalidateExperimentList, refetchExperiment],
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

  return {
    canDelete: canDeleteExperiment(experiment),
    canRename: canRenameExperiment(experiment),
    deleteExperiment: () => setShowDeleteExperimentModal(true),
    overlayBodies,
    saveExperimentName,
  };
};
