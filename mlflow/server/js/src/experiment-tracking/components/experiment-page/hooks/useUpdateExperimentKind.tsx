import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { ExperimentKind } from '../../../constants';
import { MlflowService } from '../../../sdk/MlflowService';

/**
 * An utility wrapper hook to update the experiment kind.
 * The success callback is optional but it's part of the mutation to include it in the loading state.
 */
export const useUpdateExperimentKind = (onSuccess?: () => void) =>
  useMutation<unknown, Error, { experimentId: string; kind: ExperimentKind }>({
    mutationFn: ({ experimentId, kind }) =>
      MlflowService.setExperimentTag({
        experiment_id: experimentId,
        key: 'mlflow.experimentKind',
        value: kind,
      }).then(() => onSuccess?.() ?? Promise.resolve()),
  });
