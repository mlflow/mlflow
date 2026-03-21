import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';

export const useDeleteRuns = ({ onSuccess, onError }: { onSuccess: () => void; onError?: (error: Error) => void }) => {
  const { mutate, isLoading } = useMutation({
    mutationFn: ({ runUuids }: { runUuids: string[] }) =>
      Promise.all(runUuids.map((runUuid) => MlflowService.deleteRun({ run_id: runUuid }))),
    onSuccess,
    onError,
  });

  return { mutate, isLoading };
};
