import { useCallback } from 'react';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { useQuery } from '@databricks/web-shared/query-client';
import type { RunEntity } from '../../../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';

export const EVAL_RUNS_BASELINE_TAG = 'mlflow.evaluation.baselineRunId';

interface ExperimentTag {
  key?: string | null;
  value?: string | null;
}

export const getEvalRunsBaselineRunUuid = (tags?: readonly ExperimentTag[] | null): string | undefined => {
  const value = tags?.find(({ key }) => key === EVAL_RUNS_BASELINE_TAG)?.value;
  if (!value) {
    return undefined;
  }
  try {
    const legacyValue: unknown = JSON.parse(value);
    if (
      typeof legacyValue === 'object' &&
      legacyValue !== null &&
      'runUuid' in legacyValue &&
      typeof legacyValue.runUuid === 'string'
    ) {
      return legacyValue.runUuid;
    }
    return typeof legacyValue === 'object' ? undefined : value;
  } catch {
    return value;
  }
};

export const useGetEvalRunsBaseline = (runUuid?: string) =>
  useQuery<RunEntity, Error>({
    queryKey: ['MLFLOW_EVAL_RUNS_BASELINE', runUuid],
    queryFn: async () => (await MlflowService.getRun({ run_id: runUuid })).run,
    enabled: Boolean(runUuid),
    // A deleted baseline run fails every time, so don't retry it.
    retry: false,
    refetchOnWindowFocus: false,
  });

export const isMatchingEvalRunsBaseline = (
  run: { info: { runUuid?: string; experimentId?: string; lifecycleStage?: string | null } } | undefined,
  runUuid: string,
  experimentId: string,
) =>
  run?.info.runUuid === runUuid &&
  run.info.experimentId === experimentId &&
  run.info.lifecycleStage?.toLowerCase() === 'active';

export const useSetEvalRunsBaseline = ({
  experimentId,
  onSuccess,
  onError,
}: {
  experimentId: string;
  onSuccess: () => void;
  onError?: (error: unknown) => void;
}) => {
  // Errors stay as the request's ErrorWrapper, so the notification shows the server's message (with basic
  // auth, a plain-text "Permission denied").
  const mutation = useMutation<unknown, unknown, string | undefined>({
    mutationFn: (runUuid) =>
      runUuid
        ? MlflowService.setExperimentTag({
            experiment_id: experimentId,
            key: EVAL_RUNS_BASELINE_TAG,
            value: runUuid,
          })
        : MlflowService.deleteExperimentTag({
            experiment_id: experimentId,
            key: EVAL_RUNS_BASELINE_TAG,
          }),
    onSuccess,
    onError,
  });

  return {
    setBaseline: mutation.mutate,
    clearBaseline: () => mutation.mutate(undefined),
    isSaving: mutation.isLoading,
  };
};

/** The experiment's baseline run and the controls to change it. Only mounted with the new UI. */
export const useEvalRunsBaselineState = (experimentId: string) => {
  const { data: experiment, refetch: refetchExperiment } = useGetExperimentQuery({ experimentId });
  const baselineRunUuid = getEvalRunsBaselineRunUuid(experiment?.tags);
  const { data: fetchedBaselineRun, refetch: refetchBaselineRun } = useGetEvalRunsBaseline(baselineRunUuid);
  const {
    setBaseline,
    clearBaseline,
    isSaving: isSavingBaseline,
  } = useSetEvalRunsBaseline({
    experimentId,
    onSuccess: () => {
      void refetchExperiment();
    },
    onError: (baselineError) => Utils.logErrorAndNotifyUser(baselineError),
  });
  const refetchBaseline = useCallback(() => {
    void refetchExperiment();
    if (baselineRunUuid) {
      void refetchBaselineRun();
    }
  }, [baselineRunUuid, refetchBaselineRun, refetchExperiment]);

  return {
    baselineRunUuid,
    fetchedBaselineRun,
    setBaseline,
    clearBaseline,
    isSavingBaseline,
    refetchBaseline,
  };
};

export type EvalRunsBaselineState = ReturnType<typeof useEvalRunsBaselineState>;
