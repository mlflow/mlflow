import { useCallback, useMemo, useState } from 'react';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { MlflowService } from '../../../sdk/MlflowService';
import {
  EVAL_RUNS_BASELINE_TAG,
  parseBaselineTag,
  serializeBaselineTag,
  type EvalRunsBaselineTagValue,
} from '../EvalRunsBaseline.utils';

/**
 * The baseline lives in an experiment tag, so it is shared by everyone looking
 * at the experiment. Reading it is free — `useGetExperimentQuery` already
 * selects `tags` — and writing it is a single `setExperimentTag` call.
 */
export const useEvalRunsBaseline = ({ experimentId }: { experimentId: string }) => {
  const { data: experiment, loading, refetch } = useGetExperimentQuery({ experimentId });
  const [isSaving, setIsSaving] = useState(false);

  const baseline: EvalRunsBaselineTagValue | undefined = useMemo(() => {
    const tag = experiment?.tags?.find(({ key }) => key === EVAL_RUNS_BASELINE_TAG);
    return parseBaselineTag(tag?.value);
  }, [experiment?.tags]);

  const setBaseline = useCallback(
    async (runUuid: string, setBy?: string) => {
      setIsSaving(true);
      try {
        await MlflowService.setExperimentTag({
          experiment_id: experimentId,
          key: EVAL_RUNS_BASELINE_TAG,
          value: serializeBaselineTag({ runUuid, setBy, setAt: Date.now() }),
        });
        await refetch();
      } finally {
        setIsSaving(false);
      }
    },
    [experimentId, refetch],
  );

  const clearBaseline = useCallback(async () => {
    setIsSaving(true);
    try {
      await MlflowService.deleteExperimentTag({ experiment_id: experimentId, key: EVAL_RUNS_BASELINE_TAG });
      await refetch();
    } finally {
      setIsSaving(false);
    }
  }, [experimentId, refetch]);

  return { baseline, baselineRunUuid: baseline?.runUuid, isLoading: loading, isSaving, setBaseline, clearBaseline };
};
