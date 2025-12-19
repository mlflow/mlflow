import type { QueryFunctionContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { transformGetRunResponse } from '../../../sdk/FieldNameTransformers';
import { MlflowService } from '../../../sdk/MlflowService';
import type { GetRunApiResponse } from '../../../types';

type UseRegisteredModelRelatedRunNamesQueryKey = ['prompt_associated_runs', string];

export const usePromptRunsInfo = (runUuids: string[] = []) => {
  const queryResults = useQueries({
    queries: runUuids.map((runUuid) => ({
      queryKey: ['prompt_associated_runs', runUuid] as UseRegisteredModelRelatedRunNamesQueryKey,
      queryFn: async ({
        queryKey: [, runUuid],
      }: QueryFunctionContext<UseRegisteredModelRelatedRunNamesQueryKey>): Promise<GetRunApiResponse | null> => {
        try {
          const data = await MlflowService.getRun({ run_id: runUuid });
          return transformGetRunResponse(data);
        } catch (e) {
          return null;
        }
      },
    })),
  });

  // Create a map of run_id to run info
  const runInfoMap: Record<string, any | undefined> = {};

  queryResults.forEach((queryResult, index) => {
    const runUuid = runUuids[index];
    runInfoMap[runUuid] = queryResult.data?.run?.info;
  });

  return {
    isLoading: runUuids.length > 0 && queryResults.some((queryResult) => queryResult.isLoading),
    runInfoMap,
  };
};
