import { QueryFunctionContext, useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { transformGetRunResponse } from '../../../sdk/FieldNameTransformers';
import { MlflowService } from '../../../sdk/MlflowService';
import { GetRunApiResponse } from '../../../types';

type UseRegisteredModelRelatedRunNamesQueryKey = ['prompt_source_run', string];

export const usePromptSourceRunsInfo = (runUuids: string[] = []) => {
  const queryResults = useQueries({
    queries: runUuids.map((runUuid) => ({
      queryKey: ['prompt_source_run', runUuid] as UseRegisteredModelRelatedRunNamesQueryKey,
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

  return {
    isLoading: runUuids.length > 0 && queryResults.some((queryResult) => queryResult.isLoading),
    sourceRunInfos: queryResults.map((queryResult) => queryResult.data?.run?.info),
  };
};
