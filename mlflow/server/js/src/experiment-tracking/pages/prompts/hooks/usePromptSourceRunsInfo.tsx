import { QueryFunctionContext, useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { compact, uniq } from 'lodash';
import { useMemo } from 'react';
import { transformGetRunResponse } from '../../../sdk/FieldNameTransformers';
import { MlflowService } from '../../../sdk/MlflowService';
import { GetRunApiResponse } from '../../../types';
import { RegisteredPromptVersion } from '../types';
import { REGISTERED_PROMPT_SOURCE_RUN_ID } from '../utils';

type UseRegisteredModelRelatedRunNamesQueryKey = ['prompt_source_run', string];

export const usePromptSourceRunsInfo = (registeredPromptVersions: RegisteredPromptVersion[] = []) => {
  const runUuids = useMemo(
    () =>
      uniq(
        compact(
          registeredPromptVersions.map(
            (version) => version?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_SOURCE_RUN_ID)?.value,
          ),
        ),
      ),
    [registeredPromptVersions],
  );

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
