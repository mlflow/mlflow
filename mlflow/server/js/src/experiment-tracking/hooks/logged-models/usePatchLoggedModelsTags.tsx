import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { entries } from 'lodash';
import { loggedModelsDataRequest } from './request.utils';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const usePatchLoggedModelsTags = ({ loggedModelId }: { loggedModelId?: string }) => {
  const { isLoading, error, mutateAsync } = useMutation<unknown, Error, Record<string, string>>({
    mutationFn: async (variables) => {
      const requestBody = {
        tags: entries(variables).map(([key, value]) => ({ key, value })),
      };

      return loggedModelsDataRequest(
        getAjaxUrl(`ajax-api/2.0/mlflow/logged-models/${loggedModelId}/tags`),
        'PATCH',
        requestBody,
      );
    },
  });

  return {
    isLoading,
    error,
    patch: mutateAsync,
  } as const;
};
