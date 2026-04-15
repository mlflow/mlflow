import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchAPI, getAjaxUrl, HTTPMethods } from '../../common/utils/FetchUtils';

type UpdateWorkspaceParams = {
  name: string;
  description?: string;
  default_artifact_root?: string;
};

type UpdateWorkspaceResponse = {
  workspace: {
    name: string;
    description?: string;
    default_artifact_root?: string;
  };
};

/**
 * Custom hook to update workspace metadata.
 * Uses React Query mutation for efficient state management.
 */
export const useUpdateWorkspace = () => {
  const queryClient = useQueryClient();

  return useMutation<UpdateWorkspaceResponse, Error, UpdateWorkspaceParams>({
    mutationFn: async ({ name, description, default_artifact_root }) => {
      const requestBody: { description?: string; default_artifact_root?: string } = {};

      if (description !== undefined) {
        requestBody.description = description;
      }

      if (default_artifact_root !== undefined) {
        requestBody.default_artifact_root = default_artifact_root;
      }

      return fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/workspaces/${encodeURIComponent(name)}`), {
        method: HTTPMethods.PATCH,
        body: JSON.stringify(requestBody),
      });
    },
    onSuccess: () => {
      // Invalidate workspaces query to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['workspaces'] });
    },
  });
};
