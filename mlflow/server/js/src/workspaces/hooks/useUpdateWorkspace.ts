import { useMutation, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchAPI, getAjaxUrl, HTTPMethods } from '../../common/utils/FetchUtils';
import type { Workspace, WorkspaceTraceArchivalConfigInput } from '../types';

type UpdateWorkspaceParams = {
  name: string;
  description?: string;
  default_artifact_root?: string;
  trace_archival_config?: WorkspaceTraceArchivalConfigInput;
};

type UpdateWorkspaceResponse = {
  workspace: Workspace;
};

/**
 * Custom hook to update workspace metadata.
 * Uses React Query mutation for efficient state management.
 */
export const useUpdateWorkspace = () => {
  const queryClient = useQueryClient();

  return useMutation<UpdateWorkspaceResponse, Error, UpdateWorkspaceParams>({
    mutationFn: async ({ name, description, default_artifact_root, trace_archival_config }) => {
      const requestBody: {
        description?: string;
        default_artifact_root?: string;
        trace_archival_config?: WorkspaceTraceArchivalConfigInput;
      } = {};

      if (description !== undefined) {
        requestBody.description = description;
      }

      if (default_artifact_root !== undefined) {
        requestBody.default_artifact_root = default_artifact_root;
      }

      if (trace_archival_config) {
        const requestTraceArchivalConfig: WorkspaceTraceArchivalConfigInput = {};

        if (trace_archival_config.location !== undefined) {
          requestTraceArchivalConfig.location = trace_archival_config.location;
        }

        if (trace_archival_config.retention !== undefined) {
          requestTraceArchivalConfig.retention = trace_archival_config.retention;
        }

        if (Object.keys(requestTraceArchivalConfig).length > 0) {
          requestBody.trace_archival_config = requestTraceArchivalConfig;
        }
      }

      return fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/workspaces/${encodeURIComponent(name)}`), {
        method: HTTPMethods.PATCH,
        body: JSON.stringify(requestBody),
        headers: { 'X-MLFLOW-WORKSPACE': '' },
      });
    },
    onSuccess: () => {
      // Invalidate workspaces query to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['workspaces'] });
    },
  });
};
