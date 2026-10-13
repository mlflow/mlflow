import { useMemo } from 'react';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchAPI, getAjaxUrl, HTTPMethods } from '../../common/utils/FetchUtils';
import type { Workspace } from '../types';

type WorkspacesResponse = {
  workspaces?: Workspace[];
};

const WORKSPACES_ENDPOINT = 'ajax-api/3.0/mlflow/workspaces';

/**
 * Custom hook to fetch and manage workspaces data.
 * Uses React Query for efficient data fetching and caching.
 */
export const useWorkspaces = (enabled: boolean) => {
  // Fetch workspaces using React Query
  const { data, isLoading, isError, refetch } = useQuery<WorkspacesResponse, Error>({
    queryKey: ['workspaces'],
    queryFn: async () => {
      // Don't send X-MLFLOW-WORKSPACE header when listing workspaces
      // Otherwise we get stuck when current workspace is filtered out
      return fetchAPI(getAjaxUrl(WORKSPACES_ENDPOINT), {
        method: HTTPMethods.GET,
        headers: {
          'X-MLFLOW-WORKSPACE': '',
        },
      });
    },
    enabled,
    refetchOnWindowFocus: false,
    retry: false,
  });

  // Transform and filter workspaces data
  const workspaces = useMemo(() => {
    const fetched = Array.isArray(data?.workspaces) ? data.workspaces : [];
    const filteredWorkspaces: Workspace[] = [];
    for (const item of fetched as Array<Workspace | Record<string, unknown>>) {
      if (item && typeof (item as Workspace)?.name === 'string') {
        const workspaceItem = item as Workspace;
        const traceArchivalConfig =
          workspaceItem.trace_archival_config && typeof workspaceItem.trace_archival_config === 'object'
            ? {
                location: workspaceItem.trace_archival_config.location ?? null,
                retention: workspaceItem.trace_archival_config.retention ?? null,
              }
            : null;
        filteredWorkspaces.push({
          name: workspaceItem.name,
          description: workspaceItem.description ?? null,
          default_artifact_root: workspaceItem.default_artifact_root ?? null,
          trace_archival_config: traceArchivalConfig,
        });
      }
    }
    return filteredWorkspaces;
  }, [data]);

  return {
    workspaces,
    isLoading,
    isError,
    refetch,
  } as const;
};

export type { Workspace };
