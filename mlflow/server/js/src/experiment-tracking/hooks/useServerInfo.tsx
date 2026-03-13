import type { ReactNode } from 'react';
import React, { useEffect } from 'react';
import type { QueryClient } from '../../common/utils/reactQueryHooks';
import { useQuery, useQueryClient } from '../../common/utils/reactQueryHooks';
import { getAjaxUrl, getDefaultHeaders } from '../../common/utils/FetchUtils';

export const SERVER_INFO_QUERY_KEY = 'serverInfo';

interface ServerInfoResponse {
  store_type: string | null;
  workspaces_enabled: boolean;
}

// Default response when the API call fails (e.g., older server without this endpoint)
const DEFAULT_RESPONSE: ServerInfoResponse = { store_type: '', workspaces_enabled: false };

// Module-level reference to the QueryClient for synchronous access
let queryClientRef: QueryClient | null = null;

/**
 * Fetches server info from the backend.
 * Returns default response if the request fails.
 * Uses default headers for OAuth/K8s deployments that rely on cookie-derived headers + Authorization.
 */
async function fetchServerInfo(): Promise<ServerInfoResponse> {
  try {
    let cookieString = '';
    if (typeof document !== 'undefined' && typeof document.cookie === 'string') {
      cookieString = document.cookie || '';
    }

    const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/server-info'), {
      method: 'GET',
      headers: {
        ...getDefaultHeaders(cookieString),
      },
    });
    if (!response.ok) {
      // If the endpoint doesn't exist or returns an error, return default
      return DEFAULT_RESPONSE;
    }
    return response.json();
  } catch {
    // Network error or other failure - return default
    return DEFAULT_RESPONSE;
  }
}

/**
 * Hook to get server info from the backend.
 * This information is fetched once and cached for the session.
 */
export function useServerInfo() {
  return useQuery({
    queryKey: [SERVER_INFO_QUERY_KEY],
    queryFn: fetchServerInfo,
    staleTime: Infinity, // This info doesn't change during the session
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    retry: false,
  });
}

/**
 * Hook to check if the tracking store is using FileStore.
 * Returns true if FileStore is being used, false otherwise.
 * Returns undefined while loading.
 */
export function useIsFileStore(): boolean | undefined {
  const { data } = useServerInfo();
  return data ? data.store_type === 'FileStore' : undefined;
}

interface ServerInfoProviderProps {
  children: ReactNode;
}

/**
 * Provider component that captures the QueryClient reference for synchronous access.
 * Wrap your app with this inside QueryClientProvider.
 */
export const ServerInfoProvider = ({ children }: ServerInfoProviderProps) => {
  const queryClient = useQueryClient();

  // Store queryClient reference for synchronous access during render
  queryClientRef = queryClient;

  useEffect(() => {
    return () => {
      // Only clear if this is still the active reference
      if (queryClientRef === queryClient) {
        queryClientRef = null;
      }
    };
  }, [queryClient]);

  // Trigger the query so it's cached for synchronous access
  useServerInfo();

  return <>{children}</>;
};

export const useWorkspacesEnabled = (): { workspacesEnabled: boolean; loading: boolean } => {
  const { data, isLoading } = useServerInfo();
  return {
    workspacesEnabled: data?.workspaces_enabled ?? false,
    loading: isLoading,
  };
};

// For synchronous access (e.g., in WorkspaceUtils)
// Returns the cached value from React Query or false if not yet loaded
export const getWorkspacesEnabledSync = (): boolean => {
  const cachedData = queryClientRef?.getQueryData<ServerInfoResponse>([SERVER_INFO_QUERY_KEY]);
  return cachedData?.workspaces_enabled ?? false;
};

// For testing purposes - allows resetting the cached state
export const resetServerInfoCache = (): void => {
  queryClientRef?.removeQueries({ queryKey: [SERVER_INFO_QUERY_KEY] });
};
