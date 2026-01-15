import React, { createContext, useContext, useMemo, ReactNode, useEffect } from 'react';
import { useQuery, useQueryClient, QueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from './FetchUtils';

const SERVER_FEATURES_ENDPOINT = 'ajax-api/2.0/mlflow/server-features';

// Query key for React Query caching and deduplication
export const SERVER_FEATURES_QUERY_KEY = ['serverFeatures'] as const;

interface ServerFeatures {
  workspacesEnabled: boolean;
}

interface ServerFeaturesContextValue {
  features: ServerFeatures;
  loading: boolean;
  error: Error | null;
}

const defaultFeatures: ServerFeatures = {
  workspacesEnabled: false, // Default to disabled until we know for sure
};

const ServerFeaturesContext = createContext<ServerFeaturesContextValue>({
  features: defaultFeatures,
  loading: true,
  error: null,
});

// Module-level reference to the QueryClient for synchronous access
let queryClientRef: QueryClient | null = null;

/**
 * Query function for fetching server features.
 * Handles 404 (old backend) by returning workspacesEnabled: false.
 */
async function fetchServerFeatures(): Promise<ServerFeatures> {
  try {
    const response = await fetch(getAjaxUrl(SERVER_FEATURES_ENDPOINT));

    if (response.status === 404) {
      // Old backend that doesn't support the endpoint - assume workspaces disabled
      return { workspacesEnabled: false };
    }

    if (!response.ok) {
      // Other error - assume workspaces disabled for safety
      return { workspacesEnabled: false };
    }

    const data = await response.json();
    return {
      workspacesEnabled: Boolean(data?.workspaces_enabled),
    };
  } catch {
    // Network error or other issue - assume workspaces disabled
    return { workspacesEnabled: false };
  }
}

interface ServerFeaturesProviderProps {
  children: ReactNode;
}

export const ServerFeaturesProvider = ({ children }: ServerFeaturesProviderProps) => {
  const queryClient = useQueryClient();

  // Store queryClient reference for synchronous access
  useEffect(() => {
    queryClientRef = queryClient;
    return () => {
      // Only clear if this is still the active reference
      if (queryClientRef === queryClient) {
        queryClientRef = null;
      }
    };
  }, [queryClient]);

  const { data, isLoading, error } = useQuery<ServerFeatures, Error, ServerFeatures, typeof SERVER_FEATURES_QUERY_KEY>(
    SERVER_FEATURES_QUERY_KEY,
    {
      queryFn: fetchServerFeatures,
      staleTime: Infinity, // Never refetch automatically - features don't change during session
      cacheTime: Infinity, // Keep in cache indefinitely
      retry: false, // Don't retry - we handle errors gracefully by returning disabled
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: false,
    },
  );

  const features: ServerFeatures = data ?? defaultFeatures;

  const value = useMemo(
    () => ({
      features,
      loading: isLoading,
      error: error ?? null,
    }),
    [features, isLoading, error],
  );

  return <ServerFeaturesContext.Provider value={value}>{children}</ServerFeaturesContext.Provider>;
};

export const useServerFeatures = (): ServerFeaturesContextValue => {
  return useContext(ServerFeaturesContext);
};

export const useWorkspacesEnabled = (): { workspacesEnabled: boolean; loading: boolean } => {
  const { features, loading } = useServerFeatures();
  return {
    workspacesEnabled: features.workspacesEnabled,
    loading,
  };
};

// For synchronous access (e.g., in WorkspaceUtils)
// Returns the cached value from React Query or false if not yet loaded
export const getWorkspacesEnabledSync = (): boolean => {
  const cachedData = queryClientRef?.getQueryData<ServerFeatures>(SERVER_FEATURES_QUERY_KEY);
  return cachedData?.workspacesEnabled ?? false;
};

// For testing purposes - allows resetting the cached state
export const resetServerFeaturesCache = (): void => {
  queryClientRef?.removeQueries({ queryKey: SERVER_FEATURES_QUERY_KEY });
};
