import type { ReactNode } from 'react';
import { useEffect } from 'react';
import type { QueryClient } from '../../common/utils/reactQueryHooks';
import { useQuery, useQueryClient } from '../../common/utils/reactQueryHooks';
import { fetchAPI, getAjaxUrl } from '../../common/utils/FetchUtils';

const SERVER_INFO_QUERY_KEY = 'serverInfo';

interface ServerInfoResponse {
  store_type: string | null;
  workspaces_enabled: boolean;
  trace_archival_enabled: boolean;
  multipart_uploads_enabled: boolean;
  multipart_downloads_enabled: boolean;
  features_enabled?: Record<FeatureKey, boolean>;
}

/**
 * Valid keys for the `features_enabled` map in server-info.
 * Add new entries here when introducing a new runtime feature toggle.
 */
export const SERVER_FEATURE_KEYS = {
  GATEWAY: 'gateway',
} as const;

export type FeatureKey = (typeof SERVER_FEATURE_KEYS)[keyof typeof SERVER_FEATURE_KEYS];

// Default response when the API call fails (e.g., older server without this endpoint)
const DEFAULT_RESPONSE: ServerInfoResponse = {
  store_type: '',
  workspaces_enabled: false,
  trace_archival_enabled: false,
  multipart_uploads_enabled: false,
  multipart_downloads_enabled: false,
};

// Module-level reference to the QueryClient for synchronous access
let queryClientRef: QueryClient | null = null;

/**
 * How long to wait for server-info before giving up and using DEFAULT_RESPONSE.
 *
 * This is a backstop for a request that never settles, not a latency budget. MlflowRouter holds
 * the whole UI on a skeleton until this query resolves, and without a bound there is nothing to
 * resolve it.
 *
 * The bound is deliberately generous because the two failure directions are not symmetric. Timing
 * out late costs a few more seconds of skeleton. Timing out early lands the app in its
 * "workspaces disabled" fallback against a server that has them enabled, and that state is cached
 * for the session, so it does not self-correct. This endpoint can also legitimately be slow on a
 * cold server: it initialises the tracking store and, with proxied artifacts, may construct an
 * artifact repository client before it answers.
 */
export const SERVER_INFO_TIMEOUT_MS = 30_000;

/**
 * Fetches server info from the backend.
 * Returns default response if the request fails or does not respond in time.
 * Uses default headers for OAuth/K8s deployments that rely on cookie-derived headers + Authorization.
 */
async function fetchServerInfo(): Promise<ServerInfoResponse> {
  // AbortSignal.timeout would be tidier, but browserslist still includes chrome >= 94 and it
  // requires 103+.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), SERVER_INFO_TIMEOUT_MS);

  try {
    return await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/server-info'), { signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      // Without this the timeout is indistinguishable from a server that reports these defaults.
      // eslint-disable-next-line no-console
      console.warn(`MLflow: server-info did not respond within ${SERVER_INFO_TIMEOUT_MS}ms; using defaults.`);
    }
    // Network error, abort, or other failure - return default
    return DEFAULT_RESPONSE;
  } finally {
    clearTimeout(timeoutId);
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

export function useTraceArchivalEnabled(): boolean {
  const { data } = useServerInfo();
  return data?.trace_archival_enabled ?? false;
}

export function useMultipartDownloadsEnabled(): boolean {
  const { data } = useServerInfo();
  return data?.multipart_downloads_enabled ?? false;
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

export const getMultipartDownloadsEnabledSync = (): boolean => {
  const cachedData = queryClientRef?.getQueryData<ServerInfoResponse>([SERVER_INFO_QUERY_KEY]);
  return cachedData?.multipart_downloads_enabled ?? false;
};

/**
 * Subscribes React components to a server feature value and re-renders when server-info loads.
 * Prefer this hook in React render paths.
 */
export const useFeatureEnabled = (key: FeatureKey, defaultValue = true): boolean => {
  const { data } = useServerInfo();
  return data?.features_enabled?.[key] ?? defaultValue;
};

/**
 * Reads a server feature from the current cache without subscribing to updates.
 * Use this accessor only where React hooks are unavailable.
 */
export const getFeatureEnabledSync = (key: FeatureKey, defaultValue = true): boolean => {
  const cachedData = queryClientRef?.getQueryData<ServerInfoResponse>([SERVER_INFO_QUERY_KEY]);
  return cachedData?.features_enabled?.[key] ?? defaultValue;
};

// For testing purposes - allows resetting the cached state
export const resetServerInfoCache = (): void => {
  queryClientRef?.removeQueries({ queryKey: [SERVER_INFO_QUERY_KEY] });
};
