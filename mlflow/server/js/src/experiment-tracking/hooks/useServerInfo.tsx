import { useQuery } from '../../common/utils/reactQueryHooks';
import { getAjaxUrl } from '../../common/utils/FetchUtils';

export const SERVER_INFO_QUERY_KEY = 'serverInfo';

interface ServerInfoResponse {
  store_type: string;
}

// Default response when the API call fails (e.g., older server without this endpoint)
const DEFAULT_RESPONSE: ServerInfoResponse = { store_type: '' };

/**
 * Fetches server info from the backend.
 * Returns default response (store_type: '') if the request fails.
 */
async function fetchServerInfo(): Promise<ServerInfoResponse> {
  try {
    const response = await fetch(getAjaxUrl('server-info'), {
      method: 'GET',
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
