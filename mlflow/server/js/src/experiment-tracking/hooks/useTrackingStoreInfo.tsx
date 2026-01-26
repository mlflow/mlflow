import { useQuery } from '../../common/utils/reactQueryHooks';
import { getAjaxUrl } from '../../common/utils/FetchUtils';

export const TRACKING_STORE_INFO_QUERY_KEY = 'trackingStoreInfo';

interface TrackingStoreInfoResponse {
  is_file_store: boolean;
}

// Default response when the API call fails (e.g., older server without this endpoint)
const DEFAULT_RESPONSE: TrackingStoreInfoResponse = { is_file_store: false };

/**
 * Fetches tracking store info from the backend.
 * Returns default response (is_file_store: false) if the request fails.
 */
async function fetchTrackingStoreInfo(): Promise<TrackingStoreInfoResponse> {
  try {
    const response = await fetch(getAjaxUrl('tracking-store-info'), {
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
 * Hook to check if the tracking store backend is using FileStore.
 * This information is fetched once and cached for the session.
 */
export function useTrackingStoreInfo() {
  return useQuery({
    queryKey: [TRACKING_STORE_INFO_QUERY_KEY],
    queryFn: fetchTrackingStoreInfo,
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
  const { data } = useTrackingStoreInfo();
  return data?.is_file_store;
}
