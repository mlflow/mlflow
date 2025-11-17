import { useEffect, useState } from 'react';

interface BackendInfo {
  store_type: string;
  is_sql_backend: boolean;
}

interface UseBackendSupportResult {
  isSqlBackend: boolean | null;
  storeType: string | null;
  isLoading: boolean;
  error: Error | null;
}

/**
 * Hook to check if the backend supports SQL-only features (like Gateway/Secrets).
 *
 * @returns {UseBackendSupportResult} Backend support information
 * - isSqlBackend: true if SQL backend, false if FileStore, null if loading
 * - storeType: The backend store type (e.g., "SqlAlchemyStore", "FileStore")
 * - isLoading: true while fetching backend info
 * - error: Error if the request failed
 */
export function useBackendSupport(): UseBackendSupportResult {
  const [isSqlBackend, setIsSqlBackend] = useState<boolean | null>(null);
  const [storeType, setStoreType] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchBackendInfo = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/ajax-api/3.0/mlflow/backend-info');

        if (!response.ok) {
          throw new Error(`Failed to fetch backend info: ${response.statusText}`);
        }

        const data: BackendInfo = await response.json();
        setIsSqlBackend(data.is_sql_backend);
        setStoreType(data.store_type);
        setError(null);
      } catch (err) {
        console.error('Error fetching backend info:', err);
        setError(err instanceof Error ? err : new Error('Unknown error'));
        // Default to assuming SQL backend on error to avoid blocking UI
        setIsSqlBackend(true);
        setStoreType(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchBackendInfo();
  }, []);

  return { isSqlBackend, storeType, isLoading, error };
}
