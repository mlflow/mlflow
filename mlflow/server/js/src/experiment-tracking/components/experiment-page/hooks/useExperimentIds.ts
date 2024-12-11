import qs from 'qs';
import { useMemo } from 'react';
import { useParams, useLocation } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';

export type UseExperimentIdsResult = string[];

/**
 * Hook that returns requested experiment IDs basing on the URL.
 * It extracts ids basing on either route match (in case of a single experiment)
 * or query params (in case of comparing experiments.).
 *
 * @returns array of strings with experiment IDs
 */

export const useExperimentIds = (): UseExperimentIdsResult => {
  const params = useParams<{ experimentId?: string }>();
  const location = useLocation();

  const normalizedLocationSearch = useMemo(() => decodeURIComponent(location.search), [location.search]);

  /**
   * Memoized string containing experiment IDs for comparison ("?experiments=...")
   */
  const compareExperimentIdsQueryParam = useMemo(() => {
    const queryParams = qs.parse(normalizedLocationSearch.substring(1));
    if (queryParams['experiments']) {
      const experimentIdsRaw = queryParams['experiments'];
      return experimentIdsRaw?.toString() || '';
    }

    return '';
  }, [normalizedLocationSearch]);

  return useMemo(() => {
    // Case #1: single experiment
    if (params?.experimentId) {
      return [params?.experimentId];
    }

    // Case #2: multiple (compare) experiments
    if (compareExperimentIdsQueryParam) {
      try {
        return JSON.parse(compareExperimentIdsQueryParam);
      } catch {
        // Apparently URL is malformed
        Utils.logErrorAndNotifyUser(`Could not parse experiment query parameter ${compareExperimentIdsQueryParam}`);
        return '';
      }
    }

    return [];
  }, [compareExperimentIdsQueryParam, params?.experimentId]);
};
