import { useCallback, useMemo } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export const COMPARE_MODE_QUERY_PARAM = 'compare';
export const COMPARE_RUNS_QUERY_PARAM = 'compareRuns';
export const COMPARE_FROM_QUERY_PARAM = 'compareFrom';

/**
 * Query-param-powered state for the eval-runs compare surface on the single-run detail page.
 *
 * `compare=true` is an explicit mode flag so comparison can remain open after its last run is
 * removed, decoupled from `compareRuns` (the comma-joined UUIDs being compared
 * against the path's baseline run). Turning the mode off retains `compareRuns` so the selection is
 * restored when compare mode is enabled again.
 */
export const useEvalRunsCompareMode = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const compareMode = searchParams.get(COMPARE_MODE_QUERY_PARAM) === 'true';
  const fromRunsList = searchParams.get(COMPARE_FROM_QUERY_PARAM) === 'runs';
  const compareRunsRaw = searchParams.get(COMPARE_RUNS_QUERY_PARAM) ?? '';

  const compareRuns = useMemo(
    () =>
      compareRunsRaw
        .split(',')
        .map((uuid) => uuid.trim())
        .filter(Boolean),
    [compareRunsRaw],
  );

  const setCompareMode = useCallback(
    (nextCompareMode: boolean) => {
      setSearchParams((params) => {
        if (nextCompareMode) {
          params.set(COMPARE_MODE_QUERY_PARAM, 'true');
        } else {
          params.delete(COMPARE_MODE_QUERY_PARAM);
        }
        return params;
      });
    },
    [setSearchParams],
  );

  const setCompareRuns = useCallback(
    (runUuids: string[]) => {
      setSearchParams((params) => {
        if (runUuids.length > 0) {
          params.set(COMPARE_RUNS_QUERY_PARAM, runUuids.join(','));
        } else {
          params.delete(COMPARE_RUNS_QUERY_PARAM);
        }
        return params;
      });
    },
    [setSearchParams],
  );

  const startComparison = useCallback(
    (runUuids: string[]) => {
      setSearchParams((params) => {
        if (runUuids.length > 0) {
          params.set(COMPARE_RUNS_QUERY_PARAM, runUuids.join(','));
        } else {
          params.delete(COMPARE_RUNS_QUERY_PARAM);
        }
        params.set(COMPARE_MODE_QUERY_PARAM, 'true');
        params.delete(COMPARE_FROM_QUERY_PARAM);
        return params;
      });
    },
    [setSearchParams],
  );

  return { fromRunsList, compareMode, setCompareMode, compareRuns, setCompareRuns, startComparison };
};
