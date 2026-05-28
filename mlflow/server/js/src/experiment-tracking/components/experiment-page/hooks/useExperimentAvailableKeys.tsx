import { useEffect, useMemo, useRef, useState } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';
import { ViewType } from '../../../sdk/MlflowEnums';

/**
 * Hook that fetches the union of metric/param/tag keys present in an experiment, used by
 * the column picker to know which columns are available to add.
 *
 * Lazy by design: only fetches when `enabled` becomes true (i.e. when the picker opens),
 * caches the result per experiment-id set, and never refetches once it has data.
 *
 * Strategy: a single `search_runs(max_results=1)` call without any column-aware filters.
 * One run's key set is treated as a representative sample. For RL/multi-environment
 * evaluation workloads (where every run logs the same key set), this is exact. For
 * hyperparameter sweeps where different runs log different subsets, the picker may miss
 * keys that are only present on other runs — a follow-up `GetExperimentKeys` endpoint
 * (see #23253 discussion) would solve that properly. This frontend-only approach trades
 * completeness for not requiring a backend change.
 *
 * The fetch is intentionally separate from the runs-list fetch so the runs-list can stay
 * column-aware (small payload) while the picker still gets the full key set.
 */
export const useExperimentAvailableKeys = (experimentIds: string[], enabled: boolean) => {
  const [metricKeys, setMetricKeys] = useState<string[]>([]);
  const [paramKeys, setParamKeys] = useState<string[]>([]);
  const [tagKeys, setTagKeys] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const fetchedFor = useRef<string | null>(null);

  // Cache key: sorted experiment-id set. Refetches if the user navigates to a different
  // experiment without unmounting the picker.
  const cacheKey = useMemo(() => [...experimentIds].sort().join('|'), [experimentIds]);

  useEffect(() => {
    if (!enabled || !experimentIds.length || fetchedFor.current === cacheKey) {
      return;
    }

    fetchedFor.current = cacheKey;
    setIsLoading(true);

    MlflowService.searchRuns({
      experiment_ids: experimentIds,
      max_results: 1,
      run_view_type: ViewType.ALL,
    })
      .then((response: any) => {
        const sampleRun = response?.runs?.[0];
        if (sampleRun?.data) {
          setMetricKeys((sampleRun.data.metrics ?? []).map((m: any) => m.key));
          setParamKeys((sampleRun.data.params ?? []).map((p: any) => p.key));
          setTagKeys((sampleRun.data.tags ?? []).map((t: any) => t.key));
        }
      })
      .catch(() => {
        // Picker degrades gracefully if discovery fails — falls back to whatever the
        // runs-list response contained. Don't surface an error to the user.
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [enabled, cacheKey, experimentIds]);

  return { metricKeys, paramKeys, tagKeys, isLoading };
};
