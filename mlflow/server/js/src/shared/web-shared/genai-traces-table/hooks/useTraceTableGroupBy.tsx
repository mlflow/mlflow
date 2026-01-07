import { useMemo } from 'react';

import { useLocalStorage } from '@databricks/web-shared/hooks';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import type { TraceGroupByConfig } from '../types';
import { hasAnySessionTraces } from '../utils/GroupingUtils';

const LOCAL_STORAGE_KEY_PREFIX = 'mlflow.traces.groupBy';
const LOCAL_STORAGE_VERSION = 1;

/**
 * Hook to manage trace table group-by state with localStorage persistence.
 */
export function useTraceTableGroupBy(experimentId: string): {
  groupByConfig: TraceGroupByConfig | null;
  setGroupByConfig: (config: TraceGroupByConfig | null) => void;
} {
  const storageKey = `${LOCAL_STORAGE_KEY_PREFIX}.${experimentId}`;

  const [groupByConfig, setGroupByConfig] = useLocalStorage<TraceGroupByConfig | null>({
    key: storageKey,
    version: LOCAL_STORAGE_VERSION,
    initialValue: null,
  });

  return { groupByConfig, setGroupByConfig };
}

/**
 * Hook to check if any traces have session metadata.
 */
export function useHasSessionTraces(traces: ModelTraceInfoV3[]): boolean {
  return useMemo(() => hasAnySessionTraces(traces), [traces]);
}
