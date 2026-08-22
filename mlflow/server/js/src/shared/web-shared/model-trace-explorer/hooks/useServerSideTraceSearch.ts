import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ModelTrace, ModelTraceInfoV3, ModelTraceSpanNode } from '../ModelTrace.types';
import { getExperimentTraceV3 } from '../api';
import { isV3ModelTraceInfo, parseModelTraceToTreeWithMultipleRoots } from '../ModelTraceExplorer.utils';

const DEBOUNCE_MS = 400;

export const useServerSideTraceSearch = ({
  modelTraceInfo,
  enabled,
}: {
  modelTraceInfo: ModelTrace['info'];
  enabled: boolean;
}): {
  searchFilter: string;
  setSearchFilter: (filter: string) => void;
  filteredTreeNodes: ModelTraceSpanNode[] | null;
  isSearching: boolean;
} => {
  const [searchFilter, setSearchFilter] = useState('');
  const [filteredTreeNodes, setFilteredTreeNodes] = useState<ModelTraceSpanNode[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const abortRef = useRef(0);

  const traceId = useMemo(() => {
    if (isV3ModelTraceInfo(modelTraceInfo)) {
      return (modelTraceInfo as ModelTraceInfoV3).trace_id;
    }
    return undefined;
  }, [modelTraceInfo]);

  const executeSearch = useCallback(
    async (filter: string) => {
      if (!traceId || !enabled) return;
      const requestId = ++abortRef.current;

      if (!filter.trim()) {
        setFilteredTreeNodes(null);
        setIsSearching(false);
        return;
      }

      setIsSearching(true);
      try {
        const resp = await getExperimentTraceV3({
          traceId,
          filter: filter.trim(),
        });
        if (requestId !== abortRef.current) return;

        const spans = resp?.trace?.spans ?? [];
        const pseudoTrace: ModelTrace = {
          info: modelTraceInfo,
          data: { spans },
        };
        const nodes = parseModelTraceToTreeWithMultipleRoots(pseudoTrace);
        setFilteredTreeNodes(nodes);
      } catch {
        if (requestId === abortRef.current) {
          setFilteredTreeNodes(null);
        }
      } finally {
        if (requestId === abortRef.current) {
          setIsSearching(false);
        }
      }
    },
    [traceId, enabled, modelTraceInfo],
  );

  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => executeSearch(searchFilter), DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchFilter, executeSearch, enabled]);

  return {
    searchFilter,
    setSearchFilter,
    filteredTreeNodes,
    isSearching,
  };
};
