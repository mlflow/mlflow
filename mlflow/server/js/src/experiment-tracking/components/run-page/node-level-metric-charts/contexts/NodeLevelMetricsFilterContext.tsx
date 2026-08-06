import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { Dash } from 'plotly.js';
import { getStableColorForRun as getStableColorForNode } from '../../../../utils/RunNameUtils';
import { RUNS_COLOR_PALETTE } from '../../../../../common/color-palette';

// Upper bound on GPU indices per node, used to space out color assignments and avoid collisions.
const MAX_GPUS_PER_NODE = 16;

type FilterState = {
  selectedNodes: Set<string>;
  selectedGpus: Map<string, Set<number>>;
};

type NodeLevelCustomChartLineStyle = {
  color: string;
  dashStyle: Dash;
};

type ContextType = FilterState & {
  toggleNode: (nodeId: string) => void;
  toggleGpu: (nodeId: string, gpuIndex: number, totalGpuCount: number) => void;
  clear: () => void;
  hasAnySelection: boolean;
  getCustomLineStyle: (metricKey: string) => NodeLevelCustomChartLineStyle | null;
};

const NodeLevelMetricsFilterContext = createContext<ContextType | null>(null);

/**
 * Hook for managing node level metrics filter state.
 * Supports hybrid selection: a node can be fully selected OR have specific GPUs selected (mutually exclusive).
 */
export const useNodeLevelMetricsFilterState = () => {
  const [state, setState] = useState<FilterState>({
    selectedNodes: new Set(),
    selectedGpus: new Map(),
  });

  const toggleNode = useCallback((nodeId: string) => {
    setState(({ selectedNodes, selectedGpus }) => {
      const newNodes = new Set(selectedNodes);
      const newGpus = new Map(selectedGpus);

      if (newNodes.has(nodeId)) {
        newNodes.delete(nodeId);
      } else {
        newGpus.delete(nodeId);
        newNodes.add(nodeId);
      }

      return { selectedNodes: newNodes, selectedGpus: newGpus };
    });
  }, []);

  const toggleGpu = useCallback((nodeId: string, gpuIndex: number, totalGpuCount: number) => {
    setState(({ selectedNodes, selectedGpus }) => {
      const newNodes = new Set(selectedNodes);
      const newGpus = new Map(selectedGpus);

      if (newNodes.has(nodeId)) {
        // Deselect node and select all GPUs except this one
        newNodes.delete(nodeId);
        const gpuSet = new Set(Array.from({ length: totalGpuCount }, (_, i) => i).filter((i) => i !== gpuIndex));
        if (gpuSet.size > 0) newGpus.set(nodeId, gpuSet);
      } else {
        const gpuSet = new Set(newGpus.get(nodeId));

        if (gpuSet.has(gpuIndex)) {
          gpuSet.delete(gpuIndex);
          if (gpuSet.size === 0) {
            newGpus.delete(nodeId);
          } else {
            newGpus.set(nodeId, gpuSet);
          }
        } else {
          gpuSet.add(gpuIndex);
          // If all GPUs selected, convert to full node selection
          if (gpuSet.size === totalGpuCount) {
            newGpus.delete(nodeId);
            newNodes.add(nodeId);
          } else {
            newGpus.set(nodeId, gpuSet);
          }
        }
      }

      return { selectedNodes: newNodes, selectedGpus: newGpus };
    });
  }, []);

  const clear = useCallback(() => {
    setState({ selectedNodes: new Set(), selectedGpus: new Map() });
  }, []);

  const hasAnySelection = state.selectedNodes.size > 0 || state.selectedGpus.size > 0;

  return useMemo(
    () => ({ ...state, toggleNode, toggleGpu, clear, hasAnySelection }),
    [state, toggleNode, toggleGpu, clear, hasAnySelection],
  );
};

export const NodeLevelMetricsFilterContextProvider = ({
  children,
  value,
}: {
  children: React.ReactNode;
  value: Omit<ContextType, 'getCustomLineStyle'>;
}) => {
  const styleCache = useRef(new Map<string, NodeLevelCustomChartLineStyle | null>());

  const getCustomLineStyle = useCallback((metricKey: string): NodeLevelCustomChartLineStyle | null => {
    // Check cache first
    if (styleCache.current.has(metricKey)) {
      return styleCache.current.get(metricKey) as NodeLevelCustomChartLineStyle;
    }

    // Parse SGC metric key: system/node_{id}/... or system/node_{id}/gpu_{index}_...
    const nodeLevelMetricKeyMatch = metricKey.match(/^system\/node_(\d+)(?:\/gpu_(\d+))?/);
    if (!nodeLevelMetricKeyMatch) {
      styleCache.current.set(metricKey, null);
      return null;
    }

    const [, nodeId, gpuIndexStr] = nodeLevelMetricKeyMatch;
    const gpuIndex = gpuIndexStr ? parseInt(gpuIndexStr, 10) : undefined;

    // Use distinct colors for each (node, GPU) combination with solid lines.
    // For non-GPU node metrics: differentiate nodes by color with solid lines.
    const color =
      gpuIndex !== undefined
        ? RUNS_COLOR_PALETTE[(parseInt(nodeId, 10) * MAX_GPUS_PER_NODE + gpuIndex) % RUNS_COLOR_PALETTE.length]
        : getStableColorForNode(nodeId);
    const dashStyle = 'solid' as Dash;

    const style = { color, dashStyle };
    styleCache.current.set(metricKey, style);
    return style;
  }, []);

  const contextValue = useMemo(() => ({ ...value, getCustomLineStyle }), [value, getCustomLineStyle]);

  return (
    <NodeLevelMetricsFilterContext.Provider value={contextValue}>{children}</NodeLevelMetricsFilterContext.Provider>
  );
};

export const useNodeLevelMetricsFilterContext = () => useContext(NodeLevelMetricsFilterContext);
