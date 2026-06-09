import { useMemo } from 'react';

/**
 * Extracts and categorizes node level metric keys into node indexes, common metrics, GPU indexes, and common GPU metrics.
 * Example: for a given set of metric keys:
 * [
 * 'system/node_0/cpu_utilization_percentage',
 * 'system/node_0/system_memory_usage_percentage',
 * 'system/node_0/gpu_0_utilization_percentage',
 * 'system/node_0/gpu_0_power_usage_watts',
 * 'system/node_1/cpu_utilization_percentage',
 * 'system/node_1/system_memory_usage_percentage',
 * 'system/node_1/gpu_0_utilization_percentage',
 * 'system/node_1/gpu_0_power_usage_watts',
 * ]
 * It will return:
 * {
 *   nodeIndexes: ['0', '1'],
 *   commonMetrics: ['cpu_utilization_percentage', 'system_memory_usage_percentage'],
 *   gpuIndexes: [0],
 *   commonGpuMetrics: ['utilization_percentage', 'power_usage_watts'],
 * }
 */
export const useCategorizedNodeLevelMetricKeys = (metricKeys: string[], enabled = true) => {
  return useMemo(() => {
    if (!enabled) {
      return {
        nodeIndexes: [],
        commonMetrics: [],
        gpuIndexes: [],
        commonGpuMetrics: [],
        enabled,
      };
    }
    const nodeMetrics: Record<number, string[]> = {};
    const gpuMetrics: Record<number, string[]> = {};
    const gpuIndexes = new Set<number>();

    for (const key of metricKeys) {
      const nodeMatch = key.match(/node_(\d+)/);
      const metricMatch = key.match(/node_\d+\/(.+)/);
      if (!nodeMatch || !metricMatch) continue;

      const node = Number(nodeMatch[1]);
      const metric = metricMatch[1];

      // Track metric per node
      if (!nodeMetrics[node]) nodeMetrics[node] = [];
      nodeMetrics[node].push(metric);

      // Track GPU metrics (if any)
      const gpuMatch = metric.match(/gpu_(\d+)_(.+)/);
      if (gpuMatch) {
        const gpuIndex = Number(gpuMatch[1]);
        const gpuMetric = gpuMatch[2];
        gpuIndexes.add(gpuIndex);
        if (!gpuMetrics[gpuIndex]) gpuMetrics[gpuIndex] = [];
        gpuMetrics[gpuIndex].push(gpuMetric);
      }
    }

    const nodeIndexes = Object.keys(nodeMetrics)
      .map(Number)
      .sort((a, b) => a - b);

    // Common metrics across all nodes (excluding GPU-specific metrics)
    const commonMetrics =
      nodeIndexes.length > 0
        ? [
            ...new Set(
              nodeMetrics[nodeIndexes[0]]
                .filter((metric) => nodeIndexes.every((node) => nodeMetrics[node].includes(metric)))
                .filter((metric) => !metric.startsWith('gpu_')),
            ),
          ]
        : [];

    // Common GPU metrics (e.g., "utilization_percentage", "power_usage_watts")
    const gpuIndexList = [...gpuIndexes].sort((a, b) => a - b);
    const commonGpuMetrics =
      gpuIndexList.length > 0
        ? [
            ...new Set(
              gpuMetrics[gpuIndexList[0]].filter((metric) =>
                gpuIndexList.every((gpu) => gpuMetrics[gpu]?.includes(metric)),
              ),
            ),
          ]
        : [];

    return {
      nodeIndexes: nodeIndexes.map(String),
      commonMetrics,
      gpuIndexes: gpuIndexList,
      commonGpuMetrics,
      enabled,
    };
  }, [metricKeys, enabled]);
};
