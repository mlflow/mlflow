/**
 * Generates metric key for node level system metrics based on node ID and optional GPU index.
 */
export const createNodeLevelMetricKey = (nodeId: string | number, metricType?: string, gpuIndex?: number) => {
  if (gpuIndex !== undefined) {
    return `system/node_${nodeId}/gpu_${gpuIndex}_${metricType}`;
  }
  return `system/node_${nodeId}/${metricType}`;
};
