import { describe, test, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useCategorizedNodeLevelMetricKeys } from './useCategorizedNodeLevelMetricKeys';

describe('useCategorizedNodeLevelMetricKeys', () => {
  test('handles single node with GPU metrics', () => {
    const metricKeys = [
      'system/node_0/cpu_utilization_percentage',
      'system/node_0/system_memory_usage_percentage',
      'system/node_0/gpu_0_utilization_percentage',
      'system/node_0/gpu_0_power_usage_watts',
      'system/node_0/gpu_0_memory_usage_megabytes',
    ];
    const { result } = renderHook(() => useCategorizedNodeLevelMetricKeys(metricKeys));

    expect(result.current.nodeIndexes).toEqual(['0']);
    expect(result.current.commonMetrics).toEqual(['cpu_utilization_percentage', 'system_memory_usage_percentage']);
    expect(result.current.gpuIndexes).toEqual([0]);
    expect(result.current.commonGpuMetrics).toEqual([
      'utilization_percentage',
      'power_usage_watts',
      'memory_usage_megabytes',
    ]);
  });

  test('handles full node level metrics from multiple nodes with multiple GPUs', () => {
    const metricKeys = [
      // Node 0
      'system/node_0/cpu_utilization_percentage',
      'system/node_0/system_memory_usage_percentage',
      'system/node_0/system_memory_usage_megabytes',
      'system/node_0/network_transmit_megabytes',
      'system/node_0/network_receive_megabytes',
      'system/node_0/gpu_0_utilization_percentage',
      'system/node_0/gpu_0_power_usage_watts',
      'system/node_0/gpu_0_power_usage_percentage',
      'system/node_0/gpu_0_memory_usage_percentage',
      'system/node_0/gpu_0_memory_usage_megabytes',
      'system/node_0/disk_usage_percentage',
      'system/node_0/disk_usage_megabytes',
      'system/node_0/disk_available_megabytes',
      // Node 1
      'system/node_1/cpu_utilization_percentage',
      'system/node_1/system_memory_usage_percentage',
      'system/node_1/system_memory_usage_megabytes',
      'system/node_1/network_transmit_megabytes',
      'system/node_1/network_receive_megabytes',
      'system/node_1/gpu_0_utilization_percentage',
      'system/node_1/gpu_0_power_usage_watts',
      'system/node_1/gpu_0_power_usage_percentage',
      'system/node_1/gpu_0_memory_usage_percentage',
      'system/node_1/gpu_0_memory_usage_megabytes',
      'system/node_1/disk_usage_percentage',
      'system/node_1/disk_usage_megabytes',
      'system/node_1/disk_available_megabytes',
    ];
    const { result } = renderHook(() => useCategorizedNodeLevelMetricKeys(metricKeys));

    expect(result.current.nodeIndexes).toEqual(['0', '1']);
    expect(result.current.commonMetrics).toEqual([
      'cpu_utilization_percentage',
      'system_memory_usage_percentage',
      'system_memory_usage_megabytes',
      'network_transmit_megabytes',
      'network_receive_megabytes',
      'disk_usage_percentage',
      'disk_usage_megabytes',
      'disk_available_megabytes',
    ]);
    expect(result.current.gpuIndexes).toEqual([0]);
    expect(result.current.commonGpuMetrics).toEqual([
      'utilization_percentage',
      'power_usage_watts',
      'power_usage_percentage',
      'memory_usage_percentage',
      'memory_usage_megabytes',
    ]);
  });

  test('handles mixed non-GPU and GPU metrics across multiple nodes', () => {
    const metricKeys = [
      'system/node_0/cpu_utilization_percentage',
      'system/node_0/gpu_0_utilization_percentage',
      'system/node_0/gpu_1_utilization_percentage',
      'system/node_0/disk_usage_megabytes',
      'system/node_1/cpu_utilization_percentage',
      'system/node_1/gpu_0_utilization_percentage',
      'system/node_1/gpu_1_utilization_percentage',
      'system/node_1/disk_usage_megabytes',
      'system/node_2/cpu_utilization_percentage',
      'system/node_2/gpu_0_utilization_percentage',
      'system/node_2/gpu_1_utilization_percentage',
      'system/node_2/disk_usage_megabytes',
    ];
    const { result } = renderHook(() => useCategorizedNodeLevelMetricKeys(metricKeys));

    expect(result.current.nodeIndexes).toEqual(['0', '1', '2']);
    expect(result.current.commonMetrics).toEqual(['cpu_utilization_percentage', 'disk_usage_megabytes']);
    expect(result.current.gpuIndexes).toEqual([0, 1]);
    expect(result.current.commonGpuMetrics).toEqual(['utilization_percentage']);
  });
});
