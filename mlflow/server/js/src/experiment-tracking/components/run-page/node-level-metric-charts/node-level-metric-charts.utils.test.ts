import { describe, expect, it } from '@jest/globals';
import { createNodeLevelMetricKey } from './node-level-metric-charts.utils';

describe('node-level-metric-charts.utils', () => {
  it('should create a node level metric key', () => {
    expect(createNodeLevelMetricKey('123', 'cpu_utilization')).toBe('system/node_123/cpu_utilization');
  });

  it('should create a node level metric key with a gpu index', () => {
    expect(createNodeLevelMetricKey('123', 'utilization', 1)).toBe('system/node_123/gpu_1_utilization');
  });
});
