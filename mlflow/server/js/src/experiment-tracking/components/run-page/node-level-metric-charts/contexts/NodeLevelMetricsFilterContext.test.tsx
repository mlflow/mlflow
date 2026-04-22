import { renderHook, act } from '@testing-library/react';
import { useNodeLevelMetricsFilterState } from './NodeLevelMetricsFilterContext';
import { describe, test, expect } from '@jest/globals';

describe('useNodeLevelMetricsFilterState', () => {
  test('should initialize with empty state', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    expect(result.current.selectedNodes.size).toBe(0);
    expect(result.current.selectedGpus.size).toBe(0);
    expect(result.current.hasAnySelection).toBe(false);
  });

  test('should toggle node selection on', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleNode('node-1');
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(true);
    expect(result.current.hasAnySelection).toBe(true);
  });

  test('should toggle node selection off', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleNode('node-1');
      result.current.toggleNode('node-1');
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(false);
    expect(result.current.hasAnySelection).toBe(false);
  });

  test('should toggle GPU selection on', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleGpu('node-1', 0, 4);
    });

    expect(result.current.selectedGpus.get('node-1')?.has(0)).toBe(true);
    expect(result.current.hasAnySelection).toBe(true);
  });

  test('should toggle GPU selection off', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleGpu('node-1', 0, 4);
      result.current.toggleGpu('node-1', 0, 4);
    });

    expect(result.current.selectedGpus.has('node-1')).toBe(false);
    expect(result.current.hasAnySelection).toBe(false);
  });

  test('should convert to node selection when all GPUs are selected', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleGpu('node-1', 0, 2);
      result.current.toggleGpu('node-1', 1, 2);
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(true);
    expect(result.current.selectedGpus.has('node-1')).toBe(false);
  });

  test('should replace node selection with partial GPU selection when toggling GPU on fully selected node', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleNode('node-1');
      result.current.toggleGpu('node-1', 1, 4);
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(false);
    expect(result.current.selectedGpus.get('node-1')?.size).toBe(3);
    expect(result.current.selectedGpus.get('node-1')?.has(0)).toBe(true);
    expect(result.current.selectedGpus.get('node-1')?.has(1)).toBe(false);
    expect(result.current.selectedGpus.get('node-1')?.has(2)).toBe(true);
    expect(result.current.selectedGpus.get('node-1')?.has(3)).toBe(true);
  });

  test('should clear all selections', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleNode('node-1');
      result.current.toggleNode('node-2');
      result.current.toggleGpu('node-3', 0, 4);
      result.current.clear();
    });

    expect(result.current.selectedNodes.size).toBe(0);
    expect(result.current.selectedGpus.size).toBe(0);
    expect(result.current.hasAnySelection).toBe(false);
  });

  test('should replace GPU selection with node selection when toggling node', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleGpu('node-1', 0, 4);
      result.current.toggleGpu('node-1', 1, 4);
      result.current.toggleNode('node-1');
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(true);
    expect(result.current.selectedGpus.has('node-1')).toBe(false);
  });

  test('should handle multiple nodes with mixed selection states', () => {
    const { result } = renderHook(() => useNodeLevelMetricsFilterState());

    act(() => {
      result.current.toggleNode('node-1');
      result.current.toggleGpu('node-2', 0, 4);
      result.current.toggleGpu('node-2', 2, 4);
    });

    expect(result.current.selectedNodes.has('node-1')).toBe(true);
    expect(result.current.selectedGpus.get('node-2')?.has(0)).toBe(true);
    expect(result.current.selectedGpus.get('node-2')?.has(2)).toBe(true);
    expect(result.current.hasAnySelection).toBe(true);
  });
});
