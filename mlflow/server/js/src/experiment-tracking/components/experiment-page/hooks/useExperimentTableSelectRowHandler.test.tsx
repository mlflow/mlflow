import { describe, expect, test, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useExperimentTableSelectRowHandler } from './useExperimentTableSelectRowHandler';

describe('useExperimentTableSelectRowHandler', () => {
  test('closes column selector when at least one run is selected', () => {
    const updateViewState = jest.fn();
    const { result } = renderHook(() => useExperimentTableSelectRowHandler(updateViewState));

    result.current.onSelectionChange({
      api: {
        getSelectedRows: () => [{ runInfo: { runUuid: 'run-1' } }],
      },
    } as any);

    expect(updateViewState).toHaveBeenCalledWith({
      runsSelected: { 'run-1': true },
      columnSelectorVisible: false,
    });
  });

  test('only updates runsSelected when there are no selected rows', () => {
    const updateViewState = jest.fn();
    const { result } = renderHook(() => useExperimentTableSelectRowHandler(updateViewState));

    result.current.onSelectionChange({
      api: {
        getSelectedRows: () => [],
      },
    } as any);

    expect(updateViewState).toHaveBeenCalledWith({
      runsSelected: {},
    });
  });
});
