import { describe, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useMCPServerDetailViewState, MCPServerDetailViewMode } from './useMCPServerDetailViewState';
import { createMockMCPServerVersion } from '../test-utils';

const v1 = createMockMCPServerVersion({ version: '1' });
const v2 = createMockMCPServerVersion({ version: '2' });
const v3 = createMockMCPServerVersion({ version: '3' });

describe('useMCPServerDetailViewState', () => {
  it('starts in preview mode with no selection', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.PREVIEW);
    expect(result.current.selectedVersion).toBeUndefined();
  });

  it('setSelectedVersion updates the selected version', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    act(() => result.current.setSelectedVersion('1'));
    expect(result.current.selectedVersion).toBe('1');
  });

  it('setPreviewMode clears comparedVersion and sets preview mode', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    act(() => result.current.setCompareMode());
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.COMPARE);

    act(() => result.current.setPreviewMode());
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.PREVIEW);
    expect(result.current.viewState.comparedVersion).toBeUndefined();
  });

  it('setCompareMode auto-selects baseline and compared versions', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2, v3]));
    act(() => result.current.setSelectedVersion('1'));
    act(() => result.current.setCompareMode());

    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.COMPARE);
    expect(result.current.selectedVersion).toBeDefined();
    expect(result.current.viewState.comparedVersion).toBeDefined();
    expect(result.current.selectedVersion).not.toBe(result.current.viewState.comparedVersion);
  });

  it('switchSides swaps baseline and compared versions', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    act(() => result.current.setSelectedVersion('1'));
    act(() => result.current.setCompareMode());

    const before = {
      selected: result.current.selectedVersion,
      compared: result.current.viewState.comparedVersion,
    };

    act(() => result.current.switchSides());

    expect(result.current.selectedVersion).toBe(before.compared);
    expect(result.current.viewState.comparedVersion).toBe(before.selected);
  });

  it('auto-swaps when selecting the same version as compared', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    act(() => result.current.setSelectedVersion('1'));
    act(() => {
      result.current.setCompareMode();
    });

    const comparedBefore = result.current.viewState.comparedVersion;
    act(() => result.current.setSelectedVersion(comparedBefore!));

    expect(result.current.selectedVersion).toBe(comparedBefore);
    expect(result.current.viewState.comparedVersion).not.toBe(comparedBefore);
  });

  it('auto-swaps when setting compared to the same version as selected', () => {
    const { result } = renderHook(() => useMCPServerDetailViewState([v1, v2]));
    act(() => result.current.setSelectedVersion('1'));
    act(() => result.current.setCompareMode());

    const selectedBefore = result.current.selectedVersion!;
    act(() => result.current.setComparedVersion(selectedBefore));

    expect(result.current.viewState.comparedVersion).toBe(selectedBefore);
    expect(result.current.selectedVersion).not.toBe(selectedBefore);
  });
});
