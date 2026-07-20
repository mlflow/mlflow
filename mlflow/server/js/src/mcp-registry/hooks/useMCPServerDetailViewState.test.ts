import { describe, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { useState } from 'react';
import { useMCPServerDetailViewState, MCPServerDetailViewMode } from './useMCPServerDetailViewState';
import { createMockMCPServerVersion } from '../test-utils';

const v1 = createMockMCPServerVersion({ version: '1' });
const v2 = createMockMCPServerVersion({ version: '2' });
const v3 = createMockMCPServerVersion({ version: '3' });

const renderViewState = (versions: ReturnType<typeof createMockMCPServerVersion>[], initialVersion?: string) =>
  renderHook(
    ({ versions: v, initial }) => {
      const [selectedVersion, setSelectedVersion] = useState<string | undefined>(initial);
      return {
        selectedVersion,
        setSelectedVersion,
        ...useMCPServerDetailViewState(v, selectedVersion, setSelectedVersion),
      };
    },
    { initialProps: { versions, initial: initialVersion } },
  );

describe('useMCPServerDetailViewState', () => {
  it('starts in preview mode', () => {
    const { result } = renderViewState([v1, v2]);
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.PREVIEW);
  });

  it('auto-selects first version when none is selected', () => {
    const { result } = renderViewState([v1, v2]);
    expect(result.current.selectedVersion).toBe('1');
  });

  it('setPreviewMode clears comparedVersion and sets preview mode', () => {
    const { result } = renderViewState([v1, v2], '1');
    act(() => result.current.setCompareMode());
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.COMPARE);

    act(() => result.current.setPreviewMode());
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.PREVIEW);
    expect(result.current.viewState.comparedVersion).toBeUndefined();
  });

  it('setCompareMode auto-selects baseline and compared versions', () => {
    const { result } = renderViewState([v1, v2, v3], '1');
    act(() => result.current.setCompareMode());

    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.COMPARE);
    expect(result.current.selectedVersion).toBeDefined();
    expect(result.current.viewState.comparedVersion).toBeDefined();
    expect(result.current.selectedVersion).not.toBe(result.current.viewState.comparedVersion);
  });

  it('switchSides swaps baseline and compared versions', () => {
    const { result } = renderViewState([v1, v2], '1');
    act(() => result.current.setCompareMode());

    const before = {
      selected: result.current.selectedVersion,
      compared: result.current.viewState.comparedVersion,
    };

    act(() => result.current.switchSides());

    expect(result.current.selectedVersion).toBe(before.compared);
    expect(result.current.viewState.comparedVersion).toBe(before.selected);
  });

  it('does not rewrite selectedVersion when it is not in the versions list', () => {
    const { result } = renderViewState([v1, v2], 'nonexistent');
    expect(result.current.selectedVersion).toBe('nonexistent');
  });

  it('exits compare mode when fewer than 2 versions remain', () => {
    const { result, rerender } = renderViewState([v1, v2], '1');
    act(() => result.current.setCompareMode());
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.COMPARE);

    rerender({ versions: [v1], initial: result.current.selectedVersion });
    expect(result.current.viewState.mode).toBe(MCPServerDetailViewMode.PREVIEW);
  });
});
