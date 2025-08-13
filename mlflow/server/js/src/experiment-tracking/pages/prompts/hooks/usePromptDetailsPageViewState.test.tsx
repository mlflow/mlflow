import { usePromptDetailsPageViewState } from './usePromptDetailsPageViewState';
import { PromptVersionsTableMode } from '../utils';
import type { RegisteredPromptDetailsResponse, RegisteredPromptVersion } from '../types';
import { act, renderHook } from '@testing-library/react';

describe('usePromptDetailsPageViewState', () => {
  const mockPromptDetailsData: RegisteredPromptDetailsResponse = {
    versions: [{ version: '1' }, { version: '2' }] as RegisteredPromptVersion[],
  };

  it('should initialize with preview mode', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState());
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
  });

  it('should set table mode', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState());
    act(() => {
      result.current.setTableMode();
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.TABLE);
  });

  it('should set preview mode with selected version', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setPreviewMode({ version: '1' });
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
    expect(result.current.viewState.selectedVersion).toBe('1');
  });

  it('should set compare mode with selected and compared versions', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setCompareMode();
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.COMPARE);
    expect(result.current.viewState.selectedVersion).toBe('2');
    expect(result.current.viewState.comparedVersion).toBe('1');
  });

  it('should switch sides', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setCompareMode();
    });
    act(() => {
      result.current.switchSides();
    });
    expect(result.current.viewState.selectedVersion).toBe('1');
    expect(result.current.viewState.comparedVersion).toBe('2');
  });

  it('should set selected version', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState());
    act(() => {
      result.current.setSelectedVersion('3');
    });
    expect(result.current.viewState.selectedVersion).toBe('3');
  });

  it('should set compared version', () => {
    const { result } = renderHook(() => usePromptDetailsPageViewState());
    act(() => {
      result.current.setComparedVersion('4');
    });
    expect(result.current.viewState.comparedVersion).toBe('4');
  });
});
