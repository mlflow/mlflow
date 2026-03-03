import { describe, it, expect } from '@jest/globals';
import { usePromptDetailsPageViewState } from './usePromptDetailsPageViewState';
import { PromptVersionsTableMode } from '../utils';
import type { RegisteredPromptDetailsResponse, RegisteredPromptVersion } from '../types';
import { act, renderHook, type RenderHookOptions } from '@testing-library/react';
import { testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';

const renderHookWithRouter = <TResult, TProps>(
  hook: (props: TProps) => TResult,
  options?: RenderHookOptions<TProps> & { initialEntries?: string[] },
) => {
  const { initialEntries = ['/'], ...restOptions } = options ?? {};
  return renderHook(hook, {
    ...restOptions,
    wrapper: ({ children }) => <TestRouter initialEntries={initialEntries} routes={[testRoute(<>{children}</>)]} />,
  });
};

describe('usePromptDetailsPageViewState', () => {
  const mockPromptDetailsData: RegisteredPromptDetailsResponse = {
    versions: [{ version: '1' }, { version: '2' }] as RegisteredPromptVersion[],
  };

  it('should initialize with preview mode', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
  });

  it('should set preview mode with selected version', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setPreviewMode({ version: '1' });
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
    expect(result.current.selectedVersion).toBe('1');
  });

  it('should set compare mode with selected and compared versions', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setCompareMode();
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.COMPARE);
    expect(result.current.selectedVersion).toBe('1');
    expect(result.current.viewState.comparedVersion).toBe('2');
  });

  it('should switch sides', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    act(() => {
      result.current.setCompareMode();
    });
    act(() => {
      result.current.switchSides();
    });
    expect(result.current.selectedVersion).toBe('2');
    expect(result.current.viewState.comparedVersion).toBe('1');
  });

  it('should set selected version', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    act(() => {
      result.current.setSelectedVersion('3');
    });
    expect(result.current.selectedVersion).toBe('3');
  });

  it('should set compared version', () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    act(() => {
      result.current.setComparedVersion('4');
    });
    expect(result.current.viewState.comparedVersion).toBe('4');
  });
});
