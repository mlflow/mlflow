import { describe, it, expect } from '@jest/globals';
import { usePromptDetailsPageViewState } from './usePromptDetailsPageViewState';
import { PromptVersionsTableMode } from '../utils';
import type { RegisteredPromptDetailsResponse, RegisteredPromptVersion } from '../types';
import { act, renderHook, type RenderHookOptions } from '@testing-library/react';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';

const { history } = setupTestRouter();

const renderHookWithRouter = <TResult, TProps>(
  hook: (props: TProps) => TResult,
  options?: RenderHookOptions<TProps> & { initialEntries?: string[] },
) => {
  const { initialEntries = ['/'], ...restOptions } = options ?? {};
  return renderHook(hook, {
    ...restOptions,
    wrapper: ({ children }) => (
      <TestRouter initialEntries={initialEntries} history={history} routes={[testRoute(<>{children}</>)]} />
    ),
  });
};

describe('usePromptDetailsPageViewState', () => {
  const mockPromptDetailsData: RegisteredPromptDetailsResponse = {
    versions: [{ version: '1' }, { version: '2' }] as RegisteredPromptVersion[],
  };

  it('should initialize with preview mode', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    await waitForRoutesToBeRendered();
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
  });

  it('should set preview mode with selected version', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    await waitForRoutesToBeRendered();
    act(() => {
      result.current.setPreviewMode({ version: '1' });
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.PREVIEW);
    expect(result.current.selectedVersion).toBe('1');
  });

  it('should set compare mode with selected and compared versions', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    await waitForRoutesToBeRendered();
    act(() => {
      result.current.setCompareMode();
    });
    expect(result.current.viewState.mode).toBe(PromptVersionsTableMode.COMPARE);
    expect(result.current.selectedVersion).toBe('1');
    expect(result.current.viewState.comparedVersion).toBe('2');
  });

  it('should switch sides', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState(mockPromptDetailsData));
    await waitForRoutesToBeRendered();
    act(() => {
      result.current.setCompareMode();
    });
    act(() => {
      result.current.switchSides();
    });
    expect(result.current.selectedVersion).toBe('2');
    expect(result.current.viewState.comparedVersion).toBe('1');
  });

  it('should set selected version', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    await waitForRoutesToBeRendered();
    act(() => {
      result.current.setSelectedVersion('3');
    });
    expect(result.current.selectedVersion).toBe('3');
  });

  it('should set compared version', async () => {
    const { result } = renderHookWithRouter(() => usePromptDetailsPageViewState());
    await waitForRoutesToBeRendered();
    act(() => {
      result.current.setComparedVersion('4');
    });
    expect(result.current.viewState.comparedVersion).toBe('4');
  });
});
