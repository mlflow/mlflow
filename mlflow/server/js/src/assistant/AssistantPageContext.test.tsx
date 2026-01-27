import { renderHook, act, cleanup } from '@testing-library/react';
import {
  useRegisterAssistantContext,
  useRegisterSelectedIds,
  useAssistantPageContext,
  useAssistantPageContextActions,
} from './AssistantPageContext';

afterEach(() => {
  cleanup();
  // Reset the zustand store by removing all known keys
  const { removeContext } = renderHook(() => useAssistantPageContextActions()).result.current;
  for (const key of [
    'traceId',
    'sessionId',
    'modelName',
    'modelVersion',
    'promptName',
    'promptVersion',
    'comparedPromptVersion',
    'selectedDatasetId',
    'selectedScorerName',
    'selectedRunIds',
    'selectedTraceIds',
    'selectedSessionIds',
    'selectedModelVersions',
  ]) {
    act(() => removeContext(key as any));
  }
  cleanup();
});

describe('useRegisterAssistantContext', () => {
  it('registers traceId when value is provided and removes it when set to null', () => {
    const { rerender, unmount } = renderHook(
      ({ value }) => {
        useRegisterAssistantContext('traceId', value);
        return useAssistantPageContext();
      },
      { initialProps: { value: 'trace-123' as string | null } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({ traceId: 'trace-123' });

    rerender({ value: null });
    expect(ctx.current).not.toHaveProperty('traceId');

    rerender({ value: 'trace-456' });
    expect(ctx.current).toMatchObject({ traceId: 'trace-456' });

    unmount();
    expect(ctx.current).not.toHaveProperty('traceId');
  });

  it('registers both sessionId and traceId when selecting a trace from a session', () => {
    const { rerender, unmount } = renderHook(
      ({ traceId }: { traceId: string | null }) => {
        useRegisterAssistantContext('sessionId', 'session-abc');
        useRegisterAssistantContext('traceId', traceId);
      },
      { initialProps: { traceId: null as string | null } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());

    expect(ctx.current).toMatchObject({ sessionId: 'session-abc' });
    expect(ctx.current).not.toHaveProperty('traceId');

    rerender({ traceId: 'trace-123' });
    expect(ctx.current).toMatchObject({
      sessionId: 'session-abc',
      traceId: 'trace-123',
    });

    rerender({ traceId: null });
    expect(ctx.current).toMatchObject({ sessionId: 'session-abc' });
    expect(ctx.current).not.toHaveProperty('traceId');

    unmount();
    expect(ctx.current).not.toHaveProperty('sessionId');
    expect(ctx.current).not.toHaveProperty('traceId');
  });

  it('registers model name and version for model version page', () => {
    const { unmount } = renderHook(() => {
      useRegisterAssistantContext('modelName', 'my-model');
      useRegisterAssistantContext('modelVersion', '3');
    });

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({
      modelName: 'my-model',
      modelVersion: '3',
    });

    unmount();
    expect(ctx.current).not.toHaveProperty('modelName');
    expect(ctx.current).not.toHaveProperty('modelVersion');
  });

  it('registers prompt name, version, and compared version', () => {
    const { rerender, unmount } = renderHook(
      ({ comparedVersion }: { comparedVersion: string | null }) => {
        useRegisterAssistantContext('promptName', 'my-prompt');
        useRegisterAssistantContext('promptVersion', '2');
        useRegisterAssistantContext('comparedPromptVersion', comparedVersion);
      },
      { initialProps: { comparedVersion: null as string | null } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({
      promptName: 'my-prompt',
      promptVersion: '2',
    });
    expect(ctx.current).not.toHaveProperty('comparedPromptVersion');

    rerender({ comparedVersion: '1' });
    expect(ctx.current).toMatchObject({
      promptName: 'my-prompt',
      promptVersion: '2',
      comparedPromptVersion: '1',
    });

    unmount();
    expect(ctx.current).not.toHaveProperty('promptName');
    expect(ctx.current).not.toHaveProperty('promptVersion');
    expect(ctx.current).not.toHaveProperty('comparedPromptVersion');
  });

  it('registers selectedDatasetId', () => {
    const { unmount } = renderHook(() => {
      useRegisterAssistantContext('selectedDatasetId', 'dataset-789');
    });

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({ selectedDatasetId: 'dataset-789' });

    unmount();
    expect(ctx.current).not.toHaveProperty('selectedDatasetId');
  });

  it('registers selectedScorerName conditionally', () => {
    const { rerender, unmount } = renderHook(
      ({ isExpanded }: { isExpanded: boolean }) => {
        useRegisterAssistantContext('selectedScorerName', isExpanded ? 'relevance-scorer' : undefined);
      },
      { initialProps: { isExpanded: false } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).not.toHaveProperty('selectedScorerName');

    rerender({ isExpanded: true });
    expect(ctx.current).toMatchObject({
      selectedScorerName: 'relevance-scorer',
    });

    rerender({ isExpanded: false });
    expect(ctx.current).not.toHaveProperty('selectedScorerName');

    unmount();
  });
});

describe('useRegisterSelectedIds', () => {
  it('registers selected run IDs from row selection state', () => {
    const noSelection: Record<string, boolean> = {};
    const withSelection = {
      'run-1': true,
      'run-2': true,
      'run-3': false,
    };

    const { rerender, unmount } = renderHook(
      ({ rowSelection }) => {
        useRegisterSelectedIds('selectedRunIds', rowSelection);
      },
      { initialProps: { rowSelection: noSelection } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).not.toHaveProperty('selectedRunIds');

    rerender({ rowSelection: withSelection });
    expect(ctx.current).toMatchObject({
      selectedRunIds: expect.arrayContaining(['run-1', 'run-2']),
    });
    const runIds = ctx.current as Record<string, unknown>;
    expect(runIds.selectedRunIds).toHaveLength(2);

    rerender({ rowSelection: noSelection });
    expect(ctx.current).not.toHaveProperty('selectedRunIds');

    unmount();
  });

  it('registers selected trace IDs from row selection state', () => {
    const selection = { 'trace-a': true, 'trace-b': true };

    const { unmount } = renderHook(
      ({ rowSelection }) => {
        useRegisterSelectedIds('selectedTraceIds', rowSelection);
      },
      { initialProps: { rowSelection: selection } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({
      selectedTraceIds: expect.arrayContaining(['trace-a', 'trace-b']),
    });

    unmount();
    expect(ctx.current).not.toHaveProperty('selectedTraceIds');
  });

  it('registers selected session IDs from row selection state', () => {
    const selection = { 'session-1': true, 'session-2': true };

    const { unmount } = renderHook(
      ({ rowSelection }) => {
        useRegisterSelectedIds('selectedSessionIds', rowSelection);
      },
      { initialProps: { rowSelection: selection } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({
      selectedSessionIds: expect.arrayContaining(['session-1', 'session-2']),
    });

    unmount();
    expect(ctx.current).not.toHaveProperty('selectedSessionIds');
  });

  it('registers selected model versions from row selection state', () => {
    const selection = { '1': true, '3': true };

    const { unmount } = renderHook(
      ({ rowSelection }) => {
        useRegisterSelectedIds('selectedModelVersions', rowSelection);
      },
      { initialProps: { rowSelection: selection } },
    );

    const { result: ctx } = renderHook(() => useAssistantPageContext());
    expect(ctx.current).toMatchObject({
      selectedModelVersions: expect.arrayContaining(['1', '3']),
    });

    unmount();
    expect(ctx.current).not.toHaveProperty('selectedModelVersions');
  });
});
