import { renderHook } from '@testing-library/react';
import { useRegisterAssistantContext, useAssistantPageContext } from './AssistantPageContext';

describe('useRegisterAssistantContext', () => {
  it('registers traceId when value is provided and removes it when set to null', () => {
    // Register a traceId
    const { rerender, unmount } = renderHook(
      ({ key, value }) => {
        useRegisterAssistantContext(key, value);
        return useAssistantPageContext();
      },
      { initialProps: { key: 'traceId' as const, value: 'trace-123' as string | null } },
    );

    // Verify traceId is set
    const { result: contextResult } = renderHook(() => useAssistantPageContext());
    expect(contextResult.current).toMatchObject({ traceId: 'trace-123' });

    // Update to null (simulates closing the trace drawer)
    rerender({ key: 'traceId' as const, value: null });
    expect(contextResult.current).not.toHaveProperty('traceId');

    // Re-register with a different traceId
    rerender({ key: 'traceId' as const, value: 'trace-456' });
    expect(contextResult.current).toMatchObject({ traceId: 'trace-456' });

    // Unmount should clean up
    unmount();
    expect(contextResult.current).not.toHaveProperty('traceId');
  });

  it('registers both sessionId and traceId when selecting a trace from a session', () => {
    // Simulate the chat session page: sessionId is always registered,
    // traceId is registered when a trace is selected in the session view
    const { rerender, unmount } = renderHook(
      ({ traceId }: { traceId: string | null }) => {
        useRegisterAssistantContext('sessionId', 'session-abc');
        useRegisterAssistantContext('traceId', traceId);
        return useAssistantPageContext();
      },
      { initialProps: { traceId: null } },
    );

    const { result: contextResult } = renderHook(() => useAssistantPageContext());

    // Initially only sessionId is in context (no trace selected)
    expect(contextResult.current).toMatchObject({ sessionId: 'session-abc' });
    expect(contextResult.current).not.toHaveProperty('traceId');

    // Select a trace within the session
    rerender({ traceId: 'trace-123' });
    expect(contextResult.current).toMatchObject({
      sessionId: 'session-abc',
      traceId: 'trace-123',
    });

    // Close the trace drawer — sessionId stays, traceId removed
    rerender({ traceId: null });
    expect(contextResult.current).toMatchObject({ sessionId: 'session-abc' });
    expect(contextResult.current).not.toHaveProperty('traceId');

    // Unmount cleans up both
    unmount();
    expect(contextResult.current).not.toHaveProperty('sessionId');
    expect(contextResult.current).not.toHaveProperty('traceId');
  });
});
