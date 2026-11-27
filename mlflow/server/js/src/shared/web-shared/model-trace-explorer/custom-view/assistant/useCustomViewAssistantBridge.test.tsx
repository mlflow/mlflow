import { describe, test, expect, jest, afterEach } from '@jest/globals';
import { renderHook, act, cleanup } from '@testing-library/react';
import type { ReactNode } from 'react';

import type { AgentTraceData } from '../agent/buildAgentPrompt';
import type { CustomView } from '../customViewDefinition';
import { CustomViewAssistantConnectorProvider } from './CustomViewAssistantConnector';
import { getCustomViewAuthoringContext } from './customViewAuthoringContext';
import {
  CustomViewValidationError,
  getCustomViewSpecApplier,
  type RenderCustomViewSpec,
} from './customViewSpecApplier';
import { useCustomViewAssistantBridge } from './useCustomViewAssistantBridge';

const traceData = (): AgentTraceData => ({ metrics: { status: 'OK' } });

const activeView = (overrides: Partial<CustomView> = {}): CustomView => ({
  id: 'view-1',
  name: 'My view',
  label: 'My view',
  instruction: 'show text',
  template: [],
  createdAtMs: 1,
  ...overrides,
});

const wrapperWithConnector = (connector: {
  openAssistant?: (prompt?: string) => void;
  isStreaming?: boolean;
  isPending?: boolean;
}) =>
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <CustomViewAssistantConnectorProvider connector={connector}>{children}</CustomViewAssistantConnectorProvider>
    );
  };

// A no-op onSpec, correctly typed, for tests that don't assert on it being called.
const noopOnSpec = (_spec: RenderCustomViewSpec): Promise<void> => Promise.resolve();

afterEach(() => {
  cleanup();
});

describe('useCustomViewAssistantBridge', () => {
  test('publishes the authoring context (trace sample + template + applyTarget) while enabled', () => {
    const view = activeView({ template: [{ version: 'v0.9' } as any] });
    const { unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: view, onSpec: noopOnSpec }),
    );

    const context = getCustomViewAuthoringContext();
    expect(context).not.toBeNull();
    expect(context?.currentTemplate).toBe(view.template);
    expect(context?.applyTarget).toMatchObject({ id: 'view-1', name: 'My view' });
    expect(context?.traceSample).toMatchObject({ metrics: { status: 'OK' } });

    unmount();
    // Unmounting clears the registration so a stale context isn't read by a later turn.
    expect(getCustomViewAuthoringContext()).toBeNull();
  });

  test('does not publish context or register an applier when disabled', () => {
    const { unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec: noopOnSpec, enabled: false }),
    );

    expect(getCustomViewAuthoringContext()).toBeNull();
    expect(getCustomViewSpecApplier()).toBeNull();

    unmount();
  });

  test('omits applyTarget/currentTemplate when there is no active view yet', () => {
    const { unmount } = renderHook(() => useCustomViewAssistantBridge({ data: traceData(), onSpec: noopOnSpec }));

    const context = getCustomViewAuthoringContext();
    expect(context?.currentTemplate).toBeUndefined();
    expect(context?.applyTarget).toBeUndefined();

    unmount();
  });

  test('registers an applier that calls onSpec and reports ok on success', async () => {
    const onSpec = jest.fn(async (_spec: RenderCustomViewSpec) => {});
    const { unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec }),
    );

    const applier = getCustomViewSpecApplier();
    expect(applier).not.toBeNull();

    const spec: RenderCustomViewSpec = { title: 'Trace Summary', messages: [] };
    const result = await applier?.(spec);

    expect(onSpec).toHaveBeenCalledWith(spec);
    expect(result).toEqual({ ok: true });

    unmount();
  });

  test('a failing onSpec surfaces a structured error and sets applyError', async () => {
    const onSpec = jest.fn(async () => {
      throw new Error('invalid template');
    });
    const { result, unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec }),
    );

    const applier = getCustomViewSpecApplier();
    let applyResult;
    await act(async () => {
      applyResult = await applier?.({ title: 't', messages: [] });
    });

    expect(applyResult).toEqual({ ok: false, error: 'invalid template', retryable: false });
    expect(result.current.applyError).toBe('invalid template');

    unmount();
  });

  test('marks validation failures as retryable', async () => {
    const onSpec = jest.fn(async () => {
      throw new CustomViewValidationError('unknown component');
    });
    const { unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec }),
    );

    let applyResult;
    await act(async () => {
      applyResult = await getCustomViewSpecApplier()?.({ title: 't', messages: [] });
    });

    expect(applyResult).toEqual({ ok: false, error: 'unknown component', retryable: true });
    unmount();
  });

  test('clearApplyError resets a leftover error', async () => {
    const onSpec = jest.fn(async () => {
      throw new Error('boom');
    });
    const { result, unmount } = renderHook(() =>
      useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec }),
    );

    const applier = getCustomViewSpecApplier();
    await act(async () => {
      await applier?.({ title: 't', messages: [] });
    });
    expect(result.current.applyError).toBe('boom');

    act(() => {
      result.current.clearApplyError();
    });
    expect(result.current.applyError).toBeUndefined();

    unmount();
  });

  test('exposes the connector openAssistant/isStreaming/isPending when enabled', () => {
    const openAssistant = jest.fn();
    const wrapper = wrapperWithConnector({ openAssistant, isStreaming: true, isPending: true });
    const { result, unmount } = renderHook(
      () => useCustomViewAssistantBridge({ data: traceData(), activeView: activeView(), onSpec: noopOnSpec }),
      { wrapper },
    );

    expect(result.current.isAvailable).toBe(true);
    expect(result.current.isStreaming).toBe(true);
    expect(result.current.isPending).toBe(true);
    result.current.openAssistant?.('hi');
    expect(openAssistant).toHaveBeenCalledWith('hi');

    unmount();
  });

  test('isAvailable is false and connector state/actions are suppressed when disabled', () => {
    const openAssistant = jest.fn();
    const wrapper = wrapperWithConnector({ openAssistant, isStreaming: true, isPending: true });
    const { result, unmount } = renderHook(
      () =>
        useCustomViewAssistantBridge({
          data: traceData(),
          activeView: activeView(),
          onSpec: noopOnSpec,
          enabled: false,
        }),
      { wrapper },
    );

    expect(result.current.isAvailable).toBe(false);
    expect(result.current.openAssistant).toBeUndefined();
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.isPending).toBe(false);

    unmount();
  });
});
