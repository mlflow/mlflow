import { describe, test, expect, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

import { CustomViewAssistantConnectorProvider, useCustomViewAssistantConnector } from './CustomViewAssistantConnector';

describe('useCustomViewAssistantConnector', () => {
  test('returns an empty connector (no-op defaults) outside any provider', () => {
    const { result } = renderHook(() => useCustomViewAssistantConnector());
    expect(result.current).toEqual({});
    expect(result.current.openAssistant).toBeUndefined();
    expect(result.current.isStreaming).toBeUndefined();
  });

  test('exposes the injected connector inside CustomViewAssistantConnectorProvider', () => {
    const openAssistant = jest.fn();
    const connector = { openAssistant, isStreaming: true };
    const wrapper = ({ children }: { children: ReactNode }) => (
      <CustomViewAssistantConnectorProvider connector={connector}>{children}</CustomViewAssistantConnectorProvider>
    );

    const { result } = renderHook(() => useCustomViewAssistantConnector(), { wrapper });

    expect(result.current).toBe(connector);
    expect(result.current.isStreaming).toBe(true);
    result.current.openAssistant?.('build a view');
    expect(openAssistant).toHaveBeenCalledWith('build a view');
  });

  test('a nested provider overrides the outer connector for its subtree', () => {
    const outer = { openAssistant: jest.fn(), isStreaming: false };
    const inner = { openAssistant: jest.fn(), isStreaming: true };
    const wrapper = ({ children }: { children: ReactNode }) => (
      <CustomViewAssistantConnectorProvider connector={outer}>
        <CustomViewAssistantConnectorProvider connector={inner}>{children}</CustomViewAssistantConnectorProvider>
      </CustomViewAssistantConnectorProvider>
    );

    const { result } = renderHook(() => useCustomViewAssistantConnector(), { wrapper });

    expect(result.current).toBe(inner);
  });
});
