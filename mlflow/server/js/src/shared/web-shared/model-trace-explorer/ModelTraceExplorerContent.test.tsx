import { jest, beforeAll, afterAll, beforeEach, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { useAssistant, useRegisterAssistantContext } from '@mlflow/mlflow/src/assistant';

import { QueryClient, QueryClientProvider } from '../query-client/queryClient';
import type { ModelTrace } from './ModelTrace.types';
import { ModelTraceExplorer } from './ModelTraceExplorer';
import { MOCK_V3_TRACE } from './ModelTraceExplorer.test-utils';
import { getModelTraceId } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  useAssistant: jest.fn(),
  useRegisterAssistantContext: jest.fn(),
}));

jest.mock('./FeatureUtils', () => ({
  ...jest.requireActual<typeof import('./FeatureUtils')>('./FeatureUtils'),
  shouldEnableTracesTabLabelingSchemas: jest.fn().mockReturnValue(false),
}));
jest.mock('./hooks/useGetModelTraceInfo', () => ({
  useGetModelTraceInfo: jest.fn().mockReturnValue({ refetch: jest.fn() }),
}));

window.HTMLElement.prototype.scrollIntoView = jest.fn();

let originalResizeObserver: typeof ResizeObserver;
beforeAll(() => {
  originalResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class MockResizeObserver {
    observe = () => {};
    unobserve = () => {};
    disconnect = () => {};
  } as unknown as typeof ResizeObserver;
});
afterAll(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

const mockOpenPanel = jest.fn();

const setAssistant = (isLocalServer: boolean) => {
  jest
    .mocked(useAssistant)
    .mockReturnValue({ isLocalServer, openPanel: mockOpenPanel } as unknown as ReturnType<typeof useAssistant>);
};

beforeEach(() => {
  jest.clearAllMocks();
});

const renderExplorer = (modelTrace: ModelTrace = MOCK_V3_TRACE) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
            <ModelTraceExplorer modelTrace={modelTrace} initialActiveView="detail" />
          </ModelTraceExplorerPreferencesProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorerContent — Analyze in Assistant tab', () => {
  it('shows the tab on a local server', async () => {
    setAssistant(true);
    renderExplorer();
    expect(await screen.findByRole('tab', { name: /Analyze in Assistant/ })).toBeInTheDocument();
  });

  it('hides the tab when not on a local server', async () => {
    setAssistant(false);
    renderExplorer();
    // Wait for the explorer to render its tabs, then assert the analyze tab is absent.
    expect(await screen.findByRole('tab', { name: /Details & Timeline/ })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: /Analyze in Assistant/ })).not.toBeInTheDocument();
  });

  it('registers the viewed trace id as assistant context on a local server', () => {
    setAssistant(true);
    renderExplorer();
    expect(useRegisterAssistantContext).toHaveBeenCalledWith('traceId', getModelTraceId(MOCK_V3_TRACE));
  });

  it('does not register a trace id when not on a local server', () => {
    setAssistant(false);
    renderExplorer();
    expect(useRegisterAssistantContext).toHaveBeenCalledWith('traceId', null);
  });

  it('opens the assistant panel without switching the trace content when clicked', async () => {
    setAssistant(true);
    renderExplorer();

    const analyzeTab = await screen.findByRole('tab', { name: /Analyze in Assistant/ });
    await userEvent.click(analyzeTab);

    expect(mockOpenPanel).toHaveBeenCalled();
    // The action must not select the analyze tab or replace the current content — the
    // "Details & Timeline" tab stays selected.
    expect(analyzeTab).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tab', { name: /Details & Timeline/ })).toHaveAttribute('aria-selected', 'true');
  });
});
