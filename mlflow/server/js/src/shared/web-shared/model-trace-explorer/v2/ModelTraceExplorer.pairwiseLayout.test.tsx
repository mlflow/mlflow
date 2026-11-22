import { jest, beforeAll, afterAll, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { ModelTraceExplorer } from './ModelTraceExplorer';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';

window.HTMLElement.prototype.scrollIntoView = jest.fn();

let originalResizeObserver: typeof ResizeObserver;
beforeAll(() => {
  originalResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class MockResizeObserver {
    observe = jest.fn();
    unobserve = jest.fn();
    disconnect = jest.fn();
  } as unknown as typeof ResizeObserver;
});

afterAll(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

type ExplorerProps = Partial<ComponentProps<typeof ModelTraceExplorer>>;

const renderExplorer = (props: ExplorerProps = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
            <ModelTraceExplorerContextProvider>
              <ModelTraceExplorer modelTrace={MOCK_TRACE} {...props} />
            </ModelTraceExplorerContextProvider>
          </ModelTraceExplorerPreferencesProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ModelTraceExplorer pairwise review layout', () => {
  it('hides the span tree and offers a toggle to show it', async () => {
    const onShowSpanTreeChange = jest.fn();
    renderExplorer({ showSpanTree: false, onShowSpanTreeChange });

    const toggle = await screen.findByRole('button', { name: 'Show span tree' });
    expect(screen.queryByText('Spans')).not.toBeInTheDocument();

    await userEvent.click(toggle);
    expect(onShowSpanTreeChange).toHaveBeenCalledWith(true);
  });

  it('shows the span tree with a toggle to hide it', async () => {
    const onShowSpanTreeChange = jest.fn();
    renderExplorer({ showSpanTree: true, onShowSpanTreeChange });

    const toggle = await screen.findByRole('button', { name: 'Hide span tree' });
    expect(screen.getByText('Spans')).toBeInTheDocument();

    await userEvent.click(toggle);
    expect(onShowSpanTreeChange).toHaveBeenCalledWith(false);
  });

  it('keeps the default layout without a span tree toggle', async () => {
    renderExplorer();

    expect(await screen.findByText('Spans')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /span tree/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Assess/ })).toBeInTheDocument();
  });

  it('hides assessments when asked to', async () => {
    renderExplorer({ hideAssessments: true });

    expect(await screen.findByText('Spans')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Assess/ })).not.toBeInTheDocument();
  });
});
