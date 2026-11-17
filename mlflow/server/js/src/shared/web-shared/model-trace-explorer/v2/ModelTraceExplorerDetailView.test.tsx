import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { computeWorkflowLayout } from '../graph-view/GraphView.workflow';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import { ModelTraceExplorer } from './ModelTraceExplorer';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerContextProvider } from './ModelTraceExplorerContext';

jest.mock('../graph-view/GraphView.workflow', () => {
  const workflow = jest.requireActual<typeof import('../graph-view/GraphView.workflow')>(
    '../graph-view/GraphView.workflow',
  );
  return { ...workflow, computeWorkflowLayout: jest.fn(workflow.computeWorkflowLayout) };
});
jest.mock('../graph-view/GraphViewWorkflowCanvas', () => ({
  GraphViewWorkflowCanvas: () => <div data-testid="workflow-canvas" />,
}));

const renderExplorer = (enableGraphView = true) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <ModelTraceExplorerPreferencesProvider initialRenderMode="default">
            <ModelTraceExplorerContextProvider>
              <ModelTraceExplorer modelTrace={MOCK_TRACE} enableGraphView={enableGraphView} />
            </ModelTraceExplorerContextProvider>
          </ModelTraceExplorerPreferencesProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('workflow layout work', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('does not lay out a disabled graph', async () => {
    renderExplorer(false);
    expect(await screen.findByText('Spans')).toBeInTheDocument();
    expect(computeWorkflowLayout).not.toHaveBeenCalled();
  });

  it('waits for the graph toggle before computing the real layout', async () => {
    renderExplorer();
    await userEvent.click(await screen.findByRole('button', { name: 'Trace display settings' }));
    expect(computeWorkflowLayout).not.toHaveBeenCalled();
    await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Show graph' }));
    expect(await screen.findByTestId('workflow-canvas')).toBeInTheDocument();
    expect(computeWorkflowLayout).toHaveBeenCalledTimes(1);
  });
});
