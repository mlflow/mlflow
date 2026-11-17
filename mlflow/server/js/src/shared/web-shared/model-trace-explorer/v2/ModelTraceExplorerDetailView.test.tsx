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
  GraphViewWorkflowCanvas: ({ selectedNodeId }: { selectedNodeId: string | null }) => (
    <div data-testid="workflow-canvas" data-selected-node-id={selectedNodeId} />
  ),
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

  it('keeps a tree selection linked through showing, hiding and reopening the graph', async () => {
    renderExplorer();
    await userEvent.click(await screen.findByTestId('timeline-tree-node-_generate_response'));
    expect(screen.getByText('_generate_response-input')).toBeInTheDocument();
    expect(computeWorkflowLayout).not.toHaveBeenCalled();

    const toggleGraph = async () => {
      await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
      await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Show graph' }));
      await userEvent.keyboard('{Escape}');
    };
    await toggleGraph();
    const layout = jest.mocked(computeWorkflowLayout).mock.results[0].value;
    const node = layout.nodes.find((workflowNode) =>
      workflowNode.spans.some((span) => span.key === '_generate_response'),
    );
    expect(node).toBeDefined();
    expect(await screen.findByTestId('workflow-canvas')).toHaveAttribute('data-selected-node-id', node?.id);

    await toggleGraph();
    expect(screen.queryByTestId('workflow-canvas')).not.toBeInTheDocument();
    expect(screen.getByText('_generate_response-input')).toBeInTheDocument();
    expect(computeWorkflowLayout).toHaveBeenCalledTimes(1);

    await toggleGraph();
    expect(await screen.findByTestId('workflow-canvas')).toHaveAttribute('data-selected-node-id', node?.id);
    expect(screen.getByText('_generate_response-input')).toBeInTheDocument();
  });
});
