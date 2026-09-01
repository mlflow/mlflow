import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render } from '@databricks/web-shared/test-utils/render';

import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';
import { BrowserRouter } from '../../RoutingUtils';
import { SpanLogLevel } from '../../ModelTrace.types';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { VirtualizedTraceTree } from './VirtualizedTraceTree';

jest.mock('../../../../../common/components/ag-grid/AgGridLoader', () => ({
  MLFlowAgGridLoader: ({ rowData, columnDefs, context, getRowHeight }: any) => {
    const React = jest.requireActual<any>('react');
    // AG Grid can retain its initial grid-level context when row IDs survive a data update.
    // Keep it frozen here so state required by cell renderers must arrive through fresh row data.
    const initialContext = React.useRef(context);
    const CellRenderer = columnDefs[0].cellRendererFramework;
    return (
      <div data-testid="mock-ag-grid">
        {rowData.map((row: any) => (
          <div key={String(row.id)} data-testid={`grid-row-${row.id}`} data-row-height={getRowHeight({ data: row })}>
            <CellRenderer data={row} context={initialContext.current} />
          </div>
        ))}
      </div>
    );
  },
}));

const createMockNode = (overrides: Partial<ModelTraceSpanNode> & { key: string | number }): ModelTraceSpanNode =>
  ({
    title: String(overrides.key),
    start: 0,
    end: 1000,
    attributes: {},
    assessments: [],
    traceId: 'tr-1',
    ...overrides,
  }) as ModelTraceSpanNode;

const buildTestTree = (): ModelTraceSpanNode[] => [
  createMockNode({
    key: 'root',
    title: 'Root',
    children: [
      createMockNode({
        key: 'child-1',
        title: 'Child 1',
        children: [createMockNode({ key: 'grandchild-1', title: 'Grandchild 1' })],
      }),
      createMockNode({ key: 'child-2', title: 'Child 2' }),
    ],
  }),
];

const DEFAULT_FILTER_STATE: SpanFilterState = {
  showParents: true,
  showExceptions: true,
  spanTypeDisplayState: {},
  minLogLevel: SpanLogLevel.DEBUG,
};

const queryClient = new QueryClient();
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>
    <QueryClientProvider client={queryClient}>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ModelTraceExplorerPreferencesProvider>{children}</ModelTraceExplorerPreferencesProvider>
        </DesignSystemProvider>
      </IntlProvider>
    </QueryClientProvider>
  </BrowserRouter>
);

describe('VirtualizedTraceTree v2', () => {
  const onSelect = jest.fn();
  const setSpanFilterState = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('flattens all expanded rows and hides the timeline toggle', () => {
    render(
      <VirtualizedTraceTree
        rootNodes={buildTestTree()}
        onSelect={onSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={setSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('timeline-tree-node-root')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-child-1')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-grandchild-1')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Show execution timeline' })).not.toBeInTheDocument();
  });

  it('can expand a node again after collapsing its descendants', async () => {
    render(
      <VirtualizedTraceTree
        rootNodes={buildTestTree()}
        onSelect={onSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={setSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    await userEvent.click(screen.getByTestId('toggle-span-expanded-child-1'));

    expect(screen.queryByTestId('timeline-tree-node-grandchild-1')).not.toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-child-2')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('toggle-span-expanded-child-1'));

    expect(screen.getByTestId('timeline-tree-node-grandchild-1')).toBeInTheDocument();
  });

  it('uses the v2 row renderer and its metadata-aware row height', () => {
    const rootNodes = buildTestTree();
    rootNodes[0].tokenUsage = { total_tokens: 42 };

    render(
      <VirtualizedTraceTree
        rootNodes={rootNodes}
        selectedNode={rootNodes[0]}
        onSelect={onSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={setSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('span-metric-row-root')).toHaveTextContent('42');
    expect(screen.getByTestId('grid-row-root')).toHaveAttribute('data-row-height', '48');
  });

  it('selects rows and resets expansion when search results replace the roots', async () => {
    const { rerender } = render(
      <Wrapper>
        <VirtualizedTraceTree
          rootNodes={buildTestTree()}
          onSelect={onSelect}
          spanFilterState={DEFAULT_FILTER_STATE}
          setSpanFilterState={setSpanFilterState}
        />
      </Wrapper>,
    );

    await userEvent.click(screen.getByTestId('timeline-tree-node-child-2'));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ key: 'child-2' }));

    const newRoots = [
      createMockNode({
        key: 'new-root',
        children: [createMockNode({ key: 'new-child' })],
      }),
    ];
    rerender(
      <Wrapper>
        <VirtualizedTraceTree
          rootNodes={newRoots}
          onSelect={onSelect}
          spanFilterState={DEFAULT_FILTER_STATE}
          setSpanFilterState={setSpanFilterState}
        />
      </Wrapper>,
    );

    await waitFor(() => expect(screen.getByTestId('timeline-tree-node-new-child')).toBeInTheDocument());
    expect(screen.queryByTestId('timeline-tree-node-grandchild-1')).not.toBeInTheDocument();
  });
});
