import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';

import { VirtualizedTraceTree } from './VirtualizedTraceTree';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { SpanLogLevel } from '../ModelTrace.types';

jest.mock('../../../../common/components/ag-grid/AgGridLoader', () => ({
  MLFlowAgGridLoader: ({ rowData, onRowClicked, getRowStyle }: any) => (
    <div data-testid="mock-ag-grid">
      {(rowData || []).map((row: any) => (
        <div
          key={String(row.id)}
          data-testid={`grid-row-${row.id}`}
          data-selected={String(row.isSelected)}
          style={getRowStyle?.({ data: row })}
          onClick={() => onRowClicked?.({ data: row })}
        >
          {row.hasChildren && (
            <button
              data-testid={`toggle-${row.id}`}
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                row.onToggleExpand(row.id);
              }}
            />
          )}
          <span data-testid={`title-${row.id}`}>{row.node.title}</span>
          {row.linesToRender.length > 0 && <span data-testid={`hierarchy-bars-${row.id}`} />}
          {row.node.assessments?.length > 0 && (
            <span data-testid={`assessment-tag-${row.id}`}>{row.node.assessments.length}</span>
          )}
        </div>
      ))}
    </div>
  ),
}));

jest.mock('./TimelineTreeHeader', () => ({
  TimelineTreeHeader: () => <div data-testid="timeline-tree-header" />,
}));

function createMockNode(overrides: Partial<ModelTraceSpanNode> & { key: string | number }): ModelTraceSpanNode {
  return {
    title: String(overrides.key),
    start: 0,
    end: 1000,
    attributes: {},
    assessments: [],
    traceId: 'tr-1',
    ...overrides,
  } as ModelTraceSpanNode;
}

const DEFAULT_FILTER_STATE: SpanFilterState = {
  showParents: true,
  showExceptions: true,
  spanTypeDisplayState: {},
  minLogLevel: SpanLogLevel.DEBUG,
};

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>{children}</DesignSystemProvider>
);

function buildTestTree(): ModelTraceSpanNode[] {
  return [
    createMockNode({
      key: 'root',
      title: 'Root',
      children: [
        createMockNode({
          key: 'child-1',
          title: 'Child 1',
          children: [
            createMockNode({ key: 'grandchild-1a', title: 'Grandchild 1A' }),
            createMockNode({ key: 'grandchild-1b', title: 'Grandchild 1B' }),
          ],
        }),
        createMockNode({
          key: 'child-2',
          title: 'Child 2',
          children: [
            createMockNode({ key: 'grandchild-2a', title: 'Grandchild 2A' }),
            createMockNode({ key: 'grandchild-2b', title: 'Grandchild 2B' }),
          ],
        }),
        createMockNode({
          key: 'child-3',
          title: 'Child 3',
          children: [
            createMockNode({ key: 'grandchild-3a', title: 'Grandchild 3A' }),
            createMockNode({ key: 'grandchild-3b', title: 'Grandchild 3B' }),
          ],
        }),
      ],
    }),
  ];
}

describe('VirtualizedTraceTree', () => {
  const mockOnSelect = jest.fn();
  const mockSetSpanFilterState = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders all root nodes expanded by default', () => {
    render(
      <VirtualizedTraceTree
        rootNodes={buildTestTree()}
        onSelect={mockOnSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={mockSetSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('title-root')).toHaveTextContent('Root');
    expect(screen.getByTestId('title-child-1')).toHaveTextContent('Child 1');
    expect(screen.getByTestId('title-child-2')).toHaveTextContent('Child 2');
    expect(screen.getByTestId('title-child-3')).toHaveTextContent('Child 3');
    expect(screen.getByTestId('title-grandchild-1a')).toHaveTextContent('Grandchild 1A');
    expect(screen.getByTestId('title-grandchild-1b')).toHaveTextContent('Grandchild 1B');
    expect(screen.getByTestId('title-grandchild-2a')).toHaveTextContent('Grandchild 2A');
    expect(screen.getByTestId('title-grandchild-2b')).toHaveTextContent('Grandchild 2B');
    expect(screen.getByTestId('title-grandchild-3a')).toHaveTextContent('Grandchild 3A');
    expect(screen.getByTestId('title-grandchild-3b')).toHaveTextContent('Grandchild 3B');
  });

  it('can expand a node again after collapsing its children', async () => {
    render(
      <VirtualizedTraceTree
        rootNodes={buildTestTree()}
        onSelect={mockOnSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={mockSetSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    await userEvent.click(screen.getByTestId('toggle-child-1'));

    expect(screen.queryByTestId('title-grandchild-1a')).not.toBeInTheDocument();
    expect(screen.queryByTestId('title-grandchild-1b')).not.toBeInTheDocument();

    expect(screen.getByTestId('title-child-2')).toBeInTheDocument();
    expect(screen.getByTestId('title-grandchild-2a')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('toggle-child-1'));

    expect(screen.getByTestId('title-grandchild-1a')).toBeInTheDocument();
    expect(screen.getByTestId('title-grandchild-1b')).toBeInTheDocument();
  });

  it('selection highlights the correct row', () => {
    const rootNodes = buildTestTree();
    const selectedNode = rootNodes[0].children![1];

    render(
      <VirtualizedTraceTree
        rootNodes={rootNodes}
        selectedNode={selectedNode}
        onSelect={mockOnSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={mockSetSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('grid-row-child-2')).toHaveAttribute('data-selected', 'true');
    expect(screen.getByTestId('grid-row-child-1')).toHaveAttribute('data-selected', 'false');
  });

  it('renders hierarchy bars for nested rows', () => {
    render(
      <VirtualizedTraceTree
        rootNodes={buildTestTree()}
        onSelect={mockOnSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={mockSetSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('hierarchy-bars-grandchild-1a')).toBeInTheDocument();
    expect(screen.getByTestId('hierarchy-bars-grandchild-2b')).toBeInTheDocument();
    expect(screen.queryByTestId('hierarchy-bars-root')).not.toBeInTheDocument();
  });

  it('shows assessment tag when node has assessments', () => {
    const rootNodes = [
      createMockNode({
        key: 'assessed-node',
        title: 'Assessed Node',
        assessments: [
          {
            assessment_id: 'a1',
            assessment_name: 'Relevance',
            trace_id: 'tr-1',
            source: { source_type: 'HUMAN' as const, source_id: 'user1' },
            create_time: '2025-01-01',
            last_update_time: '2025-01-01',
            feedback: { value: 'up' },
          },
        ] as any,
      }),
    ];

    render(
      <VirtualizedTraceTree
        rootNodes={rootNodes}
        onSelect={mockOnSelect}
        spanFilterState={DEFAULT_FILTER_STATE}
        setSpanFilterState={mockSetSpanFilterState}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.getByTestId('assessment-tag-assessed-node')).toHaveTextContent('1');
  });

  it('resets expanded keys when rootNodes change', () => {
    const initialNodes = buildTestTree();

    const { rerender } = render(
      <Wrapper>
        <VirtualizedTraceTree
          rootNodes={initialNodes}
          onSelect={mockOnSelect}
          spanFilterState={DEFAULT_FILTER_STATE}
          setSpanFilterState={mockSetSpanFilterState}
        />
      </Wrapper>,
    );

    expect(screen.getByTestId('title-grandchild-1a')).toBeInTheDocument();

    const newNodes = [
      createMockNode({
        key: 'new-root',
        title: 'New Root',
        children: [createMockNode({ key: 'new-child', title: 'New Child' })],
      }),
    ];

    rerender(
      <Wrapper>
        <VirtualizedTraceTree
          rootNodes={newNodes}
          onSelect={mockOnSelect}
          spanFilterState={DEFAULT_FILTER_STATE}
          setSpanFilterState={mockSetSpanFilterState}
        />
      </Wrapper>,
    );

    expect(screen.getByTestId('title-new-root')).toBeInTheDocument();
    expect(screen.getByTestId('title-new-child')).toBeInTheDocument();
    expect(screen.queryByTestId('title-grandchild-1a')).not.toBeInTheDocument();
  });
});
