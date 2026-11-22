import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';
import { BrowserRouter } from '../../RoutingUtils';
import { MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerViewStateProvider } from '../ModelTraceExplorerViewStateContext';
import { TimelineTree } from './TimelineTree';
import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';

jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: ({
      count,
      enabled,
      estimateSize,
      rangeExtractor,
    }: {
      count: number;
      enabled: boolean;
      estimateSize: (index: number) => number;
      rangeExtractor: (range: { startIndex: number; endIndex: number; overscan: number; count: number }) => number[];
    }) => {
      const indexes = enabled ? rangeExtractor({ startIndex: 0, endIndex: 0, overscan: 0, count }) : [];
      return {
        getVirtualItems: () => indexes.map((index) => ({ index, start: index * estimateSize(index) })),
        getTotalSize: () => count * estimateSize(0),
        measureElement: jest.fn(),
      };
    },
  };
});

const nodes: ModelTraceSpanNode[] = Array.from({ length: 120 }, (_, index) => ({
  key: `span-${index}`,
  title: `span ${index}`,
  start: index,
  end: index + 1,
  traceId: 'trace',
  assessments: [],
}));

const TestTree = () => {
  const [expandedKeys, setExpandedKeys] = useState(new Set<string | number>());
  const [spanFilterState, setSpanFilterState] = useState(TEST_SPAN_FILTER_STATE);
  const [queryClient] = useState(() => new QueryClient());
  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <ModelTraceExplorerPreferencesProvider>
              <ModelTraceExplorerViewStateProvider modelTrace={MOCK_TRACE} assessmentsPaneEnabled={false}>
                <TimelineTree
                  rootNodes={nodes}
                  selectedNode={nodes.at(-1)}
                  setSelectedNode={() => {}}
                  traceStartTime={0}
                  traceEndTime={120}
                  expandedKeys={expandedKeys}
                  setExpandedKeys={setExpandedKeys}
                  spanFilterState={spanFilterState}
                  setSpanFilterState={setSpanFilterState}
                />
              </ModelTraceExplorerViewStateProvider>
            </ModelTraceExplorerPreferencesProvider>
          </DesignSystemProvider>
        </IntlProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('TimelineTree virtualization', () => {
  test('mounts viewport rows plus the far selected row', () => {
    render(<TestTree />);

    expect(screen.getAllByTestId(/^timeline-tree-node-/)).toHaveLength(2);
    expect(screen.getByTestId('timeline-tree-node-span-0')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-span-119')).toBeInTheDocument();
  });

  test('preserves the visible row when toggling the Gantt view', async () => {
    render(<TestTree />);
    const scrollElement = screen.getByTestId('timeline-tree-node-span-0').parentElement?.parentElement?.parentElement;
    if (!scrollElement) throw new Error('Timeline scroll element not found');
    expect(scrollElement).toBeInstanceOf(HTMLElement);

    scrollElement.scrollTop = 480;
    await userEvent.click(screen.getByRole('button', { name: 'Show execution timeline' }));
    expect(scrollElement.scrollTop).toBe(320);

    await userEvent.click(screen.getByRole('button', { name: 'Hide execution timeline' }));
    expect(scrollElement.scrollTop).toBe(480);
  });
});
