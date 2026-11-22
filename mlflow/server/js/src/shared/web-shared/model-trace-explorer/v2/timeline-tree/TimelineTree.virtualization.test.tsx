import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';
import { BrowserRouter } from '../../RoutingUtils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { TimelineTree } from './TimelineTree';
import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';

jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: ({ count, enabled }: { count: number; enabled: boolean }) => ({
      getVirtualItems: () =>
        enabled
          ? [
              { index: 0, start: 0 },
              { index: count - 1, start: (count - 1) * 48 },
            ]
          : [],
      getTotalSize: () => count * 48,
    }),
  };
});

const nodes: ModelTraceSpanNode[] = Array.from({ length: 120 }, (_, index) => ({
  key: `span-${index}`,
  title: `span ${index}`,
  start: index,
  end: index + 1,
  traceId: 'trace',
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
});
