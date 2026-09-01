import { describe, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TimelineTreeNode } from './TimelineTreeNode';
import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import {
  ModelTraceExplorerPreferencesContext,
  ModelTraceExplorerPreferencesProvider,
  type TimelineTreeMetric,
  useModelTraceExplorerPreferences,
} from '../ModelTraceExplorerPreferencesContext';
import { MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';
import { parseModelTraceToTree } from '../ModelTraceExplorer.utils';
import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';
import { BrowserRouter } from '../../RoutingUtils';

const TEST_NODE = parseModelTraceToTree(MOCK_TRACE);
if (!TEST_NODE) {
  throw new Error('Expected MOCK_TRACE to contain one root span');
}

const METRICS_NODE: ModelTraceSpanNode = {
  key: 'span-with-metrics',
  title: 'Generate answer',
  type: ModelSpanType.LLM,
  start: 0,
  end: 3_200_000,
  inputs: {},
  outputs: {},
  attributes: {},
  assessments: [],
  traceId: 'trace-1',
  tokenUsage: { total_tokens: 1980 },
};

const MetricPreferencesOverride = ({
  children,
  metrics,
}: {
  children: React.ReactNode;
  metrics: TimelineTreeMetric[];
}) => {
  const preferences = useModelTraceExplorerPreferences();
  return (
    <ModelTraceExplorerPreferencesContext.Provider value={{ ...preferences, timelineTreeMetrics: metrics }}>
      {children}
    </ModelTraceExplorerPreferencesContext.Provider>
  );
};

const TestWrapper = ({ node = TEST_NODE, metrics }: { node?: ModelTraceSpanNode; metrics?: TimelineTreeMetric[] }) => {
  const [selectedKey, setSelectedKey] = useState<string | number>(node.key);
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(new Set([]));
  const [queryClient] = useState(() => new QueryClient());

  const timelineTreeNode = (
    <TimelineTreeNode
      node={node}
      selectedKey={selectedKey}
      expandedKeys={expandedKeys}
      setExpandedKeys={setExpandedKeys}
      traceStartTime={0}
      traceEndTime={0}
      onSelect={(selectedNode) => setSelectedKey(selectedNode.key)}
      linesToRender={[]}
    />
  );

  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <ModelTraceExplorerPreferencesProvider>
              {metrics ? (
                <MetricPreferencesOverride metrics={metrics}>{timelineTreeNode}</MetricPreferencesOverride>
              ) : (
                timelineTreeNode
              )}
            </ModelTraceExplorerPreferencesProvider>
          </DesignSystemProvider>
        </IntlProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('TimelineTreeNode', () => {
  it('renders only the supported default metrics', () => {
    render(<TestWrapper node={METRICS_NODE} />);

    const titleRow = screen.getByTestId(`span-title-row-${METRICS_NODE.key}`);
    const metricRow = screen.getByTestId(`span-metric-row-${METRICS_NODE.key}`);
    const textBlock = screen.getByTestId(`span-text-block-${METRICS_NODE.key}`);
    expect(titleRow).toContainElement(screen.getByText('Generate answer'));
    expect(titleRow).not.toContainElement(screen.getByText('3.20s'));
    expect(metricRow).toContainElement(screen.getByText('3.20s'));
    expect(metricRow).toContainElement(screen.getByText('2K'));
    expect(textBlock).toContainElement(titleRow);
    expect(textBlock).toContainElement(metricRow);
    expect(screen.getByTestId(`span-icon-${METRICS_NODE.key}`)).toBeInTheDocument();
    expect(screen.queryByText('Completed')).not.toBeInTheDocument();
  });

  it('places a single selected metric at the right end of the title row', () => {
    render(<TestWrapper node={METRICS_NODE} metrics={['duration']} />);

    const titleRow = screen.getByTestId(`span-title-row-${METRICS_NODE.key}`);
    const inlineMetric = screen.getByTestId(`span-inline-metric-${METRICS_NODE.key}`);
    const metric = screen.getByText('3.20s');
    expect(titleRow).toContainElement(metric);
    expect(inlineMetric).toContainElement(metric);
    expect(screen.queryByTestId(`span-metric-row-${METRICS_NODE.key}`)).not.toBeInTheDocument();
  });

  it('uses a metric row when multiple metrics are selected but only one is available', () => {
    const nodeWithoutTokens = { ...METRICS_NODE, tokenUsage: undefined };
    render(<TestWrapper node={nodeWithoutTokens} metrics={['duration', 'tokens']} />);

    const titleRow = screen.getByTestId(`span-title-row-${METRICS_NODE.key}`);
    const metricRow = screen.getByTestId(`span-metric-row-${METRICS_NODE.key}`);
    expect(titleRow).not.toContainElement(screen.getByText('3.20s'));
    expect(metricRow).toContainElement(screen.getByText('3.20s'));
    expect(screen.queryByText('2K')).not.toBeInTheDocument();
  });

  it('stops the active connector below the selected span', async () => {
    render(<TestWrapper />);

    const rootConnector = screen.getByTestId('icon-bottom-connector');
    expect(rootConnector).toHaveAttribute('data-active', 'false');
    expect(rootConnector).toHaveAttribute('data-row-center', '24');

    await userEvent.click(screen.getByTestId(`toggle-span-expanded-${TEST_NODE.key}`));
    expect(screen.getByTestId('icon-left-connector')).toHaveAttribute('data-row-center', '24');
    await userEvent.click(screen.getByTestId(`timeline-tree-node-${TEST_NODE.children?.[0]?.key}`));

    expect(screen.getByTestId('icon-bottom-connector')).toHaveAttribute('data-active', 'true');
  });

  it('should expand when the expand button is clicked', async () => {
    render(<TestWrapper />);

    // Timeline hierarchy rows and expand controls currently expose only test IDs.
    expect(screen.getByTestId(`timeline-tree-node-${TEST_NODE.key}`)).toBeInTheDocument();
    expect(screen.getAllByTestId(/timeline-tree-node/)).toHaveLength(1);

    const parentExpandButton = screen.getByTestId(`toggle-span-expanded-${TEST_NODE.key}`);
    await userEvent.click(parentExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/)).toHaveLength(2);

    const childExpandButton = screen.getByTestId(`toggle-span-expanded-${TEST_NODE.children?.[0]?.key}`);
    await userEvent.click(childExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/)).toHaveLength(3);

    await userEvent.click(parentExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/)).toHaveLength(1);
  });
});
