import { describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { ModelTraceExplorerRightPaneTabs } from './ModelTraceExplorerRightPaneTabs';
import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import { getDefaultActiveTab } from '../ModelTraceExplorer.utils';

let mockComparisonNode: ModelTraceSpanNode | undefined;
jest.mock('../ModelTraceExplorerViewStateContext', () => ({
  useModelTraceExplorerViewState: () => ({ hideAssessments: true, selectedComparisonNode: mockComparisonNode }),
}));
jest.mock('./ModelTraceExplorerRightPaneHeader', () => ({ ModelTraceExplorerRightPaneHeader: () => null }));
jest.mock('./ModelTraceExplorerContentTab', () => ({ ModelTraceExplorerContentTab: () => null }));

const createNode = (): ModelTraceSpanNode => ({
  key: 'span',
  title: 'span',
  children: [],
  attributes: {},
  start: 0,
  end: 1,
  type: ModelSpanType.CHAIN,
  assessments: [],
  traceId: 'trace',
});

describe('content-tab availability', () => {
  it.each(['selected', 'comparison'])('does not materialize unrelated %s payloads', (side) => {
    const node = createNode();
    node.inputs = { question: 'synthetic café 🌳' };
    node.events = [{ name: 'exception', timestamp: 0, attributes: {} }];
    const reads = Object.fromEntries(
      ['attributes', 'outputs', 'chatMessages', 'chatTools'].map((key) => [key, jest.fn(() => undefined)]),
    );
    Object.defineProperties(
      node,
      Object.fromEntries(Object.entries(reads).map(([key, get]) => [key, { get, enumerable: true }])),
    );
    mockComparisonNode = side === 'comparison' ? node : undefined;
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ModelTraceExplorerRightPaneTabs
            activeSpan={side === 'selected' ? node : createNode()}
            searchFilter=""
            activeMatch={null}
            activeTab="content"
            setActiveTab={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByRole('tab', { name: 'Inputs / Outputs' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Events/ })).toBeInTheDocument();
    expect(getDefaultActiveTab(node)).toBe('events');
    Object.values(reads).forEach((get) => expect(get).not.toHaveBeenCalled());
  });
});
