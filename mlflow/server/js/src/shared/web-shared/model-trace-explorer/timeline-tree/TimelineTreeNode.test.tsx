import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TimelineTreeNode } from './TimelineTreeNode';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import { parseModelTraceToTree } from '../ModelTraceExplorer.utils';

const TEST_NODE = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;

const TestWrapper = () => {
  const [selectedKey, setSelectedKey] = useState<string | number>(TEST_NODE.key);
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(new Set([]));

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TimelineTreeNode
          node={TEST_NODE}
          selectedKey={selectedKey}
          expandedKeys={expandedKeys}
          setExpandedKeys={setExpandedKeys}
          traceStartTime={0}
          traceEndTime={0}
          onSelect={(node) => setSelectedKey(node.key)}
          linesToRender={[]}
        />
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('TimelineTreeNode', () => {
  it('should expand when the expand button is clicked', async () => {
    render(<TestWrapper />);

    expect(screen.getByTestId(`timeline-tree-node-${TEST_NODE.key}`)).toBeInTheDocument();
    expect(screen.getAllByTestId(/timeline-tree-node/).length).toBe(1);

    const parentExpandButton = screen.getByTestId(`toggle-span-expanded-${TEST_NODE.key}`);
    await userEvent.click(parentExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/).length).toBe(2);

    const childExpandButton = screen.getByTestId(`toggle-span-expanded-${TEST_NODE.children?.[0]?.key}`);
    await userEvent.click(childExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/).length).toBe(3);

    await userEvent.click(parentExpandButton);
    expect(screen.getAllByTestId(/timeline-tree-node/).length).toBe(1);
  });
});
