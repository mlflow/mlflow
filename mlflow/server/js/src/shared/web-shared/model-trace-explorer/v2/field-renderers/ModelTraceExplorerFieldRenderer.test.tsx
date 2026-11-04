import { describe, expect, it } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { ModelTraceExplorerFieldRenderer } from './ModelTraceExplorerFieldRenderer';

const ProvidersWrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerFieldRenderer', () => {
  it('preserves unchanged lines between separate changes in a large JSON value', () => {
    const unchangedLines = Array.from({ length: 350 }, (_, index) => `line-${index}`);
    const baseline = JSON.stringify({
      langgraph_checkpoint_ns: 'agent:baseline',
      langgraph_node: 'agent',
      langgraph_path: ['__pregel_pull', 'agent'],
      langgraph_step: 1,
      langgraph_triggers: ['branch:to:agent'],
      ls_integration: 'langgraph',
      thread_id: 'baseline-thread',
      unchanged_lines: unchangedLines,
    });
    const candidate = JSON.stringify({
      langgraph_checkpoint_ns: 'agent:candidate',
      langgraph_node: 'agent',
      langgraph_path: ['__pregel_pull', 'agent'],
      langgraph_step: 1,
      langgraph_triggers: ['branch:to:agent'],
      ls_integration: 'langgraph',
      thread_id: 'candidate-thread',
      unchanged_lines: unchangedLines,
    });

    render(
      <ModelTraceExplorerFieldRenderer
        title="metadata"
        data={baseline}
        comparisonData={candidate}
        comparisonEnabled
        comparisonRole="baseline"
        renderMode="json"
      />,
      { wrapper: ProvidersWrapper },
    );

    expect(screen.getAllByText('"langgraph_node": "agent",')).toHaveLength(1);
  });
});
