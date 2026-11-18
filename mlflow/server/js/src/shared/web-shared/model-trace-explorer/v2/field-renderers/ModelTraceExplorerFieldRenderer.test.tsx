import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { ModelTraceExplorerFieldRenderer } from './ModelTraceExplorerFieldRenderer';
import * as traceUtils from '../ModelTraceExplorer.utils';

const ProvidersWrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('ModelTraceExplorerFieldRenderer', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('retains comparison work across rerenders and updates changed values', () => {
    const data = JSON.stringify({ body: 'before comparison payload' });
    const candidate = JSON.stringify({ body: 'after comparison payload' });
    const view = (comparisonData = candidate) => (
      <ModelTraceExplorerFieldRenderer
        title="metadata"
        data={data}
        comparisonData={comparisonData}
        comparisonEnabled
        comparisonRole="baseline"
        renderMode="json"
      />
    );
    const parse = jest.spyOn(JSON, 'parse');
    const { rerender } = render(view(), { wrapper: ProvidersWrapper });
    const calls = () => parse.mock.calls.filter(([value]) => String(value).includes('comparison payload')).length;
    const initialCalls = calls();
    expect(initialCalls).toBeGreaterThan(0);
    rerender(view());
    expect(calls()).toBe(initialCalls);
    rerender(view(JSON.stringify({ body: 'replacement comparison payload' })));
    expect(screen.getByText(/replacement comparison payload/)).toBeInTheDocument();
    expect(calls()).toBeGreaterThan(initialCalls);
  });

  it('retains payload classification across unrelated rerenders', () => {
    const parsedValue = { body: 'classification payload' };
    const data = JSON.stringify(parsedValue);
    const normalize = jest.spyOn(traceUtils, 'normalizeConversation');
    const view = () => (
      <ModelTraceExplorerFieldRenderer title="metadata" data={data} parsedValue={parsedValue} renderMode="default" />
    );
    const { rerender } = render(view(), { wrapper: ProvidersWrapper });
    const calls = () => normalize.mock.calls.filter(([value]) => (value as any)?.metadata === parsedValue).length;
    const initialCalls = calls();
    expect(initialCalls).toBeGreaterThan(0);
    rerender(view());
    expect(calls()).toBe(initialCalls);
  });

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
