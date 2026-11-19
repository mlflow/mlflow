import { afterAll, beforeAll, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { cloneDeep } from 'lodash';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../query-client/queryClient';

import type { ModelTrace, ModelTraceInfo, ModelTraceSpanV2 } from './ModelTrace.types';
import { MOCK_TRACE } from './ModelTraceExplorer.test-utils';
import { ModelTraceExplorerDetailView as LegacyDetail } from './ModelTraceExplorerDetailView';
import { ModelTraceExplorerViewStateProvider as LegacyState } from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerPreferencesProvider as LegacyPreferences } from './ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerDetailView as Detail } from './v2/ModelTraceExplorerDetailView';
import { ModelTraceExplorerViewStateProvider as State } from './v2/ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerPreferencesProvider as Preferences } from './v2/ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerContextProvider } from './v2/ModelTraceExplorerContext';

const originalResizeObserver = globalThis.ResizeObserver;
beforeAll(() => {
  globalThis.ResizeObserver = class {
    observe = jest.fn();
    unobserve = jest.fn();
    disconnect = jest.fn();
  } as unknown as typeof ResizeObserver;
});
afterAll(() => {
  globalThis.ResizeObserver = originalResizeObserver;
});

const makeTrace = (): ModelTrace => {
  const trace = cloneDeep(MOCK_TRACE);
  trace.data.spans = [
    ['root', null],
    ['branch-a', 'root'],
    ['leaf-a', 'branch-a'],
    ['branch-b', 'root'],
    ['leaf-b', 'branch-b'],
  ].map(([id, parent]) => ({
    context: { span_id: id, trace_id: 'tr-expansion' },
    name: id,
    parent_id: parent,
    start_time: 0,
    end_time: 1000000,
    attributes: { 'mlflow.spanType': 'CHAIN' },
  })) as ModelTraceSpanV2[];
  return trace;
};

describe.each(['legacy', 'redesigned'])('%s detail expansion across refresh', (variant) => {
  it('retains a collapsed branch and an expanded branch when same-ID payloads are replaced', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const View = variant === 'legacy' ? LegacyDetail : Detail;
    const ViewState = variant === 'legacy' ? LegacyState : State;
    const ViewPreferences = variant === 'legacy' ? LegacyPreferences : Preferences;
    const Wrapper = ({ trace }: { trace: ModelTrace }) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={client}>
            <ViewPreferences>
              <ModelTraceExplorerContextProvider>
                <ViewState modelTrace={trace} assessmentsPaneEnabled={false}>
                  <View modelTraceInfo={trace.info} />
                </ViewState>
              </ModelTraceExplorerContextProvider>
            </ViewPreferences>
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>
    );
    const trace = makeTrace();
    const { rerender } = render(<Wrapper trace={trace} />);
    expect(screen.getByTestId('timeline-tree-node-leaf-a')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-leaf-b')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('toggle-span-expanded-branch-a'));
    expect(screen.queryByTestId('timeline-tree-node-leaf-a')).not.toBeInTheDocument();
    const refreshed = cloneDeep(trace);
    (refreshed.data.spans[0] as ModelTraceSpanV2).attributes = { updated: 'new payload' };
    rerender(<Wrapper trace={refreshed} />);
    expect(screen.queryByTestId('timeline-tree-node-leaf-a')).not.toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-leaf-b')).toBeInTheDocument();
    const growing = cloneDeep(refreshed);
    const addedBranch = cloneDeep(growing.data.spans[3]) as ModelTraceSpanV2;
    addedBranch.context.span_id = 'branch-c';
    addedBranch.name = 'branch-c';
    const addedLeaf = cloneDeep(growing.data.spans[4]) as ModelTraceSpanV2;
    addedLeaf.context.span_id = 'leaf-c';
    addedLeaf.name = 'leaf-c';
    addedLeaf.parent_id = 'branch-c';
    growing.data.spans.push(addedBranch, addedLeaf);
    rerender(<Wrapper trace={growing} />);
    expect(screen.queryByTestId('timeline-tree-node-leaf-a')).not.toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-leaf-b')).toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-leaf-c')).toBeInTheDocument();
    const shrinking = cloneDeep(growing);
    shrinking.data.spans = shrinking.data.spans.filter(
      (span) => (span as ModelTraceSpanV2).context.span_id !== 'leaf-b',
    );
    rerender(<Wrapper trace={shrinking} />);
    expect(screen.queryByTestId('timeline-tree-node-leaf-a')).not.toBeInTheDocument();
    expect(screen.queryByTestId('timeline-tree-node-leaf-b')).not.toBeInTheDocument();
    expect(screen.getByTestId('timeline-tree-node-leaf-c')).toBeInTheDocument();
    const anotherTrace = cloneDeep(shrinking);
    (anotherTrace.info as ModelTraceInfo).request_id = 'tr-other-expansion';
    (anotherTrace.data.spans as ModelTraceSpanV2[]).forEach((span) => {
      span.context.trace_id = 'tr-other-expansion';
    });
    rerender(<Wrapper trace={anotherTrace} />);
    expect(screen.getByTestId('timeline-tree-node-leaf-a')).toBeInTheDocument();
    client.clear();
  });
});
