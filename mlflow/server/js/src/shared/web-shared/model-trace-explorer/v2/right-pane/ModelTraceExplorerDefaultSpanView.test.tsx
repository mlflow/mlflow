import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import userEvent from '@testing-library/user-event';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { ModelTraceExplorerViewStateContext } from '../ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerPayloadCacheProvider } from './ModelTraceExplorerPayloadCache';
import { ModelTraceExplorerDefaultSpanView, useSpanComparisonFields } from './ModelTraceExplorerDefaultSpanView';
import { MOCK_CHAT_SPAN } from '../../ModelTraceExplorer.test-utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';

const payload = { document: { body: 'unchanged synthetic payload café 日本語 🙂' } };
const buildSpan = (): ModelTraceSpanNode => ({
  ...MOCK_CHAT_SPAN,
  key: 'payload-span',
  chatMessageFormat: undefined,
  chatMessages: undefined,
  chatTools: undefined,
  inputs: payload,
  outputs: undefined,
});
const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ModelTraceExplorerPayloadCacheProvider>{children}</ModelTraceExplorerPayloadCacheProvider>
      </QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);
const spanView = (span: ModelTraceSpanNode) => (
  <ModelTraceExplorerDefaultSpanView activeSpan={span} searchFilter="" activeMatch={null} defaultRenderMode="default" />
);

describe('selected payload reuse', () => {
  it.each(['json', 'default'] as const)('does not reparse an ordinary selected span in %s mode', (mode) => {
    const parse = jest.spyOn(JSON, 'parse');
    render(
      <ModelTraceExplorerDefaultSpanView
        activeSpan={buildSpan()}
        searchFilter=""
        activeMatch={null}
        defaultRenderMode={mode}
      />,
      { wrapper: Providers },
    );
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
    expect(parse.mock.calls.filter(([value]) => String(value).includes(payload.document.body))).toHaveLength(0);
  });

  it.each([{ '': payload.document }, [payload.document]])(
    'reuses Pretty values for empty keys and arrays: %j',
    (inputs) => {
      const parse = jest.spyOn(JSON, 'parse');
      render(spanView({ ...buildSpan(), inputs }), { wrapper: Providers });
      expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
        'unchanged synthetic payload café 日本語',
      );
      expect(parse.mock.calls.filter(([value]) => String(value).includes(payload.document.body))).toHaveLength(0);
    },
  );

  it('does not build fallback comparison fields in an ordinary trace', () => {
    const message = { role: 'user' as const, content: 'ordinary fallback café' };
    const span = { ...buildSpan(), inputs: undefined, chatMessages: [message] };
    const stringify = jest.spyOn(JSON, 'stringify');
    render(spanView(span), { wrapper: Providers });
    expect(stringify.mock.calls.filter(([value]) => Array.isArray(value) && value.includes(message))).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent('ordinary fallback café');
  });

  it('shares fallback conversation field serialization between badges and the visible comparison pane', () => {
    const span: ModelTraceSpanNode = {
      ...buildSpan(),
      inputs: undefined,
      chatMessages: [
        { role: 'user', content: 'cache-fallback user café 日本語 🙂' },
        { role: 'assistant', content: 'cache-fallback assistant' },
      ],
    };
    const other: ModelTraceSpanNode = {
      ...span,
      key: 'other',
      chatMessages: [
        { role: 'user', content: 'cache-fallback changed user' },
        { role: 'assistant', content: 'cache-fallback changed assistant' },
      ],
    };
    const Badge = ({ value }: { value: ModelTraceSpanNode }) => {
      const fields = useSpanComparisonFields(value);
      return <span>{fields.inputs.length + fields.outputs.length}</span>;
    };
    const pane = (visible: boolean, active = span) => (
      <>
        <Badge value={active} />
        <Badge value={other} />
        {visible && (
          <ModelTraceExplorerViewStateContext.Consumer>
            {(state) => (
              <ModelTraceExplorerViewStateContext.Provider
                value={{ ...state, selectedNodeComparisonRole: 'baseline', selectedComparisonNode: other }}
              >
                {spanView(active)}
              </ModelTraceExplorerViewStateContext.Provider>
            )}
          </ModelTraceExplorerViewStateContext.Consumer>
        )}
      </>
    );
    const originalMessages = new Set([...(span.chatMessages ?? []), ...(other.chatMessages ?? [])]);
    const stringify = jest.spyOn(JSON, 'stringify');
    const calls = () =>
      stringify.mock.calls.filter(
        ([value, _replacer, space]) =>
          space === 2 && Array.isArray(value) && value.some((message) => originalMessages.has(message)),
      );
    const { rerender } = render(pane(true), { wrapper: Providers });
    expect(calls()).toHaveLength(4);
    rerender(pane(false));
    rerender(pane(true, { ...span }));
    expect(calls()).toHaveLength(4);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent('cache-fallback user café');
    const replacement = { ...span, chatMessages: [{ role: 'user' as const, content: 'cache-fallback replacement' }] };
    originalMessages.add(replacement.chatMessages[0]);
    rerender(pane(true, replacement));
    expect(calls()).toHaveLength(5);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'cache-fallback replacement',
    );
  });

  it.each(['json', 'yaml'] as const)('does not serialize hidden %s sections during a comparison', (mode) => {
    const span = buildSpan();
    const stringify = jest.spyOn(JSON, 'stringify');
    render(
      <ModelTraceExplorerViewStateContext.Consumer>
        {(state) => (
          <ModelTraceExplorerViewStateContext.Provider
            value={{ ...state, selectedNodeComparisonRole: 'baseline', selectedComparisonNode: buildSpan() }}
          >
            <ModelTraceExplorerDefaultSpanView
              activeSpan={span}
              searchFilter=""
              activeMatch={null}
              defaultRenderMode={mode}
            />
          </ModelTraceExplorerViewStateContext.Provider>
        )}
      </ModelTraceExplorerViewStateContext.Consumer>,
      { wrapper: Providers },
    );
    expect(stringify.mock.calls.filter(([value]) => value === payload)).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload',
    );
  });

  it('retains full section JSON when JSON tab content remounts', () => {
    const pane = (visible: boolean) => (
      <ModelTraceExplorerPayloadCacheProvider>
        {visible && (
          <ModelTraceExplorerDefaultSpanView
            activeSpan={buildSpan()}
            searchFilter=""
            activeMatch={null}
            defaultRenderMode="json"
          />
        )}
      </ModelTraceExplorerPayloadCacheProvider>
    );
    const stringify = jest.spyOn(JSON, 'stringify');
    const { rerender } = render(pane(true), { wrapper: Providers });
    expect(stringify.mock.calls.filter(([value]) => value === payload)).toHaveLength(1);
    rerender(pane(false));
    stringify.mockClear();
    rerender(pane(true));
    expect(stringify.mock.calls.filter(([value]) => value === payload)).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
  });

  it('shows null when switching a fallback conversation with undefined inputs to JSON', async () => {
    const span = {
      ...buildSpan(),
      inputs: undefined,
      chatMessages: [{ role: 'user' as const, content: 'fallback user text' }],
    };
    render(spanView(span), { wrapper: Providers });
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent('fallback user text');
    await userEvent.click(screen.getByRole('button', { name: 'Pretty' }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'JSON' }));
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent('null');
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).not.toHaveTextContent('fallback user text');
  });

  it('reuses serialized fields after tab content remounts and displays replacement payloads', () => {
    const pane = (span?: ModelTraceSpanNode) => (
      <ModelTraceExplorerPayloadCacheProvider>{span && spanView(span)}</ModelTraceExplorerPayloadCacheProvider>
    );
    const { rerender } = render(pane(buildSpan()), { wrapper: Providers });
    rerender(pane());
    const stringify = jest.spyOn(JSON, 'stringify');
    rerender(pane(buildSpan()));
    expect(stringify.mock.calls.filter(([value]) => value === payload.document)).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
    rerender(pane({ ...buildSpan(), inputs: { document: { body: 'replacement payload' } } }));
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent('replacement payload');
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).not.toHaveTextContent(
      'unchanged synthetic payload',
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });
  it('does not serialize a section again when its JSON view rerenders', () => {
    const view = () => (
      <ModelTraceExplorerDefaultSpanView
        activeSpan={buildSpan()}
        searchFilter=""
        activeMatch={null}
        defaultRenderMode="json"
      />
    );
    const { rerender } = render(view(), { wrapper: Providers });
    const stringify = jest.spyOn(JSON, 'stringify');
    rerender(view());
    expect(stringify.mock.calls.filter(([value]) => value === payload)).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
  });
  it('preserves displayed payload without serializing an unchanged payload when the span wrapper changes', () => {
    const { rerender } = render(spanView(buildSpan()), { wrapper: Providers });
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
    const stringify = jest.spyOn(JSON, 'stringify');
    const parse = jest.spyOn(JSON, 'parse');
    rerender(spanView(buildSpan()));
    expect(stringify.mock.calls.filter(([value]) => value === payload.document)).toHaveLength(0);
    expect(parse.mock.calls.filter(([value]) => String(value).includes(payload.document.body))).toHaveLength(0);
    expect(screen.getByTestId('model-trace-explorer-default-span-view')).toHaveTextContent(
      'unchanged synthetic payload café 日本語',
    );
  });
});
