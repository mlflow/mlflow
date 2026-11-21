import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCodeSnippet as LegacySnippet } from './ModelTraceExplorerCodeSnippet';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippet as Snippet } from './v2/ModelTraceExplorerCodeSnippet';
import { CodeSnippetRenderMode as RedesignedRenderMode } from './v2/ModelTrace.types';

jest.mock('./ModelTraceExplorerCodeSnippetBody', () => ({
  ModelTraceExplorerCodeSnippetBody: ({ renderMode }: { renderMode: string }) => (
    <div data-testid="body-mode">{renderMode}</div>
  ),
}));
jest.mock('./v2/ModelTraceExplorerCodeSnippetBody', () => ({
  ModelTraceExplorerCodeSnippetBody: ({ renderMode }: { renderMode: string }) => (
    <div data-testid="body-mode">{renderMode}</div>
  ),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

afterEach(() => {
  jest.restoreAllMocks();
});

describe.each([
  {
    name: 'legacy',
    element: (data: string, explicit = false) => (
      <LegacySnippet
        title="Payload"
        data={data}
        initialRenderMode={explicit ? CodeSnippetRenderMode.TEXT : undefined}
      />
    ),
    explicitMode: 'text',
  },
  {
    name: 'redesigned',
    element: (data: string, explicit = false) => (
      <Snippet title="Payload" data={data} initialRenderMode={explicit ? RedesignedRenderMode.YAML : undefined} />
    ),
    explicitMode: 'yaml',
  },
])('$name snippet header', ({ element, explicitMode }) => {
  it.each(['{"text":"value"}', '[1,"two"]', 'null', 'false', '42', ' \r\n\t"café 🌲"'])(
    'selects the same default mode for %s without parsing its payload',
    (data) => {
      const expected = typeof JSON.parse(data) === 'string' ? 'markdown' : 'json';
      const parse = jest.spyOn(JSON, 'parse');
      const { container } = render(element(data), { wrapper });
      expect(screen.getByTestId('body-mode')).toHaveTextContent(expected);
      const dropdown = container.querySelector(
        '[data-component-id="shared.model-trace-explorer.snippet-render-mode-tag"]',
      );
      expect(Boolean(dropdown)).toBe(expected === 'markdown');
      expect(parse.mock.calls.filter(([text]) => text === data)).toHaveLength(0);
    },
  );

  it('avoids an additional full parse for a large object and preserves an explicit mode', () => {
    const data = JSON.stringify({ text: 'café 🌲'.repeat(20000) });
    const parse = jest.spyOn(JSON, 'parse');
    render(element(data, true), { wrapper });
    expect(screen.getByTestId('body-mode')).toHaveTextContent(explicitMode);
    expect(parse.mock.calls.filter(([text]) => text === data)).toHaveLength(0);
  });

  it('updates the default mode when replacement data changes between a string and an object', () => {
    const { rerender } = render(element('"hello"'), { wrapper });
    expect(screen.getByTestId('body-mode')).toHaveTextContent('markdown');
    rerender(element('{"text":"hello"}'));
    expect(screen.getByTestId('body-mode')).toHaveTextContent('json');
    rerender(element('"replacement"'));
    expect(screen.getByTestId('body-mode')).toHaveTextContent('markdown');
  });
});
