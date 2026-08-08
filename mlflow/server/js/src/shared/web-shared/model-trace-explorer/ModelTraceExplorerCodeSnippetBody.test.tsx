import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';

// Capture the text passed to the clipboard so we can assert copied === displayed content.
const mockClipboardCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockClipboardCopy }),
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const longString = JSON.stringify('a'.repeat(500));

beforeEach(() => {
  mockClipboardCopy.mockClear();
});

describe('ModelTraceExplorerCodeSnippetBody', () => {
  it('renders a copy button in JSON render mode', () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.JSON} />,
      { wrapper: Wrapper },
    );

    // The DS Button spreads componentId as data-component-id onto the real DOM element.
    // DesignSystemEventProvider.tsx line 225-226: componentId -> 'data-component-id'.
    expect(container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]')).not.toBeNull();
  });

  it('renders a copy button in markdown render mode', () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Wrapper },
    );

    // The markdown branch uses componentId="shared.model-trace-explorer.copy-snippet-markdown",
    // which is distinct from the JSON/text branch. This assertion verifies the markdown path
    // specifically — a tooltip-portal failure cannot cause a false pass.
    expect(
      container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet-markdown"]'),
    ).not.toBeNull();
  });

  it('renders a copy button in text render mode', () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.TEXT} />,
      { wrapper: Wrapper },
    );

    expect(container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]')).not.toBeNull();
  });

  it('copies the unescaped rendered text in text render mode, not the raw JSON string', () => {
    // The data prop is a JSON-encoded string with escaped newlines. The UI renders the
    // parsed/unescaped value, so the copy button must place that same value on the clipboard.
    const data = JSON.stringify('line one\nline two');
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={data} renderMode={CodeSnippetRenderMode.TEXT} />,
      { wrapper: Wrapper },
    );

    const copyButton = container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]');
    fireEvent.click(copyButton as Element);

    expect(mockClipboardCopy).toHaveBeenCalledWith('line one\nline two');
  });

  it('copies the unescaped rendered text in markdown render mode', () => {
    const data = JSON.stringify('# Heading\n\nbody text');
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={data} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Wrapper },
    );

    const copyButton = container.querySelector(
      '[data-component-id="shared.model-trace-explorer.copy-snippet-markdown"]',
    );
    fireEvent.click(copyButton as Element);

    expect(mockClipboardCopy).toHaveBeenCalledWith('# Heading\n\nbody text');
  });
});
