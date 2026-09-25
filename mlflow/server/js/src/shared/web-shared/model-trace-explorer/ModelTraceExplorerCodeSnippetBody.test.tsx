import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const rawString = 'a'.repeat(500);
const longString = JSON.stringify(rawString);

describe('ModelTraceExplorerCodeSnippetBody', () => {
  beforeEach(() => {
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(async () => {}) },
      writable: true,
    });
  });

  it('renders a copy button in JSON render mode', () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.JSON} />,
      { wrapper: Wrapper },
    );

    expect(container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]')).not.toBeNull();
  });

  it('renders a copy button in markdown render mode', () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Wrapper },
    );

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

  it('copies the full unescaped value in markdown mode while the preview is truncated', async () => {
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Wrapper },
    );

    // the value is long enough to be truncated behind "See more"
    expect(screen.getByText('See more')).toBeInTheDocument();

    const copyButton = container.querySelector(
      '[data-component-id="shared.model-trace-explorer.copy-snippet-markdown"]',
    );
    await userEvent.click(copyButton as Element);

    // the full parsed value is copied, not the truncated preview or the raw JSON-escaped string
    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith(rawString);
  });
});
