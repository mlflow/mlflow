import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const longString = JSON.stringify('a'.repeat(500));

describe('ModelTraceExplorerCodeSnippetBody', () => {
  it('renders a copy button in JSON render mode', () => {
    render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.JSON} />,
      { wrapper: Wrapper },
    );

    // SnippetCopyAction renders a button with CopyIcon; the tooltip is "Copy"
    expect(screen.queryByTitle('Copy')).not.toBeNull();
  });

  it('renders a copy button in markdown render mode', () => {
    render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Wrapper },
    );

    expect(screen.queryByTitle('Copy')).not.toBeNull();
  });

  it('renders a copy button in text render mode', () => {
    render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.TEXT} />,
      { wrapper: Wrapper },
    );

    expect(screen.queryByTitle('Copy')).not.toBeNull();
  });
});
