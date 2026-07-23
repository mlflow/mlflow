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
    const { container } = render(
      <ModelTraceExplorerCodeSnippetBody data={longString} renderMode={CodeSnippetRenderMode.JSON} />,
      { wrapper: Wrapper },
    );

    // The DS Button spreads componentId as data-component-id onto the real DOM element.
    // DesignSystemEventProvider.tsx line 225-226: componentId -> 'data-component-id'.
    expect(
      container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]'),
    ).not.toBeNull();
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

    expect(
      container.querySelector('[data-component-id="shared.model-trace-explorer.copy-snippet"]'),
    ).not.toBeNull();
  });
});
