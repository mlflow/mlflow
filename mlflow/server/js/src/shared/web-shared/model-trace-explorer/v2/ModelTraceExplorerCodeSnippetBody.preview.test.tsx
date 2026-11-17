import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';

describe('collapsed text preview allocation', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('shows a short preview without splitting every line of the payload', () => {
    const text = Array.from({ length: 5000 }, (_, index) => `line-${index}: café 日本語 🙂`).join('\n');
    const originalSplit = String.prototype.split;
    const payloadSplits: string[] = [];
    jest.spyOn(String.prototype, 'split').mockImplementation(function (
      this: string,
      ...args: Parameters<string['split']>
    ) {
      if (String(this) === text) {
        payloadSplits.push(text);
      }
      return originalSplit.apply(this, args);
    });
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ModelTraceExplorerCodeSnippetBody data={JSON.stringify(text)} renderMode={CodeSnippetRenderMode.TEXT} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
    expect(screen.getByText(/line-0:/)).toBeInTheDocument();
    expect(screen.queryByText(/line-4999:/)).not.toBeInTheDocument();
    expect(payloadSplits).toHaveLength(0);
  });
});
