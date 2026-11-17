import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { formatJsonStringAsYaml } from './ModelTraceExplorerYaml.utils';

describe('collapsed text preview allocation', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([CodeSnippetRenderMode.TEXT, CodeSnippetRenderMode.YAML])(
    'shows a short %s preview without splitting the complete payload',
    (renderMode) => {
      const text = Array.from({ length: 5000 }, (_, index) => `line-${index}: café 日本語 🙂`).join('\n');
      const data = JSON.stringify(renderMode === CodeSnippetRenderMode.YAML ? { document: text } : text);
      const completeSnippet = renderMode === CodeSnippetRenderMode.YAML ? formatJsonStringAsYaml(data) : text;
      const originalSplit = String.prototype.split;
      const payloadSplits: string[] = [];
      jest.spyOn(String.prototype, 'split').mockImplementation(function (
        this: string,
        ...args: Parameters<string['split']>
      ) {
        if (String(this) === completeSnippet) {
          payloadSplits.push(completeSnippet);
        }
        return originalSplit.apply(this, args);
      });
      const { container } = render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <ModelTraceExplorerCodeSnippetBody data={data} renderMode={renderMode} />
          </DesignSystemProvider>
        </IntlProvider>,
      );
      expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
      expect(container.querySelector('pre')).toHaveTextContent('line-0:');
      expect(container.querySelector('pre')).not.toHaveTextContent('line-4999:');
      expect(payloadSplits).toHaveLength(0);
    },
  );
});
