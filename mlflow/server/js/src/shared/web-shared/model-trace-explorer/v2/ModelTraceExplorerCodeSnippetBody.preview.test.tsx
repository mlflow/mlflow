import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { formatJsonStringAsYaml } from './ModelTraceExplorerYaml.utils';
import { ModelTraceExplorerCodeSnippetBody as LegacyCodeSnippetBody } from '../ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode as LegacyRenderMode } from '../ModelTrace.types';

describe('collapsed text preview allocation', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each(['v1', 'v2-text', 'v2-yaml'])(
    'shows a short %s preview without splitting the complete payload',
    (version) => {
      const renderMode = version === 'v2-yaml' ? CodeSnippetRenderMode.YAML : CodeSnippetRenderMode.TEXT;
      const text = Array.from({ length: 5000 }, (_, index) => `line-${index}: café 日本語 🙂`).join('\n');
      const data = JSON.stringify(renderMode === CodeSnippetRenderMode.YAML ? { document: text } : text);
      const completeSnippet = renderMode === CodeSnippetRenderMode.YAML ? formatJsonStringAsYaml(data) : text;
      const originalSplit = String.prototype.split;
      let payloadSplits = 0;
      jest.spyOn(String.prototype, 'split').mockImplementation(function (
        this: string,
        ...args: Parameters<string['split']>
      ) {
        if (String(this) === completeSnippet) {
          payloadSplits += 1;
        }
        return originalSplit.apply(this, args);
      });
      const { container } = render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            {version === 'v1' ? (
              <LegacyCodeSnippetBody data={data} renderMode={LegacyRenderMode.TEXT} />
            ) : (
              <ModelTraceExplorerCodeSnippetBody data={data} renderMode={renderMode} />
            )}
          </DesignSystemProvider>
        </IntlProvider>,
      );
      expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
      expect(container.querySelector('pre')).toHaveTextContent('line-0:');
      expect(container.querySelector('pre')).not.toHaveTextContent('line-4999:');
      expect(payloadSplits).toBe(0);
    },
  );
});
