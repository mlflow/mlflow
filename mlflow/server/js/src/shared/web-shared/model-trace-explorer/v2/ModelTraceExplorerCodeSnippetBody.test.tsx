import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent } from '@testing-library/react';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { formatJsonStringAsYaml } from './ModelTraceExplorerYaml.utils';

const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('trace snippet preview', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    mockCopy.mockReset();
  });

  it.each([CodeSnippetRenderMode.MARKDOWN, CodeSnippetRenderMode.YAML, CodeSnippetRenderMode.TEXT])(
    'bounds expanded %s text and keeps its full Copy value',
    (renderMode) => {
      jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
        const height = this.hasAttribute('data-index') ? 20 : 400;
        return { x: 0, y: 0, top: 0, left: 0, width: 1000, height, right: 1000, bottom: height, toJSON: () => ({}) };
      });
      const text = '**synthetic snippet** café 日本語 🙂\n'.repeat(8000);
      const data = JSON.stringify(text);
      const { container } = render(<ModelTraceExplorerCodeSnippetBody data={data} renderMode={renderMode} />, {
        wrapper: Providers,
      });
      expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
      fireEvent.click(screen.getByText('See more'));
      expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('synthetic snippet');
      expect(container.querySelectorAll('*').length).toBeLessThan(200);
      expect(screen.getByText(/Browser find and text selection cover only part of the value/)).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
      expect(mockCopy).toHaveBeenCalledWith(
        renderMode === CodeSnippetRenderMode.YAML
          ? formatJsonStringAsYaml(data)
          : renderMode === CodeSnippetRenderMode.MARKDOWN
            ? text
            : data,
      );
    },
  );

  it('keeps the preview bounded without splitting the complete large value', () => {
    const text = Array.from({ length: 3000 }, (_, index) => `synthetic-line-${index}`).join('\n');
    const split = jest.spyOn(String.prototype, 'split');
    render(
      <ModelTraceExplorerCodeSnippetBody data={JSON.stringify(text)} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Providers },
    );
    expect(screen.getByText('See more')).toBeInTheDocument();
    expect(screen.getByText(/synthetic-line-0/)).toBeInTheDocument();
    expect(screen.queryByText(/synthetic-line-2999/)).not.toBeInTheDocument();
    expect(split.mock.instances.some((receiver) => String(receiver) === text)).toBe(false);
  });

  it('limits line splitting even below the preview character limit', () => {
    const text = Array.from({ length: 20 }, (_, index) => `line-${index}`).join('\n');
    expect(text.length).toBeLessThan(300);
    const split = jest.spyOn(String.prototype, 'split');
    render(
      <ModelTraceExplorerCodeSnippetBody data={JSON.stringify(text)} renderMode={CodeSnippetRenderMode.MARKDOWN} />,
      { wrapper: Providers },
    );
    expect(screen.getByText('See more')).toBeInTheDocument();
    const fullValueCalls = split.mock.calls.filter((_, index) => String(split.mock.instances[index]) === text);
    expect(fullValueCalls).toContainEqual(['\n', 11]);
    expect(fullValueCalls.every(([, limit]) => limit !== undefined && Number(limit) <= 11)).toBe(true);
    expect(screen.queryByText(/line-19/)).not.toBeInTheDocument();
  });
});
