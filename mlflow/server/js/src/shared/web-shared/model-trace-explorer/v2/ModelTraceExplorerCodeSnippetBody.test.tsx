import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent, waitFor } from '@testing-library/react';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';
import { formatJsonStringAsYaml } from './ModelTraceExplorerYaml.utils';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
import { parseModelTraceToTreeWithMultipleRoots } from './ModelTraceExplorer.utils';

const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));
HTMLElement.prototype.scrollIntoView = jest.fn();
HTMLElement.prototype.scrollTo = jest.fn();

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

  it.each(Object.values(CodeSnippetRenderMode))(
    'bounds large %s search hits and preserves full Copy',
    async (renderMode) => {
      jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
        const height = this.hasAttribute('data-index') ? 20 : 400;
        return { x: 0, y: 0, top: 0, left: 0, width: 1000, height, right: 1000, bottom: height, toJSON: () => ({}) };
      });
      const data = JSON.stringify('Search-hit café 日本語 🙂 Search-hit\n' + 'filler\n'.repeat(20000));
      const exec = jest.spyOn(RegExp.prototype, 'exec');
      const lower = jest.spyOn(String.prototype, 'toLowerCase');
      const snippet = (matchIndex: number) => (
        <ModelTraceExplorerCodeSnippetBody
          data={data}
          renderMode={renderMode}
          searchFilter="search-hit"
          activeMatch={{
            span: parseModelTraceToTreeWithMultipleRoots(MOCK_TRACE)[0],
            section: 'outputs',
            key: '',
            isKeyMatch: false,
            matchIndex,
          }}
          containsActiveMatch
        />
      );
      const revealCountBefore = jest.mocked(HTMLElement.prototype.scrollIntoView).mock.calls.length;
      const { container, rerender } = render(snippet(0), { wrapper: Providers });
      await waitFor(() =>
        expect(jest.mocked(HTMLElement.prototype.scrollIntoView).mock.calls.length).toBeGreaterThan(revealCountBefore),
      );
      expect(screen.getByRole('region', { name: 'Trace text' })).toHaveTextContent('Search-hit');
      expect(container.querySelectorAll('*').length).toBeLessThan(300);
      const searchedData = renderMode === CodeSnippetRenderMode.YAML ? formatJsonStringAsYaml(data) : data;
      const scans = () => exec.mock.calls.filter(([value]) => value === searchedData).length;
      const lowerings = () => lower.mock.instances.filter((value) => String(value) === searchedData).length;
      const initialScans = scans();
      const initialLowerings = lowerings();
      expect(initialScans).toBeGreaterThan(0);
      expect(initialLowerings).toBe(1);
      const reveals = jest.mocked(HTMLElement.prototype.scrollIntoView).mock.instances;
      const firstReveal = reveals[reveals.length - 1];
      expect(firstReveal).toHaveTextContent('Search-hit');
      const initialScrolls = jest.mocked(HTMLElement.prototype.scrollTo).mock.calls.length;
      const initialReveals = jest.mocked(HTMLElement.prototype.scrollIntoView).mock.calls.length;
      rerender(snippet(1));
      await waitFor(() =>
        expect(jest.mocked(HTMLElement.prototype.scrollIntoView).mock.calls.length).toBeGreaterThan(initialReveals),
      );
      expect(scans()).toBe(initialScans);
      expect(lowerings()).toBe(initialLowerings);
      const nextReveal = reveals[reveals.length - 1];
      expect(nextReveal).toHaveTextContent('Search-hit');
      expect(nextReveal).not.toBe(firstReveal);
      expect(jest.mocked(HTMLElement.prototype.scrollTo).mock.calls.length).toBeGreaterThan(initialScrolls);
      fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
      expect(mockCopy).toHaveBeenCalledWith(
        renderMode === CodeSnippetRenderMode.YAML ? formatJsonStringAsYaml(data) : data,
      );
    },
  );

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
