import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerRetrieverDocument as LegacyDocument } from './right-pane/ModelTraceExplorerRetrieverDocument';
import { ModelTraceExplorerRetrieverDocument as RedesignedDocument } from './v2/right-pane/ModelTraceExplorerRetrieverDocument';
import { ModelTraceExplorerRetrieverDocumentFull as LegacyFull } from './right-pane/ModelTraceExplorerRetrieverDocumentFull';
import { ModelTraceExplorerRetrieverDocumentFull as RedesignedFull } from './v2/right-pane/ModelTraceExplorerRetrieverDocumentFull';
import { LARGE_TRACE_TEXT_LIMIT } from './constants';

const mockMarkdownInputs: unknown[] = [];
const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));
jest.mock('../genai-markdown-renderer/GenAIMarkdownRenderer', () => ({
  GenAIMarkdownRenderer: ({ children }: { children: string }) => {
    mockMarkdownInputs.push(children);
    return <span>{children}</span>;
  },
}));

const largeText = `DOC-START café 🌳\n${'A synthetic document line café 🌳 with complete original text.\n'.repeat(2500)}DOC-END café 🌳`;
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>
    <IntlProvider locale="en">{children}</IntlProvider>
  </DesignSystemProvider>
);

describe.each([
  ['legacy', LegacyDocument],
  ['redesigned', RedesignedDocument],
] as const)('%s expanded retriever document', (_, Document) => {
  beforeEach(() => {
    mockMarkdownInputs.length = 0;
    mockCopy.mockReset();
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const height = this.hasAttribute('data-index') ? 20 : 400;
      return { x: 0, y: 0, top: 0, left: 0, bottom: height, right: 400, width: 400, height, toJSON: () => ({}) };
    });
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('keeps oversized expanded content out of Markdown and bounds rendered text', async () => {
    render(<Document text={largeText} metadata={{ source: 'fixture-source' }} />, { wrapper: Wrapper });
    await userEvent.click(screen.getByText(/^DOC-START/));
    expect(mockMarkdownInputs).toEqual([]);
    const region = screen.getByRole('region', { name: 'Trace text' });
    expect(region.textContent?.length).toBeLessThan(10000);
    expect(region.textContent).toContain('DOC-START café 🌳');
    expect(screen.getByText(/fixture-source/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'See less' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(largeText);
  });

  it('preserves Markdown for short documents and supports collapse', async () => {
    render(<Document text="**short café 🌳 document**" metadata={{ source: 'fixture-source' }} />, {
      wrapper: Wrapper,
    });
    await userEvent.click(screen.getByText('**short café 🌳 document**'));
    expect(mockMarkdownInputs).toEqual(['**short café 🌳 document**']);
    await userEvent.click(screen.getByRole('button', { name: 'See less' }));
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
    expect(screen.getByText('**short café 🌳 document**')).toBeInTheDocument();
  });
});

describe.each([
  ['legacy', LegacyFull],
  ['redesigned', RedesignedFull],
] as const)('%s full document boundary', (_, Full) => {
  beforeEach(() => {
    mockMarkdownInputs.length = 0;
  });

  it('retains Markdown at the text limit', () => {
    const text = 'x'.repeat(LARGE_TRACE_TEXT_LIMIT);
    render(<Full text={text} metadataTags={[]} setExpanded={jest.fn()} />, { wrapper: Wrapper });
    expect(mockMarkdownInputs).toEqual([text]);
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
  });

  it('uses the bounded renderer one unit above the limit', () => {
    render(<Full text={'x'.repeat(LARGE_TRACE_TEXT_LIMIT + 1)} metadataTags={[]} setExpanded={jest.fn()} />, {
      wrapper: Wrapper,
    });
    expect(mockMarkdownInputs).toEqual([]);
    expect(screen.getByRole('region', { name: 'Trace text' })).toBeInTheDocument();
  });

  it.each([null, undefined, false, 0])('retains malformed runtime content %s without throwing', (value) => {
    render(<Full text={value as unknown as string} metadataTags={[]} setExpanded={jest.fn()} />, {
      wrapper: Wrapper,
    });
    expect(mockMarkdownInputs).toEqual([value]);
    expect(screen.queryByRole('region', { name: 'Trace text' })).not.toBeInTheDocument();
  });
});
