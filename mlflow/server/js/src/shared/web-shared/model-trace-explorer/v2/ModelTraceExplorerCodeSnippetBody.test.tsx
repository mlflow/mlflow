import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { dump } from 'js-yaml';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import type { SearchMatch } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { ModelTraceExplorerPayloadCacheProvider } from './right-pane/ModelTraceExplorerPayloadCache';

jest.mock('js-yaml', () => {
  const actual = jest.requireActual<typeof import('js-yaml')>('js-yaml');
  return { ...actual, dump: jest.fn(actual.dump) };
});

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('collapsed YAML serialization', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    HTMLElement.prototype.scrollIntoView = jest.fn();
    Object.defineProperty(global.navigator, 'clipboard', {
      value: { writeText: jest.fn(async () => {}) },
      writable: true,
    });
  });

  it('does not dump the complete collection just to display its collapsed preview', () => {
    const records = Array.from({ length: 5000 }, (_, index) => ({ index, message: 'café 日本語 🙂' }));
    render(
      <ModelTraceExplorerCodeSnippetBody data={JSON.stringify(records)} renderMode={CodeSnippetRenderMode.YAML} />,
      { wrapper: Wrapper },
    );

    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
    const largestDumpedCollection = Math.max(
      ...jest.mocked(dump).mock.calls.map(([value]: unknown[]) => (Array.isArray(value) ? value.length : 0)),
    );
    expect(largestDumpedCollection).toBeLessThan(1000);
  });

  it('copies the full Unicode YAML before expanding its partial preview', async () => {
    const records = Array.from({ length: 500 }, (_, index) => ({ index, message: `record ${index}: café 日本語 🙂` }));
    const fullYaml = jest
      .requireActual<typeof import('js-yaml')>('js-yaml')
      .dump(records, {
        noRefs: true,
        lineWidth: 120,
        sortKeys: false,
      })
      .trimEnd();
    render(
      <ModelTraceExplorerCodeSnippetBody data={JSON.stringify(records)} renderMode={CodeSnippetRenderMode.YAML} />,
      { wrapper: Wrapper },
    );
    expect(screen.queryByText(/record 499/)).not.toBeInTheDocument();
    expect(
      jest.mocked(dump).mock.calls.some(([value]: unknown[]) => Array.isArray(value) && value.length === 500),
    ).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(global.navigator.clipboard.writeText).toHaveBeenCalledWith(fullYaml);
    expect(
      jest.mocked(dump).mock.calls.some(([value]: unknown[]) => Array.isArray(value) && value.length === 500),
    ).toBe(true);
    expect(screen.queryByText(/record 499/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'See more' }));
    expect(screen.getByText(/record 499/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'See less' }));
    expect(screen.queryByText(/record 499/)).not.toBeInTheDocument();
  });

  const records = Array.from({ length: 500 }, (_, index) => ({
    index,
    message: `record${index} content café 日本語 🙂`,
  }));
  const activeMatch: SearchMatch = {
    span: { key: 'span' } as SearchMatch['span'],
    section: 'inputs',
    key: 'data',
    isKeyMatch: false,
    matchIndex: 0,
  };

  it('shares the full dump across search, expansion and remounts under the pane cache', async () => {
    const data = JSON.stringify(records);
    const CachedWrapper = ({ children }: { children: React.ReactNode }) => (
      <Wrapper>
        <ModelTraceExplorerPayloadCacheProvider>{children}</ModelTraceExplorerPayloadCacheProvider>
      </Wrapper>
    );
    const { rerender } = render(
      <ModelTraceExplorerCodeSnippetBody data={data} renderMode={CodeSnippetRenderMode.YAML} />,
      { wrapper: CachedWrapper },
    );
    const previewDumps = jest.mocked(dump).mock.calls.length;
    expect(previewDumps).toBeGreaterThan(0);
    rerender(
      <ModelTraceExplorerCodeSnippetBody
        data={data}
        renderMode={CodeSnippetRenderMode.YAML}
        searchFilter="record499"
        activeMatch={activeMatch}
      />,
    );
    expect(screen.getByText('record499')).toBeInTheDocument();
    const fullDumps = () =>
      jest.mocked(dump).mock.calls.filter(([value]: unknown[]) => Array.isArray(value) && value.length === 500);
    expect(fullDumps()).toHaveLength(1);
    rerender(<ModelTraceExplorerCodeSnippetBody data={data} renderMode={CodeSnippetRenderMode.YAML} />);
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(fullDumps()).toHaveLength(1);
    await userEvent.click(screen.getByRole('button', { name: 'See more' }));
    expect(screen.getByText(/record499/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'See less' }));
    await userEvent.click(screen.getByRole('button', { name: 'See more' }));
    expect(screen.getByText(/record499/)).toBeInTheDocument();
    rerender(<></>);
    rerender(<ModelTraceExplorerCodeSnippetBody data={data} renderMode={CodeSnippetRenderMode.YAML} />);
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(global.navigator.clipboard.writeText).toHaveBeenLastCalledWith(
      jest
        .requireActual<typeof import('js-yaml')>('js-yaml')
        .dump(records, {
          noRefs: true,
          lineWidth: 120,
          sortKeys: false,
        })
        .trimEnd(),
    );
    expect(fullDumps()).toHaveLength(1);
    expect(jest.mocked(dump)).toHaveBeenCalledTimes(previewDumps + 1);
  });

  it.each([true, false])('keeps a tail search match visible when the snippet is active=%s', (containsActiveMatch) => {
    render(
      <ModelTraceExplorerCodeSnippetBody
        data={JSON.stringify(records)}
        renderMode={CodeSnippetRenderMode.YAML}
        searchFilter="record499"
        activeMatch={activeMatch}
        containsActiveMatch={containsActiveMatch}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByText('record499')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'See more' })).not.toBeInTheDocument();
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(containsActiveMatch ? 1 : 0);
  });

  it('keeps unrelated search snippets deferred', () => {
    expect(JSON.stringify(records).length).toBeGreaterThan(24000);
    render(
      <ModelTraceExplorerCodeSnippetBody
        data={JSON.stringify(records)}
        renderMode={CodeSnippetRenderMode.YAML}
        searchFilter="missingpayload"
        activeMatch={activeMatch}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
    expect(
      jest.mocked(dump).mock.calls.some(([value]: unknown[]) => Array.isArray(value) && value.length === 500),
    ).toBe(false);
  });

  it.each(['', 'missingpayload'])('honors initialExpanded with search %j', (searchFilter) => {
    render(
      <ModelTraceExplorerCodeSnippetBody
        data={JSON.stringify(records)}
        renderMode={CodeSnippetRenderMode.YAML}
        initialExpanded
        searchFilter={searchFilter}
        activeMatch={activeMatch}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByText(/record499/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'See less' })).toBeInTheDocument();
  });
});
