import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';

import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';
import { CodeSnippetRenderMode } from './ModelTrace.types';

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('trace snippet preview', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

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
