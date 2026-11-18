import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { dump } from 'js-yaml';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';

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
    const copy = screen.getAllByRole('button').find((button) => button.textContent !== 'See more');
    await userEvent.click(copy!);
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
});
