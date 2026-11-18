import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render } from '@databricks/web-shared/test-utils/render';

import { CollapsibleJsonViewer } from './CollapsibleJsonViewer';
import { ModelTraceExplorerCodeSnippetBody, JSON_PREVIEW_ROW_LIMIT } from './ModelTraceExplorerCodeSnippetBody';
import { ModelTraceExplorerJsonRowsViewer } from './ModelTraceExplorerJsonRowsViewer';
import { JSON_ROWS_MAX_DEPTH } from './ModelTraceExplorerJsonRows.utils';

const mockCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));

const payload = Object.fromEntries(Array.from({ length: 5000 }, (_, index) => [`field-${index}`, `value-${index}`]));

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('large structured JSON preview', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    mockCopy.mockClear();
  });
  it('copies the complete Unicode JSON while only the preview is mounted', async () => {
    const user = userEvent.setup();
    const data = JSON.stringify({ ...payload, unicode: 'café 日本語 🙂 Кириллица العربية', final: 'complete' });
    render(<ModelTraceExplorerCodeSnippetBody data={data} />, { wrapper: Providers });
    await user.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(data);
    expect(JSON.parse(mockCopy.mock.calls[0][0] as string)).toEqual(JSON.parse(data));
    expect(screen.queryByText('"field-4999"')).not.toBeInTheDocument();
  });

  it('keeps clipped JSON containers non-interactive for mouse and keyboard users', async () => {
    const user = userEvent.setup();
    const value = { document: payload };
    render(<ModelTraceExplorerCodeSnippetBody data={JSON.stringify(value)} />, { wrapper: Providers });
    expect(screen.queryByRole('button', { name: 'document' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'JSON container' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy' })).toBeInTheDocument();
    await user.click(screen.getByText('"document"'));
    expect(screen.getByText('"field-0"')).toBeInTheDocument();
  });

  it('mounts only the preview rows before See more is requested', () => {
    render(<ModelTraceExplorerCodeSnippetBody data={JSON.stringify(payload)} />, { wrapper: Providers });
    expect(screen.getByText('"field-0"')).toBeInTheDocument();
    expect(screen.queryByText('"field-4999"')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
  });
});

describe('large JSON row interaction', () => {
  beforeEach(() => {
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const height = this.hasAttribute('data-index') ? 24 : 480;
      return { x: 0, y: 0, top: 0, left: 0, right: 800, bottom: height, width: 800, height, toJSON: () => ({}) };
    });
  });
  afterEach(() => {
    jest.restoreAllMocks();
    mockCopy.mockClear();
  });

  it('keeps the viewport and focused toggle mounted on collapse', async () => {
    const user = userEvent.setup();
    const value = { document: Object.fromEntries(Array.from({ length: 200 }, (_, index) => [`key-${index}`, index])) };
    render(<ModelTraceExplorerJsonRowsViewer value={value} expandDepth={1} />, { wrapper: Providers });
    const viewport = screen.getByRole('region', { name: 'JSON' });
    const toggle = screen.getByRole('button', { name: 'document' });
    toggle.focus();
    await user.keyboard('{Enter}');
    expect(screen.getByRole('region', { name: 'JSON' })).toBe(viewport);
    expect(screen.getByRole('button', { name: 'document' })).toBe(toggle);
    expect(toggle).toHaveFocus();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await user.keyboard('{Enter}');
    expect(toggle).toHaveFocus();
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('"key-0"')).toBeInTheDocument();
  });

  it.each([
    Array.from({ length: 1000 }, (_, index) => index),
    Object.fromEntries(Array.from({ length: 1000 }, (_, index) => [`k${index}`, index])),
  ])('bounds previews with many short entries below the character threshold', (value) => {
    const data = JSON.stringify(value);
    expect(data.length).toBeLessThan(100_000);
    render(<CollapsibleJsonViewer data={data} rowLimit={JSON_PREVIEW_ROW_LIMIT} />, { wrapper: Providers });
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.queryByText('999')).not.toBeInTheDocument();
  });

  it.each([198, 199])('selects the correct renderer for an array with %i entries plus two brackets', (entries) => {
    const data = JSON.stringify(Array.from({ length: entries }, (_, index) => index));
    render(<CollapsibleJsonViewer data={data} rowLimit={JSON_PREVIEW_ROW_LIMIT} />, { wrapper: Providers });
    if (entries === 198) {
      expect(screen.getAllByText('197').length).toBeGreaterThan(0);
    } else {
      expect(screen.queryByText('197')).not.toBeInTheDocument();
    }
  });

  it('renders the maximum-depth message when the nested path remains expanded', () => {
    let value: unknown = 'hidden leaf';
    for (let depth = 0; depth < JSON_ROWS_MAX_DEPTH + 2; depth++) value = [value];
    render(
      <ModelTraceExplorerJsonRowsViewer value={value as unknown[]} expandDepth={JSON_ROWS_MAX_DEPTH} rowLimit={300} />,
      { wrapper: Providers },
    );
    expect(screen.getByText('[Max depth reached]')).toBeInTheDocument();
    expect(screen.queryByText('"hidden leaf"')).not.toBeInTheDocument();
  });

  it.each([100_000, 1_000_001])('keeps small nesting expanded beside a %i-character string', (length) => {
    const data = JSON.stringify({
      nested: { deeper: { deepest: { leaf: 'visible default' } } },
      padding: 'z'.repeat(length),
    });
    const { rerender } = render(<CollapsibleJsonViewer data={data} rowLimit={JSON_PREVIEW_ROW_LIMIT} />, {
      wrapper: Providers,
    });
    expect(screen.getByText('"visible default"')).toBeInTheDocument();
    rerender(<CollapsibleJsonViewer data={data} initialExpanded rowLimit={JSON_PREVIEW_ROW_LIMIT} />);
    expect(screen.getByText('"visible default"')).toBeInTheDocument();
    rerender(
      <CollapsibleJsonViewer data={JSON.stringify({ nested: { deeper: { deepest: { leaf: 'visible default' } } } })} />,
    );
    expect(screen.getByText('"visible default"')).toBeInTheDocument();
  });

  it('virtualizes expanded rows and reaches the final value through the real scroll listener', async () => {
    const user = userEvent.setup();
    render(<ModelTraceExplorerCodeSnippetBody data={JSON.stringify(payload)} />, { wrapper: Providers });
    await user.click(screen.getByRole('button', { name: 'See more' }));
    const viewport = screen.getByRole('region', { name: 'JSON' });
    expect(screen.getByText('"field-0"')).toBeInTheDocument();
    expect(screen.queryByText('"field-4999"')).not.toBeInTheDocument();
    fireEvent.scroll(viewport, { target: { scrollTop: 5000 * 24 } });
    await waitFor(() => expect(screen.getByText('"field-4999"')).toBeInTheDocument());
    expect(screen.queryByText('"field-100"')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'See less' }));
    expect(screen.getByText('"field-0"')).toBeInTheDocument();
    expect(screen.queryByText('"field-4999"')).not.toBeInTheDocument();
  });

  it('preserves keyboard container toggling, punctuation keys and replacement Unicode values', async () => {
    const user = userEvent.setup();
    const padded = {
      'a.b': { 'quoted"key': 'café 日本語 🙂' },
      padding: Array.from({ length: 300 }, (_, index) => index),
    };
    const { rerender } = render(<CollapsibleJsonViewer data={JSON.stringify(padded)} initialExpanded />, {
      wrapper: Providers,
    });
    const toggle = screen.getByRole('button', { name: 'a.b' });
    toggle.focus();
    await user.keyboard('{Enter}');
    expect(screen.queryByText('"café 日本語 🙂"')).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await user.keyboard(' ');
    expect(screen.getByText('"café 日本語 🙂"')).toBeInTheDocument();
    rerender(
      <CollapsibleJsonViewer
        data={JSON.stringify({ ...padded, 'a.b': { 'quoted"key': 'updated Кириллица العربية' } })}
        initialExpanded
      />,
    );
    expect(screen.getByText('"updated Кириллица العربية"')).toBeInTheDocument();
    expect(screen.queryByText('"café 日本語 🙂"')).not.toBeInTheDocument();
  });
});
