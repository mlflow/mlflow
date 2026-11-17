import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render } from '@databricks/web-shared/test-utils/render';

import { CollapsibleJsonViewer } from './CollapsibleJsonViewer';
import { ModelTraceExplorerCodeSnippetBody, JSON_PREVIEW_ROW_LIMIT } from './ModelTraceExplorerCodeSnippetBody';
import { ModelTraceExplorerJsonRowsViewer } from './ModelTraceExplorerJsonRowsViewer';

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

  it('mounts only the preview rows before See more is requested', () => {
    render(<ModelTraceExplorerCodeSnippetBody data={JSON.stringify(payload)} />, { wrapper: Providers });
    expect(screen.getByText('"field-0"')).toBeInTheDocument();
    expect(screen.queryByText('"field-4999"')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'See more' })).toBeInTheDocument();
  });
});

describe('large JSON row interaction', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    mockCopy.mockClear();
  });

  it('keeps the viewport and focused toggle mounted on collapse', async () => {
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const height = this.hasAttribute('data-index') ? 24 : 480;
      return { x: 0, y: 0, top: 0, left: 0, right: 800, bottom: height, width: 800, height, toJSON: () => ({}) };
    });
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

  it('uses the large-payload expansion depth with either initialExpanded value', () => {
    const data = JSON.stringify({ nested: { leaf: 'visible default' }, padding: 'z'.repeat(100_000) });
    const { rerender } = render(<CollapsibleJsonViewer data={data} rowLimit={JSON_PREVIEW_ROW_LIMIT} />, {
      wrapper: Providers,
    });
    expect(screen.getByText('"visible default"')).toBeInTheDocument();
    rerender(<CollapsibleJsonViewer data={data} initialExpanded rowLimit={JSON_PREVIEW_ROW_LIMIT} />);
    expect(screen.getByText('"visible default"')).toBeInTheDocument();
  });

  it('virtualizes expanded rows and reaches the final value through the real scroll listener', async () => {
    jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const height = this.hasAttribute('data-index') ? 24 : 480;
      return { x: 0, y: 0, top: 0, left: 0, right: 800, bottom: height, width: 800, height, toJSON: () => ({}) };
    });
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
    const padded = { 'a.b': { 'quoted"key': 'café 日本語 🙂' }, padding: 'z'.repeat(100_000) };
    const { rerender } = render(
      <CollapsibleJsonViewer data={JSON.stringify(padded)} initialExpanded rowLimit={JSON_PREVIEW_ROW_LIMIT} />,
      {
        wrapper: Providers,
      },
    );
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
        rowLimit={JSON_PREVIEW_ROW_LIMIT}
      />,
    );
    expect(screen.getByText('"updated Кириллица العربية"')).toBeInTheDocument();
    expect(screen.queryByText('"café 日本語 🙂"')).not.toBeInTheDocument();
  });
});
