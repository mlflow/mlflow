import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { HashRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';
import { MOCK_SPAN_LINKS } from '../../ModelTraceExplorer.test-utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { ModelTraceExplorerLinksTab } from './ModelTraceExplorerLinksTab';

const mockClipboardCopy = jest.fn();

jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockClipboardCopy }),
}));

const mockUseSpanLinkHrefs = jest.fn((traceIds: string[]) => {
  const hrefs: Record<string, string> = {};
  for (const traceId of traceIds) {
    hrefs[traceId] = `/experiments/1/traces?traceId=${traceId}`;
  }
  return hrefs;
});

jest.mock('../../hooks/useSpanLinkHref', () => ({
  useSpanLinkHrefs: (traceIds: string[]) => mockUseSpanLinkHrefs(traceIds),
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient();
  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <HashRouter>{children}</HashRouter>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

const createSpan = (overrides: Partial<ModelTraceSpanNode> = {}): ModelTraceSpanNode => ({
  key: 'span-with-links',
  title: 'Span with links',
  start: 0,
  end: 1000,
  attributes: {},
  type: ModelSpanType.FUNCTION,
  assessments: [],
  traceId: 'tr-current',
  ...overrides,
});

describe('ModelTraceExplorerLinksTab', () => {
  afterEach(() => {
    mockClipboardCopy.mockClear();
    mockUseSpanLinkHrefs.mockImplementation((traceIds: string[]) => {
      const hrefs: Record<string, string> = {};
      for (const traceId of traceIds) {
        hrefs[traceId] = `/experiments/1/traces?traceId=${traceId}`;
      }
      return hrefs;
    });
  });

  it('renders link details in simple cards', () => {
    const span = createSpan({ links: MOCK_SPAN_LINKS });

    render(<ModelTraceExplorerLinksTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('abc123de')).toBeInTheDocument();
    expect(screen.getByText('789xyz00')).toBeInTheDocument();
    expect(screen.queryByText(MOCK_SPAN_LINKS[0].trace_id)).not.toBeInTheDocument();
    expect(screen.queryByText(MOCK_SPAN_LINKS[1].trace_id)).not.toBeInTheDocument();
    expect(screen.getByText('aabbccdd')).toBeInTheDocument();
    expect(screen.getByText('11223344')).toBeInTheDocument();
    expect(screen.queryByText(MOCK_SPAN_LINKS[0].span_id)).not.toBeInTheDocument();
    expect(screen.getAllByText('Trace ID')).toHaveLength(2);
    expect(screen.getAllByText('Span ID')).toHaveLength(2);
    expect(screen.getByText('relationship')).toBeInTheDocument();
  });

  it('links to a resolved trace', () => {
    const span = createSpan({ links: [MOCK_SPAN_LINKS[0]] });

    render(<ModelTraceExplorerLinksTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByRole('link', { name: 'Jump to linked span' })).toHaveAttribute(
      'href',
      `#/experiments/1/traces?traceId=${MOCK_SPAN_LINKS[0].trace_id}`,
    );
    expect(screen.getByRole('link', { name: 'Jump to linked span' })).toHaveAttribute('target', '_blank');
  });

  it('copies the full linked trace and span IDs from their pills', async () => {
    const span = createSpan({ links: [MOCK_SPAN_LINKS[0]] });

    render(<ModelTraceExplorerLinksTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    await userEvent.click(screen.getByRole('button', { name: 'abc123de' }));
    expect(mockClipboardCopy).toHaveBeenLastCalledWith(MOCK_SPAN_LINKS[0].trace_id);

    await userEvent.click(screen.getByRole('button', { name: 'aabbccdd' }));
    expect(mockClipboardCopy).toHaveBeenLastCalledWith(MOCK_SPAN_LINKS[0].span_id);
  });

  it('selects a linked span in the current trace without opening a new tab', async () => {
    const onSelectSpan = jest.fn();
    const linkedSpanId = 'linked-span';
    const linkedSpan = createSpan({
      key: linkedSpanId,
      title: 'Retrieve order context',
      type: ModelSpanType.RETRIEVER,
    });
    const span = createSpan({
      links: [{ trace_id: 'tr-current', span_id: linkedSpanId }],
    });

    render(
      <ModelTraceExplorerLinksTab
        activeSpan={span}
        searchFilter=""
        activeMatch={null}
        onSelectSpan={onSelectSpan}
        spanNodes={{ [linkedSpanId]: linkedSpan }}
      />,
      { wrapper: Wrapper },
    );

    expect(screen.queryByText('Trace ID')).not.toBeInTheDocument();
    expect(screen.queryByText('current')).not.toBeInTheDocument();
    expect(screen.getByText('Retrieve order context')).toBeInTheDocument();
    expect(screen.queryByText(linkedSpanId)).not.toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Jump to linked span' }));

    expect(onSelectSpan).toHaveBeenCalledWith(linkedSpanId);
  });

  it('shows the first three attributes and expands the rest', async () => {
    const span = createSpan({
      links: [
        {
          trace_id: 'tr-linked',
          span_id: 'linked-span',
          attributes: { first: 1, second: 2, third: 3, fourth: 4 },
        },
      ],
    });

    render(<ModelTraceExplorerLinksTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('first')).toBeInTheDocument();
    expect(screen.getByText('third')).toBeInTheDocument();
    expect(screen.queryByText('fourth')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'See more' }));

    expect(screen.getByText('fourth')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'See less' }));
    expect(screen.queryByText('fourth')).not.toBeInTheDocument();
  });

  it('renders an unresolved trace with a disabled jump button', () => {
    mockUseSpanLinkHrefs.mockReturnValue({});
    const span = createSpan({ links: [MOCK_SPAN_LINKS[0]] });

    render(<ModelTraceExplorerLinksTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('abc123de')).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Jump to linked span' })).toBeDisabled();
  });

  it('renders nothing when a span has no links', () => {
    const { container } = render(
      <ModelTraceExplorerLinksTab activeSpan={createSpan()} searchFilter="" activeMatch={null} />,
      { wrapper: Wrapper },
    );

    expect(container).toBeEmptyDOMElement();
  });
});
