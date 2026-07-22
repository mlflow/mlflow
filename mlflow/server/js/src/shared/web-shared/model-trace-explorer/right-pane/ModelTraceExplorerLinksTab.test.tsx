import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { ModelTraceExplorerLinksTab } from './ModelTraceExplorerLinksTab';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { MOCK_LINKS_SPAN, MOCK_SPAN_LINKS } from '../ModelTraceExplorer.test-utils';

const mockUseSpanLinkHref = jest.fn(
  (traceId: string | undefined) =>
    traceId ? `/experiments/1/traces?selectedEvaluationId=${traceId}` : undefined,
);

jest.mock('../hooks/useSpanLinkHref', () => ({
  useSpanLinkHref: (...args: any[]) => mockUseSpanLinkHref(...args),
}));

const queryClient = new QueryClient();

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>{children}</MemoryRouter>
      </QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

const createMockSpanNode = (overrides: Partial<ModelTraceSpanNode> = {}): ModelTraceSpanNode => ({
  key: 'test-span',
  title: 'Test Span',
  start: 0,
  end: 1000,
  inputs: undefined,
  outputs: undefined,
  attributes: {},
  type: ModelSpanType.FUNCTION,
  assessments: [],
  traceId: 'tr-test',
  ...overrides,
});

describe('ModelTraceExplorerLinksTab', () => {
  it('renders empty state when span has no links', () => {
    const span = createMockSpanNode();
    render(<ModelTraceExplorerLinksTab activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('No links found')).toBeInTheDocument();
  });

  it('renders empty state when links array is empty', () => {
    const span = createMockSpanNode({ links: [] });
    render(<ModelTraceExplorerLinksTab activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('No links found')).toBeInTheDocument();
  });

  it('renders link entries with trace_id and span_id', () => {
    render(<ModelTraceExplorerLinksTab activeSpan={MOCK_LINKS_SPAN} />, { wrapper: Wrapper });

    expect(screen.getByText(MOCK_SPAN_LINKS[0].trace_id)).toBeInTheDocument();
    expect(screen.getByText(MOCK_SPAN_LINKS[1].trace_id)).toBeInTheDocument();
  });

  it('renders navigation links for each span link', () => {
    render(<ModelTraceExplorerLinksTab activeSpan={MOCK_LINKS_SPAN} />, { wrapper: Wrapper });

    const navLinks = screen.getAllByRole('link');
    expect(navLinks).toHaveLength(2);
    expect(navLinks[0]).toHaveAttribute(
      'href',
      `/experiments/1/traces?selectedEvaluationId=${MOCK_SPAN_LINKS[0].trace_id}`,
    );
  });

  it('renders link attributes when present', () => {
    const span = createMockSpanNode({
      links: [
        {
          trace_id: 'tr-with-attrs',
          span_id: 'aabb000000000000',
          attributes: { relationship: 'caused_by', priority: 'high' },
        },
      ],
    });
    render(<ModelTraceExplorerLinksTab activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('tr-with-attrs')).toBeInTheDocument();
    expect(screen.getByText('attributes')).toBeInTheDocument();
  });

  it('renders trace_id as plain text when href is unavailable', () => {
    mockUseSpanLinkHref.mockReturnValue(undefined);
    const span = createMockSpanNode({
      links: [{ trace_id: 'tr-unresolved', span_id: 'eeff000000000000' }],
    });
    render(<ModelTraceExplorerLinksTab activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('tr-unresolved')).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    mockUseSpanLinkHref.mockRestore();
  });

  it('does not render attributes section when link has no attributes', () => {
    const span = createMockSpanNode({
      links: [{ trace_id: 'tr-no-attrs', span_id: 'ccdd000000000000' }],
    });
    render(<ModelTraceExplorerLinksTab activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('tr-no-attrs')).toBeInTheDocument();
    expect(screen.queryByText('attributes')).not.toBeInTheDocument();
  });
});
