import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ModelTraceExplorerAttributesTab } from './ModelTraceExplorerAttributesTab';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import { BrowserRouter } from '../RoutingUtils';

jest.mock('../hooks/useGatewayTraceLink', () => ({
  useGatewayTraceLink: (traceId: string | undefined) => {
    if (traceId === 'gw-trace-123') {
      return '/experiments/exp-456/traces?selectedEvaluationId=gw-trace-123';
    }
    return undefined;
  },
}));

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <BrowserRouter>
    <IntlProvider locale="en">
      <DesignSystemProvider>{children}</DesignSystemProvider>
    </IntlProvider>
  </BrowserRouter>
);

const createMockSpanNode = (overrides: Partial<ModelTraceSpanNode> = {}): ModelTraceSpanNode => ({
  key: 'test-span',
  title: 'Test Span',
  start: 0,
  end: 1000,
  inputs: undefined,
  outputs: undefined,
  attributes: {},
  type: ModelSpanType.LLM,
  assessments: [],
  traceId: 'test-trace-id',
  ...overrides,
});

describe('ModelTraceExplorerAttributesTab', () => {
  it('renders gateway trace link when linkedTraceId attribute is present and href resolves', () => {
    const span = createMockSpanNode({
      attributes: {
        'mlflow.gateway.linkedTraceId': 'gw-trace-123',
        endpoint_id: 'ep-789',
      },
    });
    render(<ModelTraceExplorerAttributesTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('View gateway trace')).toBeInTheDocument();
    const link = screen.getByText('View gateway trace').closest('a');
    expect(link).toHaveAttribute('href', expect.stringContaining('/experiments/exp-456/traces'));
    expect(link).toHaveAttribute('href', expect.stringContaining('selectedEvaluationId=gw-trace-123'));
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('falls back to code snippet when gateway trace href is not available', () => {
    const span = createMockSpanNode({
      attributes: {
        'mlflow.gateway.linkedTraceId': 'unknown-trace-id',
      },
    });
    render(<ModelTraceExplorerAttributesTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.queryByText('View gateway trace')).not.toBeInTheDocument();
    // The attribute key should still be visible as a code snippet title
    expect(screen.getByText('mlflow.gateway.linkedTraceId')).toBeInTheDocument();
  });

  it('renders other attributes normally alongside the gateway link', () => {
    const span = createMockSpanNode({
      attributes: {
        'mlflow.gateway.linkedTraceId': 'gw-trace-123',
        endpoint_id: 'ep-789',
        endpoint_name: 'my-endpoint',
      },
    });
    render(<ModelTraceExplorerAttributesTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('View gateway trace')).toBeInTheDocument();
    expect(screen.getByText('endpoint_id')).toBeInTheDocument();
    expect(screen.getByText('endpoint_name')).toBeInTheDocument();
  });

  it('renders empty state when no attributes exist', () => {
    const span = createMockSpanNode({ attributes: {} });
    render(<ModelTraceExplorerAttributesTab activeSpan={span} searchFilter="" activeMatch={null} />, {
      wrapper: Wrapper,
    });

    expect(screen.getByText('No attributes found')).toBeInTheDocument();
  });
});
