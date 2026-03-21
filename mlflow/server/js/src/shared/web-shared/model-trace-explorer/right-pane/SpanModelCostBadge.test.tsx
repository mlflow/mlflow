import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { SpanModelCostBadge } from './SpanModelCostBadge';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
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
  type: ModelSpanType.LLM,
  assessments: [],
  traceId: 'test-trace-id',
  ...overrides,
});

describe('SpanModelCostBadge', () => {
  it('renders nothing when there is no model or cost', () => {
    const span = createMockSpanNode();
    const { container } = render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });
    expect(container.firstChild).toBeNull();
  });

  it('renders model name when modelName is provided', () => {
    const span = createMockSpanNode({ modelName: 'gpt-4o-mini' });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('gpt-4o-mini')).toBeInTheDocument();
  });

  it('renders cost when cost is provided', () => {
    const span = createMockSpanNode({
      cost: {
        input_cost: 0.001,
        output_cost: 0.002,
        total_cost: 0.003,
      },
    });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('Cost')).toBeInTheDocument();
    expect(screen.getByText('$0.003')).toBeInTheDocument();
  });

  it('renders both model and cost when both are provided', () => {
    const span = createMockSpanNode({
      modelName: 'claude-3-sonnet',
      cost: {
        input_cost: 0.0005,
        output_cost: 0.0015,
        total_cost: 0.002,
      },
    });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('claude-3-sonnet')).toBeInTheDocument();
    expect(screen.getByText('Cost')).toBeInTheDocument();
    expect(screen.getByText('$0.002')).toBeInTheDocument();
  });

  it('renders only model when cost is not provided', () => {
    const span = createMockSpanNode({ modelName: 'gpt-4' });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('gpt-4')).toBeInTheDocument();
    expect(screen.queryByText('Cost')).not.toBeInTheDocument();
  });

  it('renders only cost when model is not provided', () => {
    const span = createMockSpanNode({
      cost: {
        input_cost: 0.001,
        output_cost: 0.002,
        total_cost: 0.003,
      },
    });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    expect(screen.queryByText('Model')).not.toBeInTheDocument();
    expect(screen.getByText('Cost')).toBeInTheDocument();
    expect(screen.getByText('$0.003')).toBeInTheDocument();
  });

  it('shows cost breakdown on hover', async () => {
    const span = createMockSpanNode({
      cost: {
        input_cost: 0.001234,
        output_cost: 0.002345,
        total_cost: 0.003579,
      },
    });
    render(<SpanModelCostBadge activeSpan={span} />, { wrapper: Wrapper });

    // Hover over the cost tag to trigger the hover card
    const costTrigger = screen.getByText('$0.003579');
    await userEvent.hover(costTrigger);

    // Check that the breakdown is shown
    expect(await screen.findByText('Cost breakdown')).toBeInTheDocument();
    expect(screen.getByText('Input cost')).toBeInTheDocument();
    expect(screen.getByText('Output cost')).toBeInTheDocument();
    expect(screen.getByText('Total')).toBeInTheDocument();
    expect(screen.getByText('$0.001234')).toBeInTheDocument();
    expect(screen.getByText('$0.002345')).toBeInTheDocument();
  });
});
