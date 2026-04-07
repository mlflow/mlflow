import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SpanRangeOverlay } from './SpanRangeOverlay';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { SpanRange } from '../hooks/useTraceViews';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const makeSpanNode = (id: string, name: string): ModelTraceSpanNode =>
  ({
    key: id,
    title: name,
    start: 0,
    end: 100,
    traceId: 'tr-001',
    type: 'CHAIN',
    attributes: {},
    assessments: [],
    inputs: { query: 'test' },
    outputs: { result: 'ok' },
  }) as any;

const nodes = [
  makeSpanNode('span-1', 'AgentExecutor'),
  makeSpanNode('span-2', 'LLM Call'),
  makeSpanNode('span-3', 'Tool: search'),
];

describe('SpanRangeOverlay', () => {
  it('renders a row for each span node with a checkbox', () => {
    render(
      <SpanRangeOverlay
        nodes={nodes}
        ranges={[]}
        onAddRange={jest.fn()}
        onRemoveRange={jest.fn()}
        onUpdateRange={jest.fn()}
      />,
      { wrapper },
    );
    expect(screen.getAllByRole('checkbox')).toHaveLength(3);
    expect(screen.getByText('AgentExecutor')).toBeInTheDocument();
    expect(screen.getByText('LLM Call')).toBeInTheDocument();
    expect(screen.getByText('Tool: search')).toBeInTheDocument();
  });

  it('highlights spans that belong to a range', () => {
    const range: SpanRange = {
      from_selector: { span_id: 'span-2' },
      to_selector: { span_id: 'span-3' },
      label: 'Range 1',
      description: '',
      position: 0,
    };
    render(
      <SpanRangeOverlay
        nodes={nodes}
        ranges={[range]}
        onAddRange={jest.fn()}
        onRemoveRange={jest.fn()}
        onUpdateRange={jest.fn()}
      />,
      { wrapper },
    );
    const checkboxes = screen.getAllByRole('checkbox');
    expect(checkboxes[0]).not.toBeChecked();
    expect(checkboxes[1]).toBeChecked();
    expect(checkboxes[2]).toBeChecked();
  });

  it('renders range badge above first span in range', () => {
    const range: SpanRange = {
      from_selector: { span_id: 'span-2' },
      label: 'Tool Calls',
      description: '',
      position: 0,
    };
    render(
      <SpanRangeOverlay
        nodes={nodes}
        ranges={[range]}
        onAddRange={jest.fn()}
        onRemoveRange={jest.fn()}
        onUpdateRange={jest.fn()}
      />,
      { wrapper },
    );
    expect(screen.getByText('Tool Calls')).toBeInTheDocument();
  });

  it('clicking checkbox on unselected span calls onAddRange', async () => {
    const onAddRange = jest.fn();
    render(
      <SpanRangeOverlay
        nodes={nodes}
        ranges={[]}
        onAddRange={onAddRange}
        onRemoveRange={jest.fn()}
        onUpdateRange={jest.fn()}
      />,
      { wrapper },
    );
    await userEvent.click(screen.getAllByRole('checkbox')[1]);
    expect(onAddRange).toHaveBeenCalledWith({ span_id: 'span-2' });
  });

  it('clicking delete on range badge calls onRemoveRange', async () => {
    const onRemoveRange = jest.fn();
    const range: SpanRange = {
      from_selector: { span_id: 'span-2' },
      label: 'Range 1',
      description: '',
      position: 0,
    };
    render(
      <SpanRangeOverlay
        nodes={nodes}
        ranges={[range]}
        onAddRange={jest.fn()}
        onRemoveRange={onRemoveRange}
        onUpdateRange={jest.fn()}
      />,
      { wrapper },
    );
    await userEvent.click(screen.getByLabelText('Delete range'));
    expect(onRemoveRange).toHaveBeenCalledWith(0);
  });
});
