import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';
import { TraceViewEditToolbar } from './TraceViewEditToolbar';
import type { SpanRange } from '../hooks/useTraceViews';

const makeRange = (label: string): SpanRange => ({
  from_selector: { span_id: 'span-a' },
  label,
  description: '',
  position: 0,
});

const Wrapper = ({ children }: { children: React.ReactNode }) => {
  const [queryClient] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={queryClient}>
      <IntlProvider locale="en">
        <DesignSystemProvider>{children}</DesignSystemProvider>
      </IntlProvider>
    </QueryClientProvider>
  );
};

describe('TraceViewEditToolbar', () => {
  it('renders name input, cancel, and save buttons', () => {
    render(
      <TraceViewEditToolbar
        name=""
        onNameChange={jest.fn()}
        ranges={[]}
        onCancel={jest.fn()}
        onSave={jest.fn()}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByPlaceholderText('View name')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
    expect(screen.getByText('Save')).toBeInTheDocument();
  });

  it('save button is disabled when name is empty', () => {
    render(
      <TraceViewEditToolbar
        name=""
        onNameChange={jest.fn()}
        ranges={[makeRange('R1')]}
        onCancel={jest.fn()}
        onSave={jest.fn()}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByText('Save').closest('button')).toBeDisabled();
  });

  it('save button is disabled when no ranges exist', () => {
    render(
      <TraceViewEditToolbar
        name="My View"
        onNameChange={jest.fn()}
        ranges={[]}
        onCancel={jest.fn()}
        onSave={jest.fn()}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    expect(screen.getByText('Save').closest('button')).toBeDisabled();
  });

  it('calls onNameChange when name input changes', async () => {
    const onNameChange = jest.fn();
    render(
      <TraceViewEditToolbar
        name=""
        onNameChange={onNameChange}
        ranges={[]}
        onCancel={jest.fn()}
        onSave={jest.fn()}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    await userEvent.type(screen.getByPlaceholderText('View name'), 'Test');
    expect(onNameChange).toHaveBeenCalled();
  });

  it('calls onCancel when cancel is clicked', async () => {
    const onCancel = jest.fn();
    render(
      <TraceViewEditToolbar
        name=""
        onNameChange={jest.fn()}
        ranges={[]}
        onCancel={onCancel}
        onSave={jest.fn()}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    await userEvent.click(screen.getByText('Cancel'));
    expect(onCancel).toHaveBeenCalled();
  });

  it('calls onSave when save is clicked with valid state', async () => {
    const onSave = jest.fn();
    render(
      <TraceViewEditToolbar
        name="My View"
        onNameChange={jest.fn()}
        ranges={[makeRange('R1')]}
        onCancel={jest.fn()}
        onSave={onSave}
        isSaving={false}
      />,
      { wrapper: Wrapper },
    );
    await userEvent.click(screen.getByText('Save'));
    expect(onSave).toHaveBeenCalled();
  });
});
