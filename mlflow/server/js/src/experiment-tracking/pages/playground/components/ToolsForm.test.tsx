import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ToolChoice } from '../types';
import { ToolsForm } from './ToolsForm';

interface RenderProps {
  value?: string;
  error?: string | null;
  toolAdded?: boolean;
  toolChoice?: ToolChoice;
}

const renderForm = ({ value = '', error, toolAdded = false, toolChoice = 'auto' }: RenderProps = {}) => {
  const onChange = jest.fn<(next: string) => void>();
  const onAddTool = jest.fn<() => void>();
  const onRemoveTool = jest.fn<() => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ToolsForm
          value={value}
          onChange={onChange}
          error={error}
          toolAdded={toolAdded}
          onAddTool={onAddTool}
          onRemoveTool={onRemoveTool}
          toolChoice={toolChoice}
          onToolChoiceChange={onToolChoiceChange}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange, onAddTool, onRemoveTool, onToolChoiceChange };
};

describe('ToolsForm', () => {
  it('shows only the Add tools button in the default (no tools added) state', () => {
    renderForm({ toolAdded: false });
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByLabelText('JSON Tool Definitions')).not.toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: 'Auto' })).not.toBeInTheDocument();
  });

  it('fires onAddTool when the Add tools button is clicked', async () => {
    const { onAddTool } = renderForm({ toolAdded: false });
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onAddTool).toHaveBeenCalledTimes(1);
  });

  it('shows the JSON textarea and an Auto/Required picker once tools are added', () => {
    renderForm({ toolAdded: true, toolChoice: 'auto' });
    expect(screen.getByLabelText('JSON Tool Definitions')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Auto' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Required' })).toBeInTheDocument();
    // Once added, the Add tools affordance is gone.
    expect(screen.queryByRole('button', { name: 'Add tools' })).not.toBeInTheDocument();
  });

  it('reflects the active tool choice in the picker', () => {
    renderForm({ toolAdded: true, toolChoice: 'required' });
    expect(screen.getByRole('radio', { name: 'Required' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Auto' })).not.toBeChecked();
  });

  it('forwards picker changes via onToolChoiceChange', async () => {
    const { onToolChoiceChange } = renderForm({ toolAdded: true, toolChoice: 'auto' });
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    expect(onToolChoiceChange).toHaveBeenLastCalledWith('required');
  });

  it('fires onRemoveTool when the Remove tools button is clicked', async () => {
    const { onRemoveTool } = renderForm({ toolAdded: true });
    await userEvent.click(screen.getByRole('button', { name: 'Remove tools' }));
    expect(onRemoveTool).toHaveBeenCalledTimes(1);
  });

  it('renders the parse error when error is non-null and the value is non-empty', () => {
    renderForm({ toolAdded: true, value: '{not-json', error: 'Invalid JSON' });
    expect(screen.getByText('Invalid JSON')).toBeInTheDocument();
  });

  it('shows an empty-required inline error when tools are added and the textarea is empty', () => {
    renderForm({ toolAdded: true, value: '' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('shows the empty-required inline error when tools are added and the value parses to []', () => {
    renderForm({ toolAdded: true, value: '[]' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('prefers the empty-required error over a stale parse-error prop when the value is empty', () => {
    renderForm({ toolAdded: true, value: '', error: 'Invalid JSON' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });
});
