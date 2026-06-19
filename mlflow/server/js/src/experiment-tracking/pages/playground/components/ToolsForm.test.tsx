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
  toolsAdded?: boolean;
  toolChoice?: ToolChoice;
}

const renderForm = ({ value = '', error, toolsAdded = false, toolChoice = 'auto' }: RenderProps = {}) => {
  const onChange = jest.fn<(next: string) => void>();
  const onToolsAddedChange = jest.fn<(next: boolean) => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ToolsForm
          value={value}
          onChange={onChange}
          error={error}
          toolsAdded={toolsAdded}
          onToolsAddedChange={onToolsAddedChange}
          toolChoice={toolChoice}
          onToolChoiceChange={onToolChoiceChange}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange, onToolsAddedChange, onToolChoiceChange };
};

describe('ToolsForm', () => {
  it('shows the Add tools button and hides the JSON textarea before tools are added', () => {
    renderForm({ toolsAdded: false });
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByLabelText('JSON tool definitions')).not.toBeInTheDocument();
  });

  it('fires onToolsAddedChange(true) when Add tools is clicked', async () => {
    const { onToolsAddedChange } = renderForm({ toolsAdded: false });
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onToolsAddedChange).toHaveBeenLastCalledWith(true);
  });

  it('shows the JSON textarea and the Auto/Required selector once tools are added', () => {
    renderForm({ toolsAdded: true });
    expect(screen.getByLabelText('JSON tool definitions')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Auto' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Required' })).toBeInTheDocument();
  });

  it('forwards picker changes via onToolChoiceChange', async () => {
    const { onToolChoiceChange } = renderForm({ toolsAdded: true, toolChoice: 'auto' });
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    expect(onToolChoiceChange).toHaveBeenLastCalledWith('required');
  });

  it('returns to the empty state without clearing the JSON when the trash icon is clicked', async () => {
    const { onChange, onToolsAddedChange } = renderForm({ toolsAdded: true, value: '[{"x":1}]' });
    await userEvent.click(screen.getByRole('button', { name: 'Remove tools' }));
    expect(onToolsAddedChange).toHaveBeenLastCalledWith(false);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('renders an inline error when error is non-null and tools are added', () => {
    renderForm({ toolsAdded: true, value: '{not-json', error: 'Invalid JSON' });
    expect(screen.getByText('Invalid JSON')).toBeInTheDocument();
  });

  it('does not render the inline error before tools are added', () => {
    renderForm({ toolsAdded: false, error: 'Invalid JSON' });
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });

  it('shows an empty-required inline error when tools are added and the textarea is empty', () => {
    renderForm({ toolsAdded: true, value: '' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('shows the empty-required inline error when tools are added and the value parses to []', () => {
    renderForm({ toolsAdded: true, value: '[]' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('prefers the empty-required error over a stale parse-error prop when the value is empty', () => {
    renderForm({ toolsAdded: true, value: '', error: 'Invalid JSON' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });
});
