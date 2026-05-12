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
  toolChoice?: ToolChoice;
}

const renderForm = ({ value = '', error, toolChoice = 'none' }: RenderProps = {}) => {
  const onChange = jest.fn<(next: string) => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ToolsForm
          value={value}
          onChange={onChange}
          error={error}
          toolChoice={toolChoice}
          onToolChoiceChange={onToolChoiceChange}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange, onToolChoiceChange };
};

describe('ToolsForm', () => {
  it('hides the JSON textarea when toolChoice is none', () => {
    renderForm({ toolChoice: 'none' });
    expect(screen.queryByLabelText('JSON Tool Definition')).not.toBeInTheDocument();
  });

  it('shows the JSON textarea when toolChoice is auto or required', () => {
    renderForm({ toolChoice: 'auto' });
    expect(screen.getByLabelText('JSON Tool Definition')).toBeInTheDocument();
  });

  it('forwards picker changes via onToolChoiceChange', async () => {
    const { onToolChoiceChange } = renderForm({ toolChoice: 'none' });
    await userEvent.click(screen.getByRole('radio', { name: 'Auto' }));
    expect(onToolChoiceChange).toHaveBeenLastCalledWith('auto');
  });

  it('renders an inline error when error is non-null and toolChoice is not none', () => {
    renderForm({ toolChoice: 'auto', value: '{not-json', error: 'Invalid JSON' });
    expect(screen.getByText('Invalid JSON')).toBeInTheDocument();
  });

  it('does not render the inline error when toolChoice is none', () => {
    renderForm({ toolChoice: 'none', error: 'Invalid JSON' });
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });

  it('shows an empty-required inline error when toolChoice is not none and the textarea is empty', () => {
    renderForm({ toolChoice: 'auto', value: '' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('shows the empty-required inline error when toolChoice is not none and the value parses to []', () => {
    renderForm({ toolChoice: 'required', value: '[]' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
  });

  it('prefers the empty-required error over a stale parse-error prop when the value is empty', () => {
    renderForm({ toolChoice: 'auto', value: '', error: 'Invalid JSON' });
    expect(screen.getByText('Add at least one tool definition')).toBeInTheDocument();
    expect(screen.queryByText('Invalid JSON')).not.toBeInTheDocument();
  });
});
