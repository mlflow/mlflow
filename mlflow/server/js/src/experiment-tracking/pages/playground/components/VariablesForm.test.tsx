import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ChatMessage } from '../types';
import { VariablesForm } from './VariablesForm';

const renderForm = (messages: ChatMessage[], value: Record<string, string> = {}) => {
  const onChange = jest.fn<(next: Record<string, string>) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <VariablesForm messages={messages} value={value} onChange={onChange} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange };
};

describe('VariablesForm', () => {
  it('renders nothing when no variables are detected in messages', () => {
    renderForm([{ role: 'user', content: 'no placeholders here' }]);
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  });

  it('renders one input per detected variable in first-appearance order', () => {
    renderForm([
      { role: 'system', content: 'Style: {{ tone }}' },
      { role: 'user', content: 'Discuss {{ topic }} in tone {{ tone }}' },
    ]);
    expect(screen.getByLabelText('tone')).toBeInTheDocument();
    expect(screen.getByLabelText('topic')).toBeInTheDocument();
    // De-duplicates: tone appears once even though it's in two messages.
    expect(screen.getAllByRole('textbox')).toHaveLength(2);
  });

  it('forwards typed values via onChange, preserving other entries', () => {
    const { onChange } = renderForm([{ role: 'user', content: '{{ topic }} and {{ tone }}' }], { tone: 'formal' });
    fireEvent.change(screen.getByLabelText('topic'), { target: { value: 'Hello world' } });
    expect(onChange).toHaveBeenLastCalledWith({ tone: 'formal', topic: 'Hello world' });
  });
});
