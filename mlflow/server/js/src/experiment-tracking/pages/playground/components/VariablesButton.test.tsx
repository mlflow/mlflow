import { jest, describe, it, expect } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { render, screen } from '@testing-library/react';
import userEventGlobal from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ChatMessage } from '../types';
import { VariablesButton } from './VariablesButton';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const renderButton = (messages: ChatMessage[], value: Record<string, string> = {}) => {
  const onChange = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <VariablesButton messages={messages} value={value} onChange={onChange} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange };
};

describe('VariablesButton', () => {
  it('shows just the bare label when no variables are detected', () => {
    renderButton([{ role: 'user', content: 'no placeholders' }]);
    expect(screen.getByRole('button', { name: /open variable values/i })).toHaveTextContent(/^Variables$/);
  });

  it('appends the variable count when at least one is detected', () => {
    renderButton([{ role: 'user', content: '{{ topic }} and {{ tone }}' }]);
    expect(screen.getByRole('button', { name: /open variable values/i })).toHaveTextContent(/Variables \(2\)/);
  });

  it('opens the drawer with the help paragraph and per-variable inputs', async () => {
    renderButton([{ role: 'user', content: 'Discuss {{ topic }}' }]);
    await userEvent.click(screen.getByRole('button', { name: /open variable values/i }));
    expect(screen.getByText(/create reusable variables/i)).toBeInTheDocument();
    expect(screen.getByLabelText('topic')).toBeInTheDocument();
  });
});
