import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ConversationMessage } from '../types';
import { PromptInputPanel } from './PromptInputPanel';

const StatefulPanel = ({ initialMessages }: { initialMessages: ConversationMessage[] }) => {
  const [messages, setMessages] = useState(initialMessages);
  return <PromptInputPanel messages={messages} onChange={setMessages} />;
};

const renderPanel = (initialMessages: ConversationMessage[]) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <StatefulPanel initialMessages={initialMessages} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

const selectRole = async (role: string) => {
  await userEvent.click(screen.getByRole('button', { name: 'Change message role' }));
  await userEvent.click(screen.getByRole('menuitemradio', { name: role }));
};

describe('PromptInputPanel', () => {
  it('keeps the role selector after a message is switched to assistant, so it can be switched back', async () => {
    renderPanel([{ role: 'user', content: 'Paris is the capital of France.' }]);

    await selectRole('Assistant');
    expect(screen.getByRole('button', { name: 'Change message role' })).toHaveTextContent('Assistant');
    expect(screen.getByDisplayValue('Paris is the capital of France.')).toBeInTheDocument();

    await selectRole('System');
    expect(screen.getByRole('button', { name: 'Change message role' })).toHaveTextContent('System');
  });

  it('lets the user write an example response in an assistant message', async () => {
    renderPanel([{ role: 'assistant', content: '' }]);

    await userEvent.type(screen.getByPlaceholderText('Type a message'), 'Sure, here is an example.');

    expect(screen.getByDisplayValue('Sure, here is an example.')).toBeInTheDocument();
  });

  it('renders a model reply read-only, without the role selector', () => {
    renderPanel([{ role: 'assistant', content: 'Hello from the model', isModelReply: true }]);

    expect(screen.getByTestId('mlflow.playground.assistant.text_card')).toHaveTextContent('Hello from the model');
    expect(screen.queryByRole('button', { name: 'Change message role' })).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Type a message')).not.toBeInTheDocument();
  });
});
