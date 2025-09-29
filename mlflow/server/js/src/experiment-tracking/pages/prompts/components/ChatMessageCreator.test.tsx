import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormProvider, useForm } from 'react-hook-form';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ChatMessageCreator } from './ChatMessageCreator';

describe('ChatMessageCreator', () => {
  const renderComponent = () => {
    const Wrapper = () => {
      const methods = useForm({ defaultValues: { messages: [{ role: 'user', content: '' }] } });
      return (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <FormProvider {...methods}>
              <ChatMessageCreator name="messages" />
            </FormProvider>
          </DesignSystemProvider>
        </IntlProvider>
      );
    };
    render(<Wrapper />);
  };

  it('allows adding and removing messages', async () => {
    renderComponent();
    expect(screen.getAllByPlaceholderText('role')).toHaveLength(1);

    await userEvent.click(screen.getAllByRole('button', { name: 'Add message' })[0]);
    expect(screen.getAllByPlaceholderText('role')).toHaveLength(2);

    await userEvent.click(screen.getAllByRole('button', { name: 'Remove message' })[1]);
    expect(screen.getAllByPlaceholderText('role')).toHaveLength(1);
  });

  it('supports custom roles', async () => {
    renderComponent();
    const input = screen.getByPlaceholderText('role');
    await userEvent.clear(input);
    await userEvent.type(input, 'wizard');
    expect(input).toHaveValue('wizard');
  });
});
