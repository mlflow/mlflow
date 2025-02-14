import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Form } from './Form';
import { DesignSystemEventProvider } from '../../design-system';

describe('Form', () => {
  it('renders the form and handles submission', async () => {
    // Arrange
    const handleSubmit = jest.fn().mockResolvedValue(undefined);
    render(
      <Form componentId="testForm" onSubmit={handleSubmit}>
        <button type="submit">Submit</button>
      </Form>,
    );

    // Act
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    // Assert
    expect(handleSubmit).toHaveBeenCalledTimes(1);
  });

  it('calls the DesignSystemEventProvider onSubmit callback', async () => {
    // Arrange
    const handleSubmit = jest.fn().mockResolvedValue(undefined);
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Form componentId="eventForm" onSubmit={handleSubmit}>
          <button type="submit">Submit</button>
        </Form>
      </DesignSystemEventProvider>,
    );

    // Act
    await userEvent.click(screen.getByRole('button', { name: /submit/i }));

    // Assert
    expect(handleSubmit).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onSubmit',
      componentId: 'eventForm',
      componentType: 'form',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('throws error if nested form components are rendered', () => {
    // Arrange
    const handleSubmit = jest.fn().mockResolvedValue(undefined);

    // Assert
    expect(() => {
      render(
        <Form componentId="outerForm" onSubmit={handleSubmit}>
          <Form componentId="innerForm" onSubmit={handleSubmit}>
            <button type="submit">Submit</button>
          </Form>
        </Form>,
      );
    }).toThrowError('DuBois Form component cannot be nested');
  });
});
