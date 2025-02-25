import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Button } from './Button';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Button', () => {
  it('handles clicks', async () => {
    // Arrange
    const handleClick = jest.fn();
    render(
      <Button componentId="codegen_design-system_src_design-system_button_button.test.tsx_11" onClick={handleClick} />,
    );
    expect(handleClick).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('handles clicks with DesignSystemEventProvider', async () => {
    // Arrange
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Button componentId="bestButtonEver" onClick={handleClick} />
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'bestButtonEver',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
  });
});
