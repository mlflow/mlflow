import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Switch } from './Switch';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Switch', () => {
  it('handles changes with DesignSystemEventProvider', async () => {
    // Arrange
    const handleOnChange = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Switch componentId="bestSwitchEver" onChange={handleOnChange} />
      </DesignSystemEventProvider>,
    );
    expect(handleOnChange).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByRole('switch'));

    // Assert
    expect(handleOnChange).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'bestSwitchEver',
      componentType: 'switch',
      shouldStartInteraction: false,
      value: true,
    });
  });
});
