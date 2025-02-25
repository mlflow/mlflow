import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ToggleButton } from './ToggleButton';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('ToggleButton', () => {
  it('handles changes with DesignSystemEventProvider', async () => {
    // Arrange
    const handleOnPressedChange = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <ToggleButton pressed={false} onPressedChange={handleOnPressedChange} componentId="bestToggleButtonEver" />
      </DesignSystemEventProvider>,
    );
    expect(handleOnPressedChange).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(handleOnPressedChange).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentType: 'toggle_button',
      componentId: 'bestToggleButtonEver',
      shouldStartInteraction: false,
      value: true,
    });
  });
});
