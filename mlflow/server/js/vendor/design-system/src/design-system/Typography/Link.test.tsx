import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Link } from './Link';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Link', () => {
  it('handles clicks', async () => {
    // Arrange
    const handleClick = jest.fn();
    render(
      <Link componentId="TEST_LINK" onClick={handleClick}>
        LINK HERE
      </Link>,
    );
    expect(handleClick).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByText('LINK HERE'));

    // Assert
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('handles clicks with DesignSystemEventProvider', async () => {
    // Arrange
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Link componentId="TEST_LINK" onClick={handleClick}>
          LINK HERE
        </Link>
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByText('LINK HERE'));

    // Assert
    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'TEST_LINK',
      componentType: 'typography_link',
      shouldStartInteraction: false,
      value: undefined,
      event: expect.any(Object),
      isInteractionSubject: undefined,
    });
  });
});
