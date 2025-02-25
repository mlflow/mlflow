import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SplitButton } from './SplitButton';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider/DesignSystemProvider';
import { DropdownMenu } from '../DropdownMenu';

describe('SplitButton', () => {
  it('handles clicks', async () => {
    const handlePrimaryClick = jest.fn();
    const handleItemClick = jest.fn();
    const menu = (
      <DropdownMenu.Content>
        <DropdownMenu.Item componentId="SPLIT_BUTTON_HERE.OPTION_1" onClick={handleItemClick}>
          Option 1
        </DropdownMenu.Item>
        <DropdownMenu.Item componentId="SPLIT_BUTTON_HERE.OPTION_2">Option 2</DropdownMenu.Item>
      </DropdownMenu.Content>
    );
    render(
      <DesignSystemProvider>
        <SplitButton
          componentId="SPLIT_BUTTON_HERE"
          onClick={handlePrimaryClick}
          menu={menu}
          menuButtonLabel="MENU_BUTTON"
        >
          CLICK ME
        </SplitButton>
      </DesignSystemProvider>,
    );

    await userEvent.click(screen.getByText('CLICK ME'));
    expect(handlePrimaryClick).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByLabelText('MENU_BUTTON'));
    await userEvent.click(screen.getByText('Option 1'));
    expect(handleItemClick).toHaveBeenCalledTimes(1);
  });

  it('emits analytics events', async () => {
    const handlePrimaryClick = jest.fn();
    const handleItemClick = jest.fn();
    const eventCallback = jest.fn();
    const menu = (
      <DropdownMenu.Content>
        <DropdownMenu.Item componentId="SPLIT_BUTTON_HERE.OPTION_1" onClick={handleItemClick}>
          Option 1
        </DropdownMenu.Item>
        <DropdownMenu.Item componentId="SPLIT_BUTTON_HERE.OPTION_2">Option 2</DropdownMenu.Item>
      </DropdownMenu.Content>
    );
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <SplitButton
            componentId="SPLIT_BUTTON_HERE"
            onClick={handlePrimaryClick}
            menu={menu}
            menuButtonLabel="MENU_BUTTON"
          >
            CLICK ME
          </SplitButton>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );

    await userEvent.click(screen.getByText('CLICK ME'));
    expect(handlePrimaryClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'SPLIT_BUTTON_HERE.primary_button',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });

    await userEvent.click(screen.getByLabelText('MENU_BUTTON'));
    await userEvent.click(screen.getByText('Option 1'));
    expect(handleItemClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onClick',
      componentId: 'SPLIT_BUTTON_HERE.OPTION_1',
      componentType: 'dropdown_menu_item',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.any(Object),
      isInteractionSubject: undefined,
    });
  });
});
