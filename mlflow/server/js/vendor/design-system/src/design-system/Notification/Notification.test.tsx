import { type NotificationType } from '@databricks/design-system';

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { Notification } from '.';
import { Button } from '../Button';
import { DesignSystemEventProvider, DesignSystemEventProviderComponentSubTypes } from '../DesignSystemEventProvider';

const BasicExample = (props: {
  eventCallback: () => void;
  handleOpenChange: () => void;
  type: NotificationType;
}): JSX.Element => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const handleClick = () => {
    props.handleOpenChange();
    setIsOpen((open) => !open);
  };

  return (
    <Notification.Provider>
      <DesignSystemEventProvider callback={props.eventCallback}>
        <Notification.Root
          componentId="codegen_design-system_src_design-system_notification_notification.test.tsx_20"
          onOpenChange={() => handleClick()}
          open={isOpen}
          severity={props.type}
          duration={3000}
          forceMount={true}
        >
          <Notification.Title>Info title notification</Notification.Title>
          <Notification.Description>
            This is the content of the notification. This is the content of the notification. This is the content of the
            notification.
          </Notification.Description>
          <Notification.Close componentId="INFO_CLOSE_CLICK" aria-label="My close icon for info" />
        </Notification.Root>
        <div style={{ display: 'flex', gap: 8 }}>
          <Button componentId="INFO_OPEN_CLICK" onClick={() => handleClick()}>
            Show info notification
          </Button>
        </div>
      </DesignSystemEventProvider>
      <Notification.Viewport />
    </Notification.Provider>
  );
};

describe('Notification', () => {
  window.HTMLElement.prototype.hasPointerCapture = jest.fn();
  const eventCallback = jest.fn();
  const handleOpenChange = jest.fn();

  it('can render and close', async () => {
    render(<BasicExample eventCallback={eventCallback} handleOpenChange={handleOpenChange} type="info" />);

    await userEvent.click(screen.getByText('Show info notification'));
    expect(handleOpenChange).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'INFO_OPEN_CLICK',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
    await userEvent.click(screen.getByLabelText('My close icon for info'));
    expect(handleOpenChange).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onClick',
      componentId: 'INFO_CLOSE_CLICK',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it.each([
    { type: 'error', componentSubType: DesignSystemEventProviderComponentSubTypes.Error },
    { type: 'info', componentSubType: DesignSystemEventProviderComponentSubTypes.Info },
    { type: 'success', componentSubType: DesignSystemEventProviderComponentSubTypes.Success },
    { type: 'warning', componentSubType: DesignSystemEventProviderComponentSubTypes.Warning },
  ])(
    'callback for %s Notifications is being recorded and sub type is being passed through',
    async ({ type, componentSubType }) => {
      (window as any).IntersectionObserver = undefined;
      const mockUseOnEventCallback = jest.fn();

      render(
        <BasicExample
          eventCallback={mockUseOnEventCallback}
          handleOpenChange={handleOpenChange}
          type={type as NotificationType}
        />,
      );

      await userEvent.click(screen.getByText('Show info notification'));

      const onViewCall = mockUseOnEventCallback.mock.calls.find(
        ([event]) => event.eventType === 'onView' && event.componentType === 'notification',
      );

      expect(onViewCall).toHaveLength(1);
      expect(onViewCall[0]).toMatchObject({
        eventType: 'onView',
        componentId: `codegen_design-system_src_design-system_notification_notification.test.tsx_20`,
        componentType: 'notification',
        componentSubType,
        shouldStartInteraction: false,
        value: undefined,
      });
    },
  );
});
