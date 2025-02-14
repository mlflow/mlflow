import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Drawer } from '.';
import type { DrawerContentProps } from './Drawer';
import { DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('Drawer Analytics Events', () => {
  const eventCallback = jest.fn();
  const Component = () => (
    <DesignSystemProvider>
      <DesignSystemEventProvider callback={eventCallback}>
        <Drawer.Root open={true}>
          <Drawer.Trigger>
            <button>Drawer Trigger</button>
          </Drawer.Trigger>
          <Drawer.Content
            title="drawer_title"
            componentId="drawer_test"
            analyticsEvents={[DesignSystemEventProviderAnalyticsEventTypes.OnView]}
          >
            <div>Main content goes here</div>
          </Drawer.Content>
        </Drawer.Root>
      </DesignSystemEventProvider>
    </DesignSystemProvider>
  );

  it('emits drawer content view close events', async () => {
    // Disable IntersectionObserver for useIsInViewport hook to trigger
    (window as any).IntersectionObserver = undefined;

    render(<Component />);

    expect(screen.getByText('Main content goes here')).toBeVisible();
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      componentId: 'drawer_test',
      componentType: 'drawer_content',
      eventType: 'onView',
      shouldStartInteraction: false,
      value: undefined,
    });

    const closeButton = screen.getByRole('button', { name: 'Close' });
    await userEvent.click(closeButton);

    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      componentId: 'drawer_test.close',
      componentType: 'button',
      eventType: 'onClick',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      event: expect.anything(),
    });
  });
});

describe('Drawer Close Button', () => {
  const TRIGGER_TEXT = 'Open drawer';
  const MAIN_CONTEXT_TEXT = 'Main content goes here';

  const Component = ({ onCloseClick }: { onCloseClick: DrawerContentProps['onCloseClick'] }) => (
    <DesignSystemProvider>
      <Drawer.Root open>
        <Drawer.Trigger>{TRIGGER_TEXT}</Drawer.Trigger>
        <Drawer.Content title="drawer_title" componentId="drawer_test" onCloseClick={onCloseClick}>
          <div>{MAIN_CONTEXT_TEXT}</div>
        </Drawer.Content>
      </Drawer.Root>
    </DesignSystemProvider>
  );

  it('calls the onClick callback if provided', async () => {
    const onCloseClick = jest.fn();
    render(<Component onCloseClick={onCloseClick} />);

    const closeButton = screen.getByRole('button', { name: 'Close' });
    await userEvent.click(closeButton);

    expect(onCloseClick).toHaveBeenCalled();
  });
});
