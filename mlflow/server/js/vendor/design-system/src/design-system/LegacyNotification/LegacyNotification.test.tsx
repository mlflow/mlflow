import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { NotificationInstance } from 'antd/lib/notification';
import { useState, Component } from 'react';

import { useLegacyNotification, withNotifications } from './LegacyNotification';
import type { WithNotificationsHOCProps } from './LegacyNotification';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('Legacy Notification', () => {
  it(`uses getPopupContainer`, async () => {
    const getPopupContainer = jest.fn().mockImplementation(() => document.body);

    function TestNotificationTrigger() {
      const [notificationAPI, contextHolder] = useLegacyNotification();

      return (
        <>
          {contextHolder}
          <button
            onClick={() => {
              notificationAPI.error({ message: 'test message' });
            }}
          >
            Trigger notification
          </button>
        </>
      );
    }

    // Arrange
    render(
      <DesignSystemProvider getPopupContainer={getPopupContainer}>
        <TestNotificationTrigger />
      </DesignSystemProvider>,
    );

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(getPopupContainer).toHaveBeenCalled();
  });

  it(`notificationAPI is stable`, async () => {
    const notificationInstances: NotificationInstance[] = [];

    function TestNotificationTrigger() {
      const [notificationAPI, contextHolder] = useLegacyNotification();
      const [, forceRender] = useState(0);

      notificationInstances.push(notificationAPI);

      return (
        <>
          {contextHolder}
          <button onClick={() => forceRender(1)}>Trigger notification</button>
        </>
      );
    }

    // Arrange
    render(
      <DesignSystemProvider>
        <TestNotificationTrigger />
      </DesignSystemProvider>,
    );

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(notificationInstances.length).toBeGreaterThan(1);
    expect(notificationInstances[0]).toBe(notificationInstances[1]);
  });

  it(`withNotifications injects props and behaves correctly`, async () => {
    class TestNotificationHocImpl extends Component<WithNotificationsHOCProps<{ messageToShow: string }>> {
      render() {
        const { messageToShow, notificationAPI, notificationContextHolder } = this.props;
        return (
          <>
            <button onClick={() => notificationAPI.open({ message: messageToShow })}>Trigger notification</button>
            {notificationContextHolder}
          </>
        );
      }
    }

    const TestNotificationHoc = withNotifications(TestNotificationHocImpl);

    // Arrange
    const wrapper = render(
      <DesignSystemProvider>
        <TestNotificationHoc messageToShow="hello there" />
      </DesignSystemProvider>,
    );

    // Assert
    expect(wrapper.queryByText('hello there')).toBeFalsy();

    // Act
    await userEvent.click(screen.getByRole('button'));

    // Assert
    expect(wrapper.queryByText('hello there')).toBeTruthy();
  });
});
