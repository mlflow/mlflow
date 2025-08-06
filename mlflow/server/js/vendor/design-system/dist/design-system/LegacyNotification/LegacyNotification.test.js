import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState, Component } from 'react';
import { useLegacyNotification, withNotifications } from './LegacyNotification';
import { DesignSystemProvider } from '../DesignSystemProvider';
describe('Legacy Notification', () => {
    it(`uses getPopupContainer`, async () => {
        const getPopupContainer = jest.fn(() => document.body);
        function TestNotificationTrigger() {
            const [notificationAPI, contextHolder] = useLegacyNotification();
            return (_jsxs(_Fragment, { children: [contextHolder, _jsx("button", { onClick: () => {
                            notificationAPI.error({ message: 'test message' });
                        }, children: "Trigger notification" })] }));
        }
        // Arrange
        render(_jsx(DesignSystemProvider, { getPopupContainer: getPopupContainer, children: _jsx(TestNotificationTrigger, {}) }));
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(getPopupContainer).toHaveBeenCalled();
    });
    it(`notificationAPI is stable`, async () => {
        const notificationInstances = [];
        function TestNotificationTrigger() {
            const [notificationAPI, contextHolder] = useLegacyNotification();
            const [, forceRender] = useState(0);
            notificationInstances.push(notificationAPI);
            return (_jsxs(_Fragment, { children: [contextHolder, _jsx("button", { onClick: () => forceRender(1), children: "Trigger notification" })] }));
        }
        // Arrange
        render(_jsx(DesignSystemProvider, { children: _jsx(TestNotificationTrigger, {}) }));
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(notificationInstances.length).toBeGreaterThan(1);
        expect(notificationInstances[0]).toBe(notificationInstances[1]);
    });
    it(`withNotifications injects props and behaves correctly`, async () => {
        class TestNotificationHocImpl extends Component {
            render() {
                const { messageToShow, notificationAPI, notificationContextHolder } = this.props;
                return (_jsxs(_Fragment, { children: [_jsx("button", { onClick: () => notificationAPI.open({ message: messageToShow }), children: "Trigger notification" }), notificationContextHolder] }));
            }
        }
        const TestNotificationHoc = withNotifications(TestNotificationHocImpl);
        // Arrange
        const wrapper = render(_jsx(DesignSystemProvider, { children: _jsx(TestNotificationHoc, { messageToShow: "hello there" }) }));
        // Assert
        expect(wrapper.queryByText('hello there')).toBeFalsy();
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(wrapper.queryByText('hello there')).toBeTruthy();
    });
});
//# sourceMappingURL=LegacyNotification.test.js.map