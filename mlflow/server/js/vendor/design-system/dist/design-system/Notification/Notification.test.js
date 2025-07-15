import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Notification } from '.';
import { Button } from '../Button';
import { DesignSystemEventProviderComponentSubTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
const BasicExample = (props) => {
    const [isOpen, setIsOpen] = useState(false);
    const handleClick = () => {
        props.handleOpenChange();
        setIsOpen((open) => !open);
    };
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(props.eventCallback);
    return (_jsxs(Notification.Provider, { children: [_jsxs(DesignSystemEventProviderForTest, { children: [_jsxs(Notification.Root, { componentId: "codegen_design-system_src_design-system_notification_notification.test.tsx_20", onOpenChange: () => handleClick(), open: isOpen, severity: props.type, duration: 3000, forceMount: true, children: [_jsx(Notification.Title, { children: "Info title notification" }), _jsx(Notification.Description, { children: "This is the content of the notification. This is the content of the notification. This is the content of the notification." }), _jsx(Notification.Close, { componentId: "INFO_CLOSE_CLICK", "aria-label": "My close icon for info" })] }), _jsx("div", { style: { display: 'flex', gap: 8 }, children: _jsx(Button, { componentId: "INFO_OPEN_CLICK", onClick: () => handleClick(), children: "Show info notification" }) })] }), _jsx(Notification.Viewport, {})] }));
};
describe('Notification', () => {
    window.HTMLElement.prototype.hasPointerCapture = jest.fn();
    const eventCallback = jest.fn();
    const handleOpenChange = jest.fn();
    it('can render and close', async () => {
        render(_jsx(BasicExample, { eventCallback: eventCallback, handleOpenChange: handleOpenChange, type: "info" }));
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
    ])('callback for %s Notifications is being recorded and sub type is being passed through', async ({ type, componentSubType }) => {
        window.IntersectionObserver = undefined;
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(BasicExample, { eventCallback: mockUseOnEventCallback, handleOpenChange: handleOpenChange, type: type }));
        await userEvent.click(screen.getByText('Show info notification'));
        const onViewCall = mockUseOnEventCallback.mock.calls.find(
        // @ts-expect-error TODO(FEINF-1796)
        ([event]) => event.eventType === 'onView' && event.componentType === 'notification');
        expect(onViewCall).toHaveLength(1);
        expect(onViewCall[0]).toMatchObject({
            eventType: 'onView',
            componentId: `codegen_design-system_src_design-system_notification_notification.test.tsx_20`,
            componentType: 'notification',
            componentSubType,
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=Notification.test.js.map