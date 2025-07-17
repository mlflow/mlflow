import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Drawer } from '.';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Drawer Analytics Events', () => {
    const { setSafex } = setupSafexTesting();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = () => (_jsx(DesignSystemProvider, { children: _jsx(DesignSystemEventProviderForTest, { children: _jsxs(Drawer.Root, { open: true, children: [_jsx(Drawer.Trigger, { children: _jsx("button", { children: "Drawer Trigger" }) }), _jsx(Drawer.Content, { title: "drawer_title", componentId: "drawer_test", analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView], children: _jsx("div", { children: "Main content goes here" }) })] }) }) }));
    describe('disabled defaultButtonComponentView', () => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultButtonComponentView': false,
            });
        });
        it('emits drawer content view close events', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            render(_jsx(Component, {}));
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
    describe('enabled defaultButtonComponentView', () => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultButtonComponentView': true,
            });
        });
        it('emits drawer content view close events', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            render(_jsx(Component, {}));
            expect(screen.getByText('Main content goes here')).toBeVisible();
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenCalledWith({
                componentId: 'drawer_test.close',
                componentType: 'button',
                eventType: 'onView',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                componentId: 'drawer_test',
                componentType: 'drawer_content',
                eventType: 'onView',
                shouldStartInteraction: false,
                value: undefined,
            });
            const closeButton = screen.getByRole('button', { name: 'Close' });
            await userEvent.click(closeButton);
            expect(eventCallback).toHaveBeenCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                componentId: 'drawer_test.close',
                componentType: 'button',
                eventType: 'onClick',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                event: expect.anything(),
            });
        });
    });
});
describe.each([false, true])('Shared Drawer tests: databricks.fe.observability.defaultButtonComponentView set to %s', (defaultButtonComponentView) => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultButtonComponentView': defaultButtonComponentView,
        });
    });
    describe('Drawer Close Button', () => {
        const TRIGGER_TEXT = 'Open drawer';
        const MAIN_CONTEXT_TEXT = 'Main content goes here';
        const Component = ({ onCloseClick }) => (_jsx(DesignSystemProvider, { children: _jsxs(Drawer.Root, { open: true, children: [_jsx(Drawer.Trigger, { children: TRIGGER_TEXT }), _jsx(Drawer.Content, { title: "drawer_title", componentId: "drawer_test", onCloseClick: onCloseClick, children: _jsx("div", { children: MAIN_CONTEXT_TEXT }) })] }) }));
        it('calls the onClick callback if provided', async () => {
            const onCloseClick = jest.fn();
            render(_jsx(Component, { onCloseClick: onCloseClick }));
            const closeButton = screen.getByRole('button', { name: 'Close' });
            await userEvent.click(closeButton);
            expect(onCloseClick).toHaveBeenCalled();
        });
    });
    describe('Drawer Overlay', () => {
        it('Displays overlay only when the drawer is open', async () => {
            const OPEN_TEXT = 'Open button';
            function TestComponent() {
                const [open, setOpen] = useState(false);
                return (_jsxs(DesignSystemProvider, { children: [_jsxs("button", { onClick: () => setOpen(true), children: [" ", OPEN_TEXT, " "] }), _jsxs(Drawer.Root, { open: open, children: [_jsx(Drawer.Trigger, { children: "Open drawer" }), _jsx(Drawer.Content, { title: "drawer_title", componentId: "drawer_test", children: _jsx("div", { children: "Main content goes here" }) })] })] }));
            }
            render(_jsx(TestComponent, {}));
            expect(screen.queryByTestId('drawer-overlay')).not.toBeInTheDocument();
            // Open the drawer
            await userEvent.click(screen.getByRole('button', { name: OPEN_TEXT }));
            // Now the overlay should be visible
            expect(screen.getByTestId('drawer-overlay')).toBeVisible();
        });
        it('Hides overlay when the drawer is nested', () => {
            render(_jsx(DesignSystemProvider, { children: _jsx(Drawer.Root, { open: true, children: _jsx(Drawer.Content, { title: "drawer_title", componentId: "drawer_test", children: _jsx(Drawer.Root, { open: true, children: _jsx(Drawer.Content, { title: "nested_drawer_title", componentId: "nested_drawer_test", children: _jsx(Drawer.Root, { open: true, children: _jsx(Drawer.Content, { title: "deep_nested_drawer_title", componentId: "deep_nested_drawer_test", children: _jsx("div", { children: "Nested content" }) }) }) }) }) }) }) }));
            const drawerOverlays = screen.getAllByTestId('drawer-overlay');
            expect(drawerOverlays).toHaveLength(3);
            expect(drawerOverlays[0]).toBeVisible();
            expect(drawerOverlays[1]).not.toBeVisible();
            expect(drawerOverlays[2]).not.toBeVisible();
        });
        it('calls onCloseClick() when closeOnClickOutside is set and clicking on overlay', async () => {
            const onCloseClick = jest.fn();
            render(_jsx(DesignSystemProvider, { children: _jsx(Drawer.Root, { open: true, children: _jsx(Drawer.Content, { title: "drawer_title", componentId: "drawer_test", onCloseClick: onCloseClick, closeOnClickOutside: true, children: _jsx("div", { children: "Main content goes here" }) }) }) }));
            expect(onCloseClick).not.toHaveBeenCalled();
            const overlay = screen.getByTestId('drawer-overlay');
            await userEvent.click(overlay);
            expect(onCloseClick).toHaveBeenCalled();
        });
    });
});
//# sourceMappingURL=Drawer.test.js.map