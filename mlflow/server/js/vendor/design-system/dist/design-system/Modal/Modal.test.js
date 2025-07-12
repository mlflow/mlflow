import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { expect, describe, it, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Modal } from './Modal';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Modal', () => {
    const { setSafex } = setupSafexTesting();
    describe.each([false, true])('Shared Modal tests: databricks.fe.observability.defaultButtonComponentView set to %s', (defaultButtonComponentView) => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultButtonComponentView': defaultButtonComponentView,
            });
        });
        it("doesn't render with ant-scrolling-effect class on body", async () => {
            const { baseElement } = render(_jsx(Modal, { componentId: "codegen_design-system_src_design-system_modal_modal.test.tsx_11", visible: true, title: "Test modal" }));
            expect(baseElement).not.toHaveClass('ant-scrolling-effect');
            expect(baseElement).toHaveClass('scroll-lock-effect');
        });
    });
    describe('disabled defaultButtonComponentView', () => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultButtonComponentView': false,
            });
        });
        it('handles views & clicks with DesignSystemEventProvider', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            // Arrange
            const handleOk = jest.fn();
            const handleCancel = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Modal, { componentId: "bestModalEver", visible: true, title: "Test modal", cancelText: "Cancel", onCancel: handleCancel, okText: "Ok", onOk: handleOk, shouldStartInteraction: true }) }));
            expect(handleCancel).not.toHaveBeenCalled();
            expect(handleOk).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test modal')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'bestModalEver',
                componentType: 'modal',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for cancel
            await userEvent.click(screen.getByText('Cancel'));
            expect(handleCancel).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByText('Ok'));
            expect(handleOk).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
        });
        it('handles views & clicks with DesignSystemEventProvider but with shouldStartInteraction not set', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            // Arrange
            const handleOk = jest.fn();
            const handleCancel = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Modal, { componentId: "bestModalEver", visible: true, title: "Test modal", cancelText: "Cancel", onCancel: handleCancel, okText: "Ok", onOk: handleOk }) }));
            expect(handleCancel).not.toHaveBeenCalled();
            expect(handleOk).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test modal')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'bestModalEver',
                componentType: 'modal',
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for cancel
            await userEvent.click(screen.getByText('Cancel'));
            expect(handleCancel).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: false,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByText('Ok'));
            expect(handleOk).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: false,
                isInteractionSubject: true,
                value: undefined,
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
        it('handles views & clicks with DesignSystemEventProvider', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            // Arrange
            const handleOk = jest.fn();
            const handleCancel = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Modal, { componentId: "bestModalEver", visible: true, title: "Test modal", cancelText: "Cancel", onCancel: handleCancel, okText: "Ok", onOk: handleOk, shouldStartInteraction: true }) }));
            expect(handleCancel).not.toHaveBeenCalled();
            expect(handleOk).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test modal')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver',
                componentType: 'modal',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for cancel
            await userEvent.click(screen.getByText('Cancel'));
            expect(handleCancel).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(4);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByText('Ok'));
            expect(handleOk).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
        });
        it('handles views & clicks with DesignSystemEventProvider but with shouldStartInteraction not set', async () => {
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
            // Arrange
            const handleOk = jest.fn();
            const handleCancel = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Modal, { componentId: "bestModalEver", visible: true, title: "Test modal", cancelText: "Cancel", onCancel: handleCancel, okText: "Ok", onOk: handleOk }) }));
            expect(handleCancel).not.toHaveBeenCalled();
            expect(handleOk).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test modal')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestModalEver',
                componentType: 'modal',
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for cancel
            await userEvent.click(screen.getByText('Cancel'));
            expect(handleCancel).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(4);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.cancel',
                componentType: 'button',
                shouldStartInteraction: false,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByText('Ok'));
            expect(handleOk).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onClick',
                componentId: 'bestModalEver.footer.ok',
                componentType: 'button',
                shouldStartInteraction: false,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
        });
    });
});
//# sourceMappingURL=Modal.test.js.map