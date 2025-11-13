import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Banner } from './Banner';
import { DesignSystemEventProviderComponentSubTypes, setupDesignSystemEventProviderForTesting, } from '../../design-system';
import { setupSafexTesting } from '../../design-system/utils/safex';
describe('<Banner/>', () => {
    const { setSafex } = setupSafexTesting();
    const renderComponent = (props) => {
        render(_jsx(Banner, { ...props }));
    };
    describe.each([false, true])('Shared banner tests: databricks.fe.observability.defaultButtonComponentView set to %s', (defaultButtonComponentView) => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultButtonComponentView': defaultButtonComponentView,
            });
        });
        it('All props loaded correctly and banner closes', async () => {
            const onAcceptMock = jest.fn();
            const onCloseMock = jest.fn();
            renderComponent({
                componentId: 'noop_testing_tracking_id',
                level: 'info',
                message: 'Test message',
                description: 'Test description',
                onAccept: onAcceptMock,
                ctaText: 'Go',
                onClose: onCloseMock,
            });
            expect(screen.getByText('Test message')).toBeVisible();
            expect(screen.getByText('Test description')).toBeVisible();
            expect(screen.getByTestId('level-info-icon')).toBeVisible();
            const closeButton = screen.getByTestId('banner-dismiss');
            const actionButton = screen.getByText('Go');
            expect(actionButton).toBeVisible();
            expect(closeButton).toBeVisible();
            await userEvent.click(actionButton);
            expect(onAcceptMock).toHaveBeenCalled();
            await userEvent.click(closeButton);
            expect(screen.queryByText('Test message')).not.toBeInTheDocument();
            expect(onCloseMock).toHaveBeenCalled();
        });
        it('Does not show accept button when onAccept is not provided', () => {
            renderComponent({
                componentId: 'noop_testing_tracking_id',
                level: 'info',
                message: 'Test message',
                description: 'Test description',
                ctaText: 'Go',
            });
            expect(screen.queryByText('Go')).not.toBeInTheDocument();
        });
        // The case where closable is undefined is already tested above
        it('Shows close button when closable is true', () => {
            renderComponent({
                componentId: 'noop_testing_tracking_id',
                level: 'info',
                message: 'Test message',
                closable: true,
            });
            expect(screen.getByTestId('banner-dismiss')).toBeInTheDocument();
        });
        it('Does not show close button when closable is false', () => {
            renderComponent({
                componentId: 'noop_testing_tracking_id',
                level: 'info',
                message: 'Test message',
                closable: false,
            });
            expect(screen.queryByTestId('banner-dismiss')).not.toBeInTheDocument();
        });
        it.each([
            { type: 'error', componentSubType: DesignSystemEventProviderComponentSubTypes.Error },
            { type: 'info', componentSubType: DesignSystemEventProviderComponentSubTypes.Info },
            { type: 'warning', componentSubType: DesignSystemEventProviderComponentSubTypes.Warning },
            { type: 'info_light_purple', componentSubType: DesignSystemEventProviderComponentSubTypes.InfoLightPurple },
            { type: 'info_dark_purple', componentSubType: DesignSystemEventProviderComponentSubTypes.InfoDarkPurple },
        ])('callback for %s Banner is being recorded and sub type is being passed through', async ({ type, componentSubType }) => {
            window.IntersectionObserver = undefined;
            const mockUseOnEventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(mockUseOnEventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Banner, { componentId: `test.internal-design-system-event-provider.${type}`, level: type, message: "text" }) }));
            expect(mockUseOnEventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: `test.internal-design-system-event-provider.${type}`,
                componentType: 'banner',
                componentSubType,
                shouldStartInteraction: false,
                value: undefined,
            });
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
            const handleAccept = jest.fn();
            const handleClose = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Banner, { componentId: "bestBannerEver", level: "info", message: "Test message", onAccept: handleAccept, ctaText: "Go", onClose: handleClose, "data-testid": "banner-container" }) }));
            expect(handleAccept).not.toHaveBeenCalled();
            expect(handleClose).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test message')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'bestBannerEver',
                componentSubType: 'info',
                componentType: 'banner',
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for accept
            await userEvent.click(screen.getByText('Go'));
            expect(handleAccept).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onClick',
                componentId: 'bestBannerEver.accept',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByTestId('banner-dismiss'));
            expect(handleClose).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onClick',
                componentId: 'bestBannerEver.close',
                componentType: 'button',
                shouldStartInteraction: true,
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
            const handleAccept = jest.fn();
            const handleClose = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Banner, { componentId: "bestBannerEver", level: "info", message: "Test message", onAccept: handleAccept, ctaText: "Go", onClose: handleClose, "data-testid": "banner-container" }) }));
            expect(handleAccept).not.toHaveBeenCalled();
            expect(handleClose).not.toHaveBeenCalled();
            // Assert for view
            expect(screen.getByText('Test message')).toBeVisible();
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestBannerEver.accept',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestBannerEver.close',
                componentType: 'button',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'bestBannerEver',
                componentSubType: 'info',
                componentType: 'banner',
                shouldStartInteraction: false,
                value: undefined,
            });
            // Assert for accept
            await userEvent.click(screen.getByText('Go'));
            expect(handleAccept).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(4);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onClick',
                componentId: 'bestBannerEver.accept',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
            await userEvent.click(screen.getByTestId('banner-dismiss'));
            expect(handleClose).toHaveBeenCalledTimes(1);
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onClick',
                componentId: 'bestBannerEver.close',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
        });
    });
});
//# sourceMappingURL=Banner.test.js.map