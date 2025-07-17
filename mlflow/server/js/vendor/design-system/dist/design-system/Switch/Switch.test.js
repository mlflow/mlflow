import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Switch } from './Switch';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
describe('Switch', () => {
    it('handles changes with DesignSystemEventProvider', async () => {
        // Arrange
        const handleOnChange = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Switch, { componentId: "bestSwitchEver", onChange: handleOnChange }) }));
        expect(handleOnChange).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        // Act
        await userEvent.click(screen.getByRole('switch'));
        // Assert
        expect(handleOnChange).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onValueChange',
            componentId: 'bestSwitchEver',
            componentType: 'switch',
            shouldStartInteraction: false,
            value: true,
        });
    });
    it('handles view event with DesignSystemEventProvider', async () => {
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
        const handleOnChange = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Switch, { componentId: "bestSwitchEver", onChange: handleOnChange, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView], checked: true }) }));
        expect(handleOnChange).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalled();
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'bestSwitchEver',
            componentType: 'switch',
            shouldStartInteraction: false,
            value: true,
        });
    });
});
//# sourceMappingURL=Switch.test.js.map