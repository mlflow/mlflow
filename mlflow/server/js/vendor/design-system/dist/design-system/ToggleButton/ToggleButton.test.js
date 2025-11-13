import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { beforeEach, describe, it, jest, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ToggleButton } from './ToggleButton';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('ToggleButton', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.toggleButton': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('handles changes with DesignSystemEventProvider', async () => {
        // Arrange
        const handleOnPressedChange = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(ToggleButton, { pressed: false, onPressedChange: handleOnPressedChange, componentId: "bestToggleButtonEver" }) }));
        await waitFor(() => {
            expect(screen.getByRole('button')).toBeVisible();
        });
        expect(handleOnPressedChange).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentType: 'toggle_button',
            componentId: 'bestToggleButtonEver',
            shouldStartInteraction: false,
            value: false,
        });
        // Act
        await userEvent.click(screen.getByRole('button'));
        // Assert
        expect(handleOnPressedChange).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentType: 'toggle_button',
            componentId: 'bestToggleButtonEver',
            shouldStartInteraction: false,
            value: true,
        });
    });
});
//# sourceMappingURL=ToggleButton.test.js.map