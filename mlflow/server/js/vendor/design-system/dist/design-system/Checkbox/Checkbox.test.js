import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Checkbox } from './Checkbox';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Checkbox', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.checkbox': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('isChecked updates correctly', async () => {
        let isChecked = false;
        const changeHandlerFn = jest.fn();
        const changeHandler = (checked) => {
            isChecked = checked;
            changeHandlerFn();
        };
        render(_jsx(Checkbox, { componentId: "codegen_design-system_src_design-system_checkbox_checkbox.test.tsx_16", isChecked: isChecked, onChange: changeHandler, children: "Basic checkbox" }));
        await userEvent.click(screen.getByRole('checkbox'));
        expect(changeHandlerFn).toHaveBeenCalledTimes(1);
        expect(isChecked).toBe(true);
    });
    it("isChecked doesn't update without onChange", async () => {
        // eslint-disable-next-line prefer-const
        let isChecked = false;
        render(_jsx(Checkbox, { componentId: "codegen_design-system_src_design-system_checkbox_checkbox.test.tsx_30", isChecked: isChecked, children: "Basic checkbox" }));
        await userEvent.click(screen.getByRole('checkbox'));
        expect(isChecked).toBe(false);
    });
    it('handles changes with DesignSystemEventProvider', async () => {
        // Arrange
        const handleOnChange = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Checkbox, { componentId: "bestCheckboxEver", onChange: handleOnChange }) }));
        await waitFor(() => {
            expect(screen.getByRole('checkbox')).toBeVisible();
        });
        expect(handleOnChange).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'bestCheckboxEver',
            componentType: 'checkbox',
            shouldStartInteraction: false,
        });
        // Act
        await userEvent.click(screen.getByRole('checkbox'));
        // Assert
        expect(handleOnChange).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'bestCheckboxEver',
            componentType: 'checkbox',
            shouldStartInteraction: false,
            value: true,
        });
    });
    it('handles views with default values with DesignSystemEventProvider', async () => {
        // Arrange
        const handleOnChange = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Checkbox, { componentId: "bestCheckboxEver", onChange: handleOnChange, isChecked: false, defaultChecked: true }) }));
        await waitFor(() => {
            expect(screen.getByRole('checkbox')).toBeVisible();
        });
        expect(handleOnChange).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'bestCheckboxEver',
            componentType: 'checkbox',
            shouldStartInteraction: false,
            value: false,
        });
        // Act
        await userEvent.click(screen.getByRole('checkbox'));
        // Assert
        expect(handleOnChange).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'bestCheckboxEver',
            componentType: 'checkbox',
            shouldStartInteraction: false,
            value: true,
        });
    });
});
//# sourceMappingURL=Checkbox.test.js.map