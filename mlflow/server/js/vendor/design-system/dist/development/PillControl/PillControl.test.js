import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PillControl } from '.';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system';
import { setupDesignSystemEventProviderForTesting } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProviderTestUtils';
describe('PillControl', () => {
    beforeEach(() => {
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('renders pills as a radio-group', async () => {
        const onValueChangeSpy = jest.fn();
        render(_jsxs(PillControl.Root, { componentId: "test", onValueChange: onValueChangeSpy, analyticsEvents: [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ], children: [_jsx(PillControl.Item, { value: "a", children: "A" }), _jsx(PillControl.Item, { value: "b", children: "B" }), _jsx(PillControl.Item, { value: "c", children: "C" }), _jsx(PillControl.Item, { value: "d", disabled: true, children: "D" })] }));
        const radioGroup = screen.getByRole('radiogroup');
        const pills = within(radioGroup).getAllByRole('radio');
        // Ensure pills are rendered
        expect(pills).toHaveLength(4);
        // Ensure pills are interactive
        await userEvent.click(pills[1]);
        expect(onValueChangeSpy).toHaveBeenCalledWith('b');
        // Ensure disabled items can not have interaction
        expect(pills[3]).toBeDisabled();
    });
    it('emits value change events without value', async () => {
        const onValueChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(PillControl.Root, { componentId: "test", onValueChange: onValueChangeSpy, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], children: [_jsx(PillControl.Item, { value: "a", children: "A" }), _jsx(PillControl.Item, { value: "b", children: "B" }), _jsx(PillControl.Item, { value: "c", children: "C" }), _jsx(PillControl.Item, { value: "d", disabled: true, children: "D" })] }) }));
        expect(onValueChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        const radioGroup = screen.getByRole('radiogroup');
        const pills = within(radioGroup).getAllByRole('radio');
        // Ensure pills are rendered
        expect(pills).toHaveLength(4);
        // Ensure pills are interactive
        await userEvent.click(pills[1]);
        expect(onValueChangeSpy).toHaveBeenCalledWith('b');
        expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onValueChange',
            componentId: 'test',
            componentType: 'pill_control',
            shouldStartInteraction: false,
            value: undefined,
        });
        // Ensure disabled items can not have interaction
        expect(pills[3]).toBeDisabled();
    });
    it('emits value change events with value', async () => {
        const onValueChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(PillControl.Root, { componentId: "test", valueHasNoPii: true, onValueChange: onValueChangeSpy, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], children: [_jsx(PillControl.Item, { value: "a", children: "A" }), _jsx(PillControl.Item, { value: "b", children: "B" }), _jsx(PillControl.Item, { value: "c", children: "C" }), _jsx(PillControl.Item, { value: "d", disabled: true, children: "D" })] }) }));
        expect(onValueChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        const radioGroup = screen.getByRole('radiogroup');
        const pills = within(radioGroup).getAllByRole('radio');
        // Ensure pills are rendered
        expect(pills).toHaveLength(4);
        // Ensure pills are interactive
        await userEvent.click(pills[1]);
        expect(onValueChangeSpy).toHaveBeenCalledWith('b');
        expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onValueChange',
            componentId: 'test',
            componentType: 'pill_control',
            shouldStartInteraction: false,
            value: 'b',
        });
        // Ensure disabled items can not have interaction
        expect(pills[3]).toBeDisabled();
    });
    it('emits view events on mount with value if valueHasNoPii', async () => {
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(PillControl.Root, { componentId: "test", valueHasNoPii: true, defaultValue: "b", analyticsEvents: [
                    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                    DesignSystemEventProviderAnalyticsEventTypes.OnView,
                ], children: [_jsx(PillControl.Item, { value: "a", children: "A" }), _jsx(PillControl.Item, { value: "b", children: "B" }), _jsx(PillControl.Item, { value: "c", children: "C" })] }) }));
        await waitFor(() => {
            expect(screen.getByRole('radiogroup')).toBeVisible();
        });
        // Verify onView was called
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'test',
            componentType: 'pill_control',
            shouldStartInteraction: false,
            value: 'b',
        });
    });
});
it('emits view events on mount without value if not valueHasNoPii', async () => {
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(PillControl.Root, { componentId: "test", defaultValue: "b", analyticsEvents: [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ], children: [_jsx(PillControl.Item, { value: "a", children: "A" }), _jsx(PillControl.Item, { value: "b", children: "B" }), _jsx(PillControl.Item, { value: "c", children: "C" })] }) }));
    await waitFor(() => {
        expect(screen.getByRole('radiogroup')).toBeVisible();
    });
    // Verify onView was called without value
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
        eventType: 'onView',
        componentId: 'test',
        componentType: 'pill_control',
        shouldStartInteraction: false,
    });
});
//# sourceMappingURL=PillControl.test.js.map