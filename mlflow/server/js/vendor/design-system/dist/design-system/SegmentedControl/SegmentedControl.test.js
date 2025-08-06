import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SegmentedControlButton, SegmentedControlGroup } from './SegmentedControl';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('SegmentedControl', () => {
    const { setSafex } = setupSafexTesting();
    const onChangeSpy = jest.fn();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const Component = ({ valueHasNoPii, value, defaultValue, }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(SegmentedControlGroup, { name: "test", componentId: "segmented_control_group_test", onChange: onChangeSpy, valueHasNoPii: valueHasNoPii, value: value, defaultValue: defaultValue, children: [_jsx(SegmentedControlButton, { value: "a", children: "A" }), _jsx(SegmentedControlButton, { value: "b", children: "B" })] }) }));
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.segmentedControlGroup': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('emits value change events without value', async () => {
        render(_jsx(Component, {}));
        await waitFor(() => {
            expect(screen.getByText('B')).toBeVisible();
        });
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
        });
        const button = screen.getByText('B');
        await userEvent.click(button);
        expect(onChangeSpy).toHaveBeenCalledWith(expect.objectContaining({ target: expect.objectContaining({ value: 'b' }) }));
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('emits value change events with value', async () => {
        render(_jsx(Component, { valueHasNoPii: true, value: "a", defaultValue: "b" }));
        await waitFor(() => {
            expect(screen.getByText('B')).toBeVisible();
        });
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
            value: 'a',
        });
        const button = screen.getByText('B');
        await userEvent.click(button);
        expect(onChangeSpy).toHaveBeenCalledWith(expect.objectContaining({ target: expect.objectContaining({ value: 'b' }) }));
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
            value: 'b',
        });
    });
    it('emits view events with default value', async () => {
        render(_jsx(Component, { valueHasNoPii: true, defaultValue: "a" }));
        await waitFor(() => {
            expect(screen.getByText('B')).toBeVisible();
        });
        expect(onChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
            value: 'a',
        });
        const button = screen.getByText('B');
        await userEvent.click(button);
        expect(onChangeSpy).toHaveBeenCalledWith(expect.objectContaining({ target: expect.objectContaining({ value: 'b' }) }));
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'segmented_control_group_test',
            componentType: 'segmented_control_group',
            shouldStartInteraction: false,
            value: 'b',
        });
    });
});
//# sourceMappingURL=SegmentedControl.test.js.map