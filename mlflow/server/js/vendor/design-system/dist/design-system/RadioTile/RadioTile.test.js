import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RadioTile } from '.';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider/DesignSystemEventProviderTestUtils';
import { Radio } from '../Radio';
import { setupSafexTesting } from '../utils/safex';
describe('RadioTile', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.radio': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('emits value change events without value', async () => {
        const onValueChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(Radio.Group, { name: "test", componentId: "test", onChange: onValueChangeSpy, value: "a", defaultValue: "b", children: [_jsx(RadioTile, { value: "a", children: "A" }), _jsx(RadioTile, { value: "b", children: "B" }), _jsx(RadioTile, { value: "c", children: "C" }), _jsx(RadioTile, { value: "d", disabled: true, children: "D" })] }) }));
        await waitFor(() => {
            expect(screen.getByText('B')).toBeVisible();
        });
        expect(onValueChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'test',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: undefined,
        });
        const radioTile = screen.getByRole('radio', { name: 'B' });
        await userEvent.click(radioTile);
        expect(onValueChangeSpy).toHaveBeenCalledWith(expect.objectContaining({
            target: expect.objectContaining({
                value: 'b',
            }),
        }));
        expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'test',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('emits value change events with value', async () => {
        const onValueChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(Radio.Group, { name: "test", componentId: "test", onChange: onValueChangeSpy, valueHasNoPii: true, value: "a", defaultValue: "b", children: [_jsx(RadioTile, { value: "a", children: "A" }), _jsx(RadioTile, { value: "b", children: "B" }), _jsx(RadioTile, { value: "c", children: "C" }), _jsx(RadioTile, { value: "d", disabled: true, children: "D" })] }) }));
        await waitFor(() => {
            expect(screen.getByText('B')).toBeVisible();
        });
        expect(onValueChangeSpy).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'test',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: 'a',
        });
        const radioTile = screen.getByRole('radio', { name: 'B' });
        await userEvent.click(radioTile);
        expect(onValueChangeSpy).toHaveBeenCalledWith(expect.objectContaining({
            target: expect.objectContaining({
                value: 'b',
            }),
        }));
        expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'test',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: 'b',
        });
    });
});
//# sourceMappingURL=RadioTile.test.js.map