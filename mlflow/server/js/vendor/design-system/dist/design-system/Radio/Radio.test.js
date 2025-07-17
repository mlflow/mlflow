import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Radio } from '.';
import { Form, useFormContext } from '../../development/Form/Form';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Radio', () => {
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
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(Radio.Group, { name: "test", componentId: "test", onChange: onValueChangeSpy, children: [_jsx(Radio, { value: "a", children: "A" }), _jsx(Radio, { value: "b", children: "B" }), _jsx(Radio, { value: "c", children: "C" }), _jsx(Radio, { value: "d", disabled: true, children: "D" })] }) }));
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
        const radio = screen.getByText('B');
        // Ensure pills are interactive
        await userEvent.click(radio);
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
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsxs(Radio.Group, { name: "test", componentId: "test", onChange: onValueChangeSpy, valueHasNoPii: true, children: [_jsx(Radio, { value: "a", children: "A" }), _jsx(Radio, { value: "b", children: "B" }), _jsx(Radio, { value: "c", children: "C" }), _jsx(Radio, { value: "d", disabled: true, children: "D" })] }) }));
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
        const radio = screen.getByText('B');
        // Ensure pills are interactive
        await userEvent.click(radio);
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
    it('works with form submission', async () => {
        // Arrange
        const handleSubmit = jest.fn().mockResolvedValue(undefined);
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        const TestComponent = () => {
            const formContext = useFormContext();
            return (_jsxs(Radio.Group, { componentId: "test-radio", onChange: (e) => {
                    formContext.formRef?.current?.requestSubmit();
                }, name: "test-radio", children: [_jsx(Radio, { value: 1, children: "1" }), _jsx(Radio, { value: 2, children: "2" })] }));
        };
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Form, { componentId: "eventForm", onSubmit: handleSubmit, children: _jsx(TestComponent, {}) }) }));
        await waitFor(() => {
            expect(screen.getByText('1')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'test-radio',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: undefined,
        });
        // Act
        const radio = screen.getByText('2');
        await userEvent.click(radio);
        // Assert
        expect(handleSubmit).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'test-radio',
            componentType: 'radio_group',
            shouldStartInteraction: false,
        });
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onSubmit',
            componentId: 'eventForm',
            componentType: 'form',
            componentSubType: undefined,
            shouldStartInteraction: true,
            event: expect.anything(),
            referrerComponent: {
                id: 'test-radio',
                type: 'radio_group',
            },
            formPropertyValues: {
                final: undefined,
                initial: undefined,
            },
            mode: 'default',
            value: undefined,
        });
    });
    it('emits value change events for standalone radio without RadioGroup', async () => {
        const onChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Radio, { componentId: "standalone-radio", value: "test", onChange: onChangeSpy, children: "Test Radio" }) }));
        await waitFor(() => {
            expect(screen.getByText('Test Radio')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'standalone-radio',
            componentType: 'radio',
            shouldStartInteraction: false,
            value: undefined,
        });
        const radio = screen.getByText('Test Radio');
        await userEvent.click(radio);
        expect(onChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'standalone-radio',
            componentType: 'radio',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('does not emit value change events for radio within RadioGroup', async () => {
        const onChangeSpy = jest.fn();
        const groupOnChangeSpy = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Radio.Group, { name: "test-group", componentId: "test-group", onChange: groupOnChangeSpy, children: _jsx(Radio, { componentId: "standalone-radio", value: "test", onChange: onChangeSpy, children: "Test Radio" }) }) }));
        await waitFor(() => {
            expect(screen.getByText('Test Radio')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'test-group',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: undefined,
        });
        const radio = screen.getByText('Test Radio');
        await userEvent.click(radio);
        expect(onChangeSpy).toHaveBeenCalledTimes(1);
        expect(groupOnChangeSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'test-group',
            componentType: 'radio_group',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=Radio.test.js.map