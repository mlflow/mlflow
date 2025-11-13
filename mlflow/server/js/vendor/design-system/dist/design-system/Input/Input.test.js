import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { expect, describe, it, jest, beforeEach } from '@jest/globals';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Input } from './Input';
import { Form } from '../../development/Form';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Input', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.input': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('calls onChange when input changes', async () => {
        const onChange = jest.fn();
        render(_jsx(Input, { componentId: "MY_TRACKING_ID", onChange: onChange }));
        const input = 'abc';
        await userEvent.type(screen.getByRole('textbox'), input);
        expect(onChange).toHaveBeenCalledTimes(input.length);
    });
    it('calls onChange when input is cleared even if onClear is defined', async () => {
        const onChange = jest.fn();
        const onClear = jest.fn();
        render(_jsx(Input, { componentId: "MY_TRACKING_ID", allowClear: true, onChange: onChange, onClear: onClear }));
        const input = 'abc';
        await userEvent.type(screen.getByRole('textbox'), input);
        await userEvent.clear(screen.getByRole('textbox'));
        expect(onChange).toHaveBeenCalledTimes(input.length + 1);
        expect(onClear).not.toHaveBeenCalled();
    });
    it('calls onClear when clear button is clicked', async () => {
        const onClear = jest.fn();
        const onChange = jest.fn();
        render(_jsx(Input, { componentId: "MY_TRACKING_ID", allowClear: true, onChange: onChange, onClear: onClear }));
        await userEvent.click(screen.getByLabelText('close-circle'));
        expect(onClear).toHaveBeenCalled();
        expect(onChange).not.toHaveBeenCalled();
    });
    it('calls onChange when clear button is clicked if onClear is not defined', async () => {
        const onChange = jest.fn();
        render(_jsx(Input, { componentId: "MY_TRACKING_ID", allowClear: true, onChange: onChange }));
        await userEvent.click(screen.getByLabelText('close-circle'));
        expect(onChange).toHaveBeenCalled();
    });
    it('calls onChange with DesignSystemEventProvider', async () => {
        // Arrange
        const eventCallback = jest.fn();
        const onChange = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Input, { allowClear: true, onChange: onChange, componentId: "bestInputEver" }) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onChange).not.toHaveBeenCalled();
        // input three letters and check onValueChange called
        await userEvent.type(screen.getByRole('textbox'), 'abc');
        await waitFor(() => expect(eventCallback).toBeCalledTimes(2), { timeout: 5000 });
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        // input three more letters immediately.
        // only one call should be made until focus event fired
        await userEvent.type(screen.getByRole('textbox'), 'def');
        expect(eventCallback).toHaveBeenCalledTimes(2);
        // focusout and focus again to allow onValueChange to be called again
        fireEvent.focusOut(screen.getByRole('textbox'));
        fireEvent.focus(screen.getByRole('textbox'));
        // called onValueChange for inputing 'hij
        await userEvent.type(screen.getByRole('textbox'), 'hij');
        await waitFor(() => expect(eventCallback).toBeCalledTimes(3), { timeout: 5000 });
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        // focusout and focus again to allow onValueChange to be called again for clear button
        fireEvent.focusOut(screen.getByRole('textbox'));
        fireEvent.focus(screen.getByRole('textbox'));
        // click clear button
        await userEvent.click(screen.getByLabelText('close-circle'));
        await waitFor(() => expect(eventCallback).toBeCalledTimes(4), { timeout: 5000 });
        // called onValueChange for clicking clear button
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('calls onChange and onFocus with DesignSystemEventProvider', async () => {
        // Arrange
        const eventCallback = jest.fn();
        const onChange = jest.fn();
        const onFocus = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Input, { allowClear: true, onChange: onChange, onFocus: onFocus, componentId: "bestInputEver" }) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onChange).not.toHaveBeenCalled();
        expect(onFocus).not.toHaveBeenCalled();
        // input three letters and check onValueChange called
        await userEvent.type(screen.getByRole('textbox'), 'abc');
        await waitFor(() => expect(eventCallback).toBeCalledTimes(2), { timeout: 5000 });
        expect(onFocus).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        // input three more letters immediately.
        // only one call should be made until focus event fired
        await userEvent.type(screen.getByRole('textbox'), 'def');
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(onFocus).toHaveBeenCalledTimes(1);
        // focusout and focus again to allow onValueChange to be called again
        fireEvent.focusOut(screen.getByRole('textbox'));
        fireEvent.focus(screen.getByRole('textbox'));
        // called onValueChange for inputing 'hij
        await userEvent.type(screen.getByRole('textbox'), 'hij');
        await waitFor(() => expect(eventCallback).toBeCalledTimes(3), { timeout: 5000 });
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onFocus).toHaveBeenCalledTimes(2);
        // focusout and focus again to allow onValueChange to be called again for clear button
        fireEvent.focusOut(screen.getByRole('textbox'));
        fireEvent.focus(screen.getByRole('textbox'));
        // click clear button
        await userEvent.click(screen.getByLabelText('close-circle'));
        await waitFor(() => expect(eventCallback).toBeCalledTimes(4), { timeout: 5000 });
        // called onValueChange for clicking clear button
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'bestInputEver',
            componentType: 'input',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onFocus).toHaveBeenCalledTimes(3);
    });
    describe('Form submission', () => {
        it('submits form on input enter', async () => {
            const eventCallback = jest.fn();
            const onFormSubmit = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Form, { onSubmit: onFormSubmit, componentId: "test-form", children: _jsx(Input, { componentId: "bestInputEver" }) }) }));
            await waitFor(() => {
                expect(screen.getByRole('textbox')).toBeVisible();
            });
            expect(onFormSubmit).not.toHaveBeenCalled();
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'bestInputEver',
                componentType: 'input',
                shouldStartInteraction: false,
                value: undefined,
            });
            await userEvent.type(screen.getByRole('textbox'), '{Enter}');
            expect(onFormSubmit).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onSubmit',
                componentId: 'test-form',
                componentType: 'form',
                referrerComponent: {
                    id: 'bestInputEver',
                    type: 'input',
                },
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                mode: 'default',
                formPropertyValues: { initial: undefined, final: undefined },
            });
        });
        it('submits form on input ctrl+enter', async () => {
            const eventCallback = jest.fn();
            const onFormSubmit = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Form, { onSubmit: onFormSubmit, componentId: "test-form", children: _jsx(Input, { componentId: "bestInputEver" }) }) }));
            await waitFor(() => {
                expect(screen.getByRole('textbox')).toBeVisible();
            });
            expect(onFormSubmit).not.toHaveBeenCalled();
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'bestInputEver',
                componentType: 'input',
                shouldStartInteraction: false,
                value: undefined,
            });
            const textbox = screen.getByRole('textbox');
            await userEvent.type(textbox, 'abc');
            await userEvent.type(textbox, '{Ctrl}');
            expect(onFormSubmit).not.toHaveBeenCalled();
            await userEvent.type(textbox, '{Ctrl>}{Enter}');
            expect(onFormSubmit).toBeCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onSubmit',
                componentId: 'test-form',
                componentType: 'form',
                referrerComponent: {
                    id: 'bestInputEver',
                    type: 'input',
                },
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                mode: 'default',
                formPropertyValues: { initial: undefined, final: undefined },
            });
        });
    });
});
//# sourceMappingURL=Input.test.js.map