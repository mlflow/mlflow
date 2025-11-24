import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { expect, describe, it, jest, beforeEach } from '@jest/globals';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TextArea } from './TextArea';
import { Form } from '../../development';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('TextArea', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.textArea': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('calls onChange when input changes', async () => {
        const onChange = jest.fn();
        render(_jsx(TextArea, { componentId: "MY_TRACKING_ID", onChange: onChange }));
        const input = 'abc';
        await userEvent.type(screen.getByRole('textbox'), input);
        expect(onChange).toHaveBeenCalledTimes(input.length);
    });
    it('calls onChange with DesignSystemEventProvider', async () => {
        // Arrange
        const eventCallback = jest.fn();
        const onChange = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TextArea, { onChange: onChange, componentId: "MY_TRACKING_ID" }) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
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
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
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
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
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
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TextArea, { allowClear: true, onChange: onChange, onFocus: onFocus, componentId: "MY_TRACKING_ID" }) }));
        await waitFor(() => {
            expect(screen.getByRole('textbox')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
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
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
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
            componentId: 'MY_TRACKING_ID',
            componentType: 'text_area',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onFocus).toHaveBeenCalledTimes(2);
    });
    describe('Form submission', () => {
        it('does not submit form when allowFormSubmitOnEnter is false', async () => {
            // Arrange
            const onSubmit = jest.fn();
            render(_jsx(Form, { onSubmit: onSubmit, componentId: "test-form", children: _jsx(TextArea, { allowClear: true, componentId: "MY_TRACKING_ID" }) }));
            // Act
            await userEvent.type(screen.getByRole('textbox'), '{Enter}');
            // Assert
            expect(onSubmit).not.toHaveBeenCalled();
        });
        it('trigger submit with enter when allowFormSubmitOnEnter is true', async () => {
            // Arrange
            const onSubmit = jest.fn();
            render(_jsx(Form, { onSubmit: onSubmit, componentId: "test-form", children: _jsx(TextArea, { allowClear: true, allowFormSubmitOnEnter: true, componentId: "MY_TRACKING_ID" }) }));
            // Act
            await userEvent.type(screen.getByRole('textbox'), '{Enter}');
            // Assert
            expect(onSubmit).toBeCalledTimes(1);
        });
        it('triggers submit with platform enter', async () => {
            // Arrange
            const eventCallback = jest.fn();
            const onSubmit = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Form, { onSubmit: onSubmit, componentId: "test-form", children: _jsx(TextArea, { allowClear: true, componentId: "MY_TRACKING_ID" }) }) }));
            await waitFor(() => {
                expect(screen.getByRole('textbox')).toBeVisible();
            });
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onView',
                componentId: 'MY_TRACKING_ID',
                componentType: 'text_area',
                shouldStartInteraction: false,
                value: undefined,
            });
            expect(onSubmit).not.toHaveBeenCalled();
            // input three letters and check onValueChange called
            const textbox = screen.getByRole('textbox');
            await userEvent.type(textbox, 'abc');
            await waitFor(() => expect(eventCallback).toBeCalledTimes(2), { timeout: 5000 });
            // Control by itself is not enough
            await userEvent.type(textbox, '{Control}');
            expect(onSubmit).not.toHaveBeenCalled();
            // But Control+Enter submits
            await userEvent.type(textbox, '{Control>}{Enter}');
            expect(onSubmit).toBeCalledTimes(1);
        });
    });
});
//# sourceMappingURL=TextArea.test.js.map