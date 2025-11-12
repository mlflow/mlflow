import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Form, RhfForm } from './Form';
import { Button, setupDesignSystemEventProviderForTesting } from '../../design-system';
describe('Form', () => {
    it('renders the form and handles submission', async () => {
        // Arrange
        const handleSubmit = jest.fn().mockResolvedValue(undefined);
        render(_jsx(Form, { componentId: "testForm", onSubmit: handleSubmit, children: _jsx("button", { type: "submit", children: "Submit" }) }));
        // Act
        await userEvent.click(screen.getByRole('button', { name: /submit/i }));
        // Assert
        expect(handleSubmit).toHaveBeenCalledTimes(1);
    });
    it('calls the DesignSystemEventProvider onSubmit callback', async () => {
        // Arrange
        const handleSubmit = jest.fn().mockResolvedValue(undefined);
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Form, { componentId: "eventForm", onSubmit: handleSubmit, children: _jsx(Button, { componentId: "submit-button-test", htmlType: "submit", children: "Submit" }) }) }));
        // Act
        await userEvent.click(screen.getByRole('button', { name: /submit/i }));
        // Assert
        expect(handleSubmit).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onSubmit',
            componentId: 'eventForm',
            componentType: 'form',
            componentSubType: undefined,
            shouldStartInteraction: true,
            value: undefined,
            mode: 'default',
            formPropertyValues: {
                initial: undefined,
                final: undefined,
            },
            referrerComponent: {
                id: 'submit-button-test',
                type: 'button',
            },
            event: expect.anything(),
        });
    });
    it('throws error if nested form components are rendered', () => {
        // Arrange
        const handleSubmit = jest.fn().mockResolvedValue(undefined);
        // Assert
        expect(() => {
            render(_jsx(Form, { componentId: "outerForm", onSubmit: handleSubmit, children: _jsx(Form, { componentId: "innerForm", onSubmit: handleSubmit, children: _jsx("button", { type: "submit", children: "Submit" }) }) }));
        }).toThrowError('DuBois Form component cannot be nested');
    });
    it('calls the DesignSystemEventProvider onSubmit callback with initial & final values', async () => {
        // Arrange
        const handleSubmit = (onValid, _) => async (e) => {
            await onValid({ testProperty: 123 }, e);
        };
        const eventCallback = jest.fn();
        const onValidCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(RhfForm, { componentId: "eventForm", handleSubmit: handleSubmit, handleValid: onValidCallback, initialState: { testProperty: 321, testFooProperty: false, testBarProperty: 'test' }, children: _jsx(Button, { componentId: "submit-button-test", htmlType: "submit", children: "Submit" }) }) }));
        // Act
        await userEvent.click(screen.getByRole('button', { name: /submit/i }));
        // Assert
        expect(onValidCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onClick',
            componentId: 'submit-button-test',
            componentType: 'button',
            componentSubType: undefined,
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
            isInteractionSubject: false,
        });
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onSubmit',
            componentId: 'eventForm',
            componentType: 'form',
            componentSubType: undefined,
            shouldStartInteraction: true,
            value: undefined,
            mode: 'associate_event_only',
            formPropertyValues: {
                initial: undefined,
                final: undefined,
            },
            referrerComponent: {
                id: 'submit-button-test',
                type: 'button',
            },
            event: expect.anything(),
        });
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onSubmit',
            componentId: 'eventForm',
            componentType: 'form',
            componentSubType: undefined,
            shouldStartInteraction: true,
            value: undefined,
            mode: 'default',
            formPropertyValues: {
                initial: {
                    testProperty: 321,
                    testFooProperty: false,
                    testBarProperty: 'test',
                },
                final: {
                    testProperty: 123,
                },
            },
            referrerComponent: {
                id: 'submit-button-test',
                type: 'button',
            },
            event: expect.anything(),
        });
    });
});
//# sourceMappingURL=Form.test.js.map