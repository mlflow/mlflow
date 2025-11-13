import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from './DesignSystemEventProvider';
const MockChildComponent = (props) => {
    const button1Context = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: 'button1',
        analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });
    const button2Context = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: 'button2',
        analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });
    const input1Context = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Input,
        componentId: 'input1',
        analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    });
    const formContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Form,
        componentId: 'form1',
        analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnSubmit],
    });
    button1Context.onView();
    button2Context.onView();
    input1Context.onView();
    return (_jsxs("div", { children: [_jsx("button", { onClick: button1Context.onClick, children: "Click me" }), _jsx("button", { onClick: button2Context.onClick, children: "View me" }), _jsx("input", { onChange: () => input1Context.onValueChange() }), _jsx("form", { onSubmit: (e) => formContext.onSubmit({
                    event: e,
                    initialState: { input1: 123 },
                    finalState: { input1: 321 },
                    referrerComponent: {
                        type: DesignSystemEventProviderComponentTypes.Button,
                        id: 'submit-button',
                    },
                }), children: _jsx("button", { type: "submit", children: "Submit" }) })] }));
};
describe('DesignSystemEventProvider', () => {
    let uuidIncrement = 0;
    beforeEach(async () => {
        uuidIncrement = 0;
        // mock uuid generation to increment on each call
        const mockGenerateUuidV4 = jest.fn();
        jest.spyOn(await import('../utils/useStableUuidV4'), 'useStableUuidV4').mockImplementation(mockGenerateUuidV4);
        mockGenerateUuidV4.mockImplementation(() => {
            return `testuuid-00v4-0000-0000-00000000000_${uuidIncrement++}`;
        });
    });
    it('provides onClick callback', async () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, {}) }));
        expect(screen.getByText('Click me')).toBeInTheDocument();
        expect(screen.getByText('View me')).toBeInTheDocument();
        await userEvent.click(screen.getByText('Click me'));
        await userEvent.click(screen.getByText('View me'));
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    it('provides onView callback', () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, {}) }));
        expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'input1',
            componentType: 'input',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    it('provides onValueChange callback', async () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, {}) }));
        expect(screen.getByRole('textbox')).toBeInTheDocument();
        await userEvent.type(screen.getByRole('textbox'), 'test');
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'input1',
            componentType: 'input',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('should use the nearest context', async () => {
        const mockUseOnEventCallbackFar = jest.fn();
        const mockUseOnEventCallbackNear = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallbackFar, children: _jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallbackNear, children: _jsx(MockChildComponent, {}) }) }));
        expect(screen.getByText('Click me')).toBeInTheDocument();
        expect(screen.getByText('View me')).toBeInTheDocument();
        await userEvent.click(screen.getByText('Click me'));
        await userEvent.click(screen.getByText('View me'));
        expect(mockUseOnEventCallbackFar).not.toHaveBeenCalled();
        expect(mockUseOnEventCallbackNear).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallbackNear).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    it('handles absence of callbacks', async () => {
        render(_jsx(MockChildComponent, {}));
        expect(screen.getByText('Click me')).toBeInTheDocument();
        expect(screen.getByText('View me')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBeInTheDocument();
        await userEvent.click(screen.getByText('Click me'));
        await userEvent.type(screen.getByRole('textbox'), 'test');
        expect(screen.getByText('Click me')).toBeInTheDocument();
        expect(screen.getByText('View me')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toHaveValue('test');
    });
    it('Overriding analyticsEvents behaves as expected', async () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView] }) }));
        await userEvent.click(screen.getByText('Click me'));
        expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'input1',
            componentType: 'input',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('provides onSubmit callback', async () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, {}) }));
        expect(screen.getByText('Submit')).toBeInTheDocument();
        await userEvent.click(screen.getByText('Submit'));
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onSubmit',
            componentId: 'form1',
            componentType: 'form',
            componentSubType: undefined,
            shouldStartInteraction: true,
            mode: 'default',
            value: undefined,
            event: expect.anything(),
            referrerComponent: { type: 'button', id: 'submit-button' },
            formPropertyValues: { initial: { input1: 123 }, final: { input1: 321 } },
        });
    });
    it('handles component view ids with callback', async () => {
        const mockUseOnEventCallback = jest.fn();
        render(_jsx(DesignSystemEventProvider, { callback: mockUseOnEventCallback, children: _jsx(MockChildComponent, { analyticsEvents: [
                    DesignSystemEventProviderAnalyticsEventTypes.OnClick,
                    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                    DesignSystemEventProviderAnalyticsEventTypes.OnView,
                ] }) }));
        expect(screen.getByText('Click me')).toBeInTheDocument();
        expect(screen.getByText('View me')).toBeInTheDocument();
        expect(screen.getByRole('textbox')).toBeInTheDocument();
        expect(mockUseOnEventCallback.mock.calls.length).toBe(3);
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'input1',
            componentType: 'input',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        const button1ComponentViewId = mockUseOnEventCallback.mock.calls[0][0]
            .componentViewId;
        const button2ComponentViewId = mockUseOnEventCallback.mock.calls[1][0]
            .componentViewId;
        const input1ComponentViewId = mockUseOnEventCallback.mock.calls[2][0]
            .componentViewId;
        expect(button1ComponentViewId).not.toBe(undefined);
        expect(button2ComponentViewId).not.toBe(undefined);
        expect(input1ComponentViewId).not.toBe(undefined);
        expect(button1ComponentViewId).not.toEqual(button2ComponentViewId);
        expect(button1ComponentViewId).not.toEqual(input1ComponentViewId);
        expect(button2ComponentViewId).not.toEqual(input1ComponentViewId);
        await userEvent.click(screen.getByText('Click me'));
        await userEvent.click(screen.getByText('View me'));
        await userEvent.type(screen.getByRole('textbox'), 'a');
        expect(mockUseOnEventCallback.mock.calls.length).toBe(6);
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button1',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'button2',
            componentType: 'button',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
        });
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'input1',
            componentType: 'input',
            componentViewId: expect.stringMatching(/testuuid-00v4-0000-0000-00000000000_\d+/),
            shouldStartInteraction: false,
            value: undefined,
        });
        const button1ComponentViewId2 = mockUseOnEventCallback.mock.calls[3][0]
            .componentViewId;
        const button2ComponentViewId2 = mockUseOnEventCallback.mock.calls[4][0]
            .componentViewId;
        const input1ComponentViewId2 = mockUseOnEventCallback.mock.calls[5][0]
            .componentViewId;
        // Component view id for click/value_change is the same component view id for view
        expect(button1ComponentViewId).toEqual(button1ComponentViewId2);
        expect(button2ComponentViewId).toEqual(button2ComponentViewId2);
        expect(input1ComponentViewId).toEqual(input1ComponentViewId2);
    });
});
//# sourceMappingURL=DesignSystemEventProvider.test.js.map