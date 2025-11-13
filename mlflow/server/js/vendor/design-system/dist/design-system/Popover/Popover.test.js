import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Popover } from '.';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';
describe('Popover', function () {
    function renderComponent() {
        return render(_jsx(DesignSystemProvider, { children: _jsxs(Popover.Root, { componentId: "codegen_design-system_src_design-system_popover_popover.test.tsx_13", children: [_jsx(Popover.Trigger, { asChild: true, children: _jsx(Button, { componentId: "codegen_design-system_src_design-system_popover_popover.test.tsx_14", "data-testid": "test-trigger", children: "Default" }) }), _jsx(Popover.Content, { align: "start", children: "Popover content" })] }) }));
    }
    // This is a trivial re-test of Radix's tests, but is provided as an
    // example of how to test the Popover component.
    it('renders popover', async () => {
        renderComponent();
        await userEvent.click(screen.getByTestId('test-trigger'));
        expect(screen.queryByText('Popover content')).not.toBeNull();
        await userEvent.keyboard('{Escape}');
        expect(screen.queryByText('Popover content')).toBeNull();
    });
    it('renders popover & emit analytics event', async () => {
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(Popover.Root, { componentId: "test-popover", analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView], children: [_jsx(Popover.Trigger, { asChild: true, children: _jsx(Button, { componentId: "codegen_design-system_src_design-system_popover_popover.test.tsx_14", "data-testid": "test-trigger", children: "Default" }) }), _jsx(Popover.Content, { align: "start", children: "Popover content" })] }) }) }));
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByTestId('test-trigger'));
        expect(screen.queryByText('Popover content')).not.toBeNull();
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onClick',
            componentId: 'codegen_design-system_src_design-system_popover_popover.test.tsx_14',
            componentType: 'button',
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
            isInteractionSubject: true,
        });
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onView',
            componentId: 'test-popover',
            componentType: 'popover',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    // Add these new test cases to the existing test file:
    it('emits view event when popover is opened via open prop', async () => {
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(Popover.Root, { componentId: "test-popover", open: true, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView], children: [_jsx(Popover.Trigger, { asChild: true, children: _jsx(Button, { componentId: "popover-test-id", "data-testid": "test-trigger", children: "Default" }) }), _jsx(Popover.Content, { align: "start", children: "Popover content" })] }) }) }));
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'test-popover',
            componentType: 'popover',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=Popover.test.js.map