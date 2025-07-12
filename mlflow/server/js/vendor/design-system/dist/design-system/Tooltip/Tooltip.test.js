import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Tooltip } from './Tooltip';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';
describe('Tooltip', () => {
    const eventCallback = jest.fn();
    const tooltipComponentId = 'tooltip-component-id';
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const renderComponent = () => {
        return render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsx(Tooltip, { content: "Hello world", componentId: tooltipComponentId, children: _jsx("button", { children: "Target" }) }) }) }));
    };
    const getButton = () => screen.getByRole('button', { name: 'Target' });
    const getTooltip = () => screen.getByRole('tooltip', { name: 'Hello world' });
    const queryTooltip = () => screen.queryByRole('tooltip', { name: 'Hello world' });
    it('renders tooltip on hover', async () => {
        render(_jsx(DesignSystemProvider, { children: _jsx(Tooltip, { componentId: "codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_29", content: "Hello world", children: _jsx("button", { children: "Target" }) }) }));
        await userEvent.hover(screen.getByRole('button', { name: 'Target' }));
        await waitFor(() => expect(screen.getByRole('tooltip', { name: 'Hello world' })).toBeInTheDocument());
    });
    it('renders tooltip on focus', async () => {
        render(_jsx(DesignSystemProvider, { children: _jsx(Tooltip, { componentId: "codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_42", content: "Hello world", children: _jsx("button", { children: "Target" }) }) }));
        const trigger = screen.getByRole('button', { name: 'Target' });
        act(() => {
            trigger.focus();
        });
        await waitFor(() => expect(screen.getByRole('tooltip', { name: 'Hello world' })).toBeInTheDocument());
    });
    it('does not render tooltip with null or undefined content', async () => {
        render(_jsxs(DesignSystemProvider, { children: [_jsx(Tooltip, { componentId: "codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_60", content: null, children: _jsx("button", { children: "null button" }) }), _jsx(Tooltip, { componentId: "codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_63", content: undefined, children: _jsx("button", { children: "undefined button" }) })] }));
        await userEvent.hover(screen.getByRole('button', { name: 'null button' }));
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
        await userEvent.hover(screen.getByRole('button', { name: 'undefined button' }));
        expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });
    it('emit onView event when tooltip is hovered', async () => {
        const { container } = renderComponent();
        expect(eventCallback).not.toHaveBeenCalled();
        await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
        // first hover: emits onView event
        await userEvent.hover(getButton());
        await waitFor(() => {
            expect(getTooltip()).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: tooltipComponentId,
            componentType: 'tooltip',
            shouldStartInteraction: false,
            value: undefined,
        });
        // JSDOM limitation: must unhover, then click away to hide tooltip and allow future hovers to trigger tooltip
        await userEvent.unhover(screen.getByRole('button', { name: 'Target' }));
        await userEvent.click(container);
        await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
        expect(eventCallback).toHaveBeenCalledTimes(1);
        // second hover: does not emit new onView event
        await userEvent.hover(getButton());
        await waitFor(() => {
            expect(getTooltip()).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
    });
    it('emit onView event when tooltip is focused', async () => {
        renderComponent();
        expect(eventCallback).not.toHaveBeenCalled();
        const trigger = getButton();
        await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
        // first focus: emits onView event
        act(() => {
            trigger.focus();
        });
        await waitFor(() => {
            expect(getTooltip()).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: tooltipComponentId,
            componentType: 'tooltip',
            shouldStartInteraction: false,
            value: undefined,
        });
        act(() => {
            trigger.blur();
        });
        await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
        expect(eventCallback).toHaveBeenCalledTimes(1);
        // second focus: does not emit new onView event
        act(() => {
            trigger.focus();
        });
        await waitFor(() => {
            expect(getTooltip()).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
    });
});
//# sourceMappingURL=Tooltip.test.js.map