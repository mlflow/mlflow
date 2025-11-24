import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DropdownMenu } from '.';
import { openDropdownMenu } from '../../test-utils/rtl';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider/DesignSystemEventProviderTestUtils';
import { DesignSystemProvider } from '../DesignSystemProvider';
import { setupSafexTesting } from '../utils/safex';
describe('DropdownMenu', function () {
    const { setSafex } = setupSafexTesting();
    function renderComponent() {
        return render(_jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsxs(DropdownMenu.Sub, { children: [_jsx(DropdownMenu.SubTrigger, { children: "Option 1" }), _jsxs(DropdownMenu.SubContent, { children: [_jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_27", children: "Option 1a" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_28", children: "Option 1b" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_29", children: "Option 1c" })] })] }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_32", children: "Option 2" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_33", children: "Option 3" })] })] }) }));
    }
    function renderDisabledComponentWithTooltip(onClick) {
        return render(_jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_39", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsxs(DropdownMenu.Sub, { children: [_jsx(DropdownMenu.SubTrigger, { children: "Option 1" }), _jsxs(DropdownMenu.SubContent, { children: [_jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_56", children: "Option 1a" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_57", children: "Option 1b" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_58", children: "Option 1c" })] })] }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_61", "data-testid": "test-disableditem", disabled: true, disabledReason: "Option disabled reason", onClick: onClick, children: "Option 2" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_69", children: "Option 3" })] })] }) }));
    }
    describe.each([false, true])('Shared DropdownMenu tests: databricks.fe.observability.defaultComponentView.dropdownMenu set to %s', (defaultComponentView) => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.dropdownMenu': defaultComponentView,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        // This is a trivial re-test of Radix's tests, but is provided as an
        // example of how to test the DropdownMenu component.
        it('renders proper number of menuitem(s) with proper text', async () => {
            renderComponent();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(3);
            expect(screen.queryByText('Option 1')).not.toBeNull();
            expect(screen.queryByText('Option 2')).not.toBeNull();
            expect(screen.queryByText('Option 3')).not.toBeNull();
            // This is known to not work correctly in `@testing-library/user-event <=13.5.0.
            await userEvent.click(screen.getByText('Option 1'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(6);
            await userEvent.keyboard('{Escape}');
            expect(screen.queryAllByRole('menuitem')).toHaveLength(0);
        });
        it("doesn't trigger click on tooltip event when disabled", async () => {
            const onClick = jest.fn();
            renderDisabledComponentWithTooltip(onClick);
            openDropdownMenu(screen.getByTestId('test-menubutton'));
            await waitFor(() => {
                expect(screen.getByTestId('test-disableditem')).toBeVisible();
            });
            userEvent.click(screen.getByTestId('test-disableditem').querySelector('span'));
            expect(onClick).not.toHaveBeenCalled();
        });
        it('emits analytics events for item', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.Item, { onClick: handleClick, componentId: "OPTION_A_TEST", children: "Option A" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_132", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).toBeCalledTimes(0);
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            await userEvent.click(screen.getByText('Option A'));
            const eventCallbackCountOnRender = defaultComponentView ? 2 : 0;
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onClick',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_item',
                componentSubType: undefined,
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: true,
            });
        });
        it('does not emit analytics events for menu item with asChild set', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.Item, { onClick: handleClick, componentId: "OPTION_A_TEST", asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "OPTION_A_TEST_CHILD", children: "Option A" }) }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_174", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            await userEvent.click(screen.getByText('Option A'));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledTimes(eventCallbackCountOnRender + 1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onClick',
                componentId: 'OPTION_A_TEST_CHILD',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
        });
        it('emits analytics events for checkbox', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.CheckboxItem, { onCheckedChange: handleClick, componentId: "OPTION_A_TEST", children: "Option A" }), _jsx(DropdownMenu.CheckboxItem, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_218", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            await userEvent.click(screen.getByText('Option A'));
            const eventCallbackCountOnRender = defaultComponentView ? 2 : 0;
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onValueChange',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
                value: true,
            });
        });
        it('emits analytics events for radio group', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsx(DropdownMenu.Content, { align: "start", children: _jsxs(DropdownMenu.RadioGroup, { componentId: "OPTION_RADIO_GROUP", children: [_jsx(DropdownMenu.RadioItem, { value: "A", onClick: handleClick, children: "Option A" }), _jsx(DropdownMenu.RadioItem, { value: "B", children: "Option B" })] }) })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            await userEvent.click(screen.getByText('Option A'));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onValueChange',
                componentId: 'OPTION_RADIO_GROUP',
                componentType: 'dropdown_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
        });
    });
    describe('Default DropdownMenu component_view tests', () => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.dropdownMenu': true,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        it('emits analytics events for item', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.Item, { onClick: handleClick, componentId: "OPTION_A_TEST", children: "Option A" }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_132", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).toBeCalledTimes(0);
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            expect(eventCallback).toBeCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_132',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByText('Option A'));
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onClick',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_item',
                componentSubType: undefined,
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: true,
            });
            // have dropdown menu disappear
            act(() => {
                screen.getByTestId('test-menubutton').blur();
            });
            expect(screen.queryAllByRole('menuitem')).toHaveLength(0);
            // click again to open the menu
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            // each open triggers a new view event
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onView',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_132',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
        });
        it('does not emit analytics events for menu item with asChild set', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.Item, { onClick: handleClick, componentId: "OPTION_A_TEST", asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "OPTION_A_TEST_CHILD", children: "Option A" }) }), _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_174", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            expect(eventCallback).toBeCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_174',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByText('Option A'));
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onClick',
                componentId: 'OPTION_A_TEST_CHILD',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.anything(),
            });
            // have dropdown menu disappear
            act(() => {
                screen.getByTestId('test-menubutton').blur();
            });
            expect(screen.queryAllByRole('menuitem')).toHaveLength(0);
            // click again to open the menu
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
            // each open triggers a new view event
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_174',
                componentType: 'dropdown_menu_item',
                shouldStartInteraction: false,
            });
        });
        it('emits analytics events for checkbox', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsxs(DropdownMenu.Content, { align: "start", children: [_jsx(DropdownMenu.CheckboxItem, { onCheckedChange: handleClick, componentId: "OPTION_A_TEST", children: "Option A" }), _jsx(DropdownMenu.CheckboxItem, { componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_218", children: "Option B" })] })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
            });
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_218',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByText('Option A'));
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onValueChange',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
                value: true,
            });
            // have dropdown menu disappear
            act(() => {
                screen.getByTestId('test-menubutton').blur();
            });
            expect(eventCallback).toBeCalledTimes(3);
            // click again to open the menu
            await userEvent.click(screen.getByTestId('test-menubutton'));
            // each open triggers a new view event
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onView',
                componentId: 'OPTION_A_TEST',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
            });
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onView',
                componentId: 'codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_218',
                componentType: 'dropdown_menu_checkbox_item',
                shouldStartInteraction: false,
            });
        });
        it('emits analytics events for radio group', async () => {
            const handleClick = jest.fn();
            const eventCallback = jest.fn();
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], componentId: "codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15", "data-testid": "test-menubutton", children: "Default" }) }), _jsx(DropdownMenu.Content, { align: "start", children: _jsxs(DropdownMenu.RadioGroup, { componentId: "OPTION_RADIO_GROUP", children: [_jsx(DropdownMenu.RadioItem, { value: "A", onClick: handleClick, children: "Option A" }), _jsx(DropdownMenu.RadioItem, { value: "B", children: "Option B" })] }) })] }) }) }));
            expect(handleClick).not.toBeCalled();
            expect(eventCallback).not.toBeCalled();
            await userEvent.click(screen.getByTestId('test-menubutton'));
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'OPTION_RADIO_GROUP',
                componentType: 'dropdown_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
            await userEvent.click(screen.getByText('Option A'));
            expect(handleClick).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onValueChange',
                componentId: 'OPTION_RADIO_GROUP',
                componentType: 'dropdown_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
            // have dropdown menu disappear
            act(() => {
                screen.getByTestId('test-menubutton').blur();
            });
            expect(eventCallback).toBeCalledTimes(2);
            // click again to open the menu
            await userEvent.click(screen.getByTestId('test-menubutton'));
            // expect no new view events
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onView',
                componentId: 'OPTION_RADIO_GROUP',
                componentType: 'dropdown_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
        });
    });
});
//# sourceMappingURL=DropdownMenu.test.js.map