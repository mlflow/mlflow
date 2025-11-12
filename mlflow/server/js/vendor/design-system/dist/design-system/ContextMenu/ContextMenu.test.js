import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ContextMenu } from './ContextMenu';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('ContextMenu', () => {
    const { setSafex } = setupSafexTesting();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const CommonComponent = ({ children }) => (_jsx(DesignSystemEventProviderForTest, { children: _jsxs(ContextMenu.Root, { children: [_jsx(ContextMenu.Trigger, { children: "Trigger" }), _jsx(ContextMenu.Content, { children: children })] }) }));
    describe.each([false, true])('Shared ContextMenu tests: databricks.fe.observability.defaultComponentView.contextMenu set to %s', (defaultComponentView) => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.contextMenu': defaultComponentView,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        it('emits click event for item', async () => {
            const onClickSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsx(ContextMenu.Item, { componentId: "context_menu_item_test", onClick: onClickSpy, children: "Item" }) }));
            expect(onClickSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitem', { name: 'Item' }));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(onClickSpy).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledTimes(eventCallbackCountOnRender + 1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onClick',
                componentId: 'context_menu_item_test',
                componentType: 'context_menu_item',
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: undefined,
            });
        });
        it('does not emit click event for item when asChild is set', async () => {
            const onClickSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsx(ContextMenu.Item, { componentId: "context_menu_item_test", onClick: onClickSpy, asChild: true, children: _jsx("button", { children: "Item" }) }) }));
            expect(onClickSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitem', { name: 'Item' }));
            expect(onClickSpy).toHaveBeenCalledTimes(1);
            expect(eventCallback).not.toHaveBeenCalled();
        });
        it('emits value change event for checkbox item', async () => {
            const onCheckedChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsx(ContextMenu.CheckboxItem, { componentId: "context_menu_checkbox_item_test", onCheckedChange: onCheckedChangeSpy, children: "Checkbox Item" }) }));
            expect(onCheckedChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Checkbox Item' }));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(onCheckedChangeSpy).toHaveBeenCalledWith(true);
            expect(eventCallback).toHaveBeenCalledTimes(eventCallbackCountOnRender + 1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onValueChange',
                componentId: 'context_menu_checkbox_item_test',
                componentType: 'context_menu_checkbox_item',
                shouldStartInteraction: false,
                value: true,
            });
        });
        it('emits value change for radio group', async () => {
            const onValueChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsxs(ContextMenu.RadioGroup, { componentId: "context_menu_radio_group_test", valueHasNoPii: true, onValueChange: onValueChangeSpy, children: [_jsx(ContextMenu.RadioItem, { value: "one", children: "Radio Item 1" }), _jsx(ContextMenu.RadioItem, { value: "two", children: "Radio Item 2" })] }) }));
            expect(onValueChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(onValueChangeSpy).toHaveBeenCalledWith('one');
            expect(eventCallback).toHaveBeenCalledTimes(eventCallbackCountOnRender + 1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: 'one',
            });
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 2' }));
            expect(onValueChangeSpy).toHaveBeenCalledWith('two');
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: 'two',
            });
        });
        it('emits value change event without value for radio group when valueHasNoPii is not set', async () => {
            const onValueChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsxs(ContextMenu.RadioGroup, { componentId: "context_menu_radio_group_test", onValueChange: onValueChangeSpy, children: [_jsx(ContextMenu.RadioItem, { value: "one", children: "Radio Item 1" }), _jsx(ContextMenu.RadioItem, { value: "two", children: "Radio Item 2" })] }) }));
            expect(onValueChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
            const eventCallbackCountOnRender = defaultComponentView ? 1 : 0;
            expect(onValueChangeSpy).toHaveBeenCalledWith('one');
            expect(eventCallback).toHaveBeenCalledTimes(eventCallbackCountOnRender + 1);
            expect(eventCallback).toHaveBeenNthCalledWith(eventCallbackCountOnRender + 1, {
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
        });
    });
    describe('Default ContextMenu component_view tests', () => {
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.contextMenu': true,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        it('emits click event for item', async () => {
            const onClickSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsx(ContextMenu.Item, { componentId: "context_menu_item_test", onClick: onClickSpy, children: "Item" }) }));
            expect(onClickSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'context_menu_item_test',
                componentType: 'context_menu_item',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByRole('menuitem', { name: 'Item' }));
            expect(onClickSpy).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onClick',
                componentId: 'context_menu_item_test',
                componentType: 'context_menu_item',
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: undefined,
            });
            // have context menu disappear
            act(() => {
                screen.getByText('Trigger').blur();
            });
            expect(screen.queryAllByRole('menuitem')).toHaveLength(0);
            // click again to open the menu
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(screen.queryAllByRole('menuitem')).toHaveLength(1);
            // each open triggers new view events
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onView',
                componentId: 'context_menu_item_test',
                componentType: 'context_menu_item',
                shouldStartInteraction: false,
            });
        });
        it('emits value change event for checkbox item', async () => {
            const onCheckedChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsx(ContextMenu.CheckboxItem, { componentId: "context_menu_checkbox_item_test", onCheckedChange: onCheckedChangeSpy, children: "Checkbox Item" }) }));
            expect(onCheckedChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'context_menu_checkbox_item_test',
                componentType: 'context_menu_checkbox_item',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Checkbox Item' }));
            expect(onCheckedChangeSpy).toHaveBeenCalledWith(true);
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onValueChange',
                componentId: 'context_menu_checkbox_item_test',
                componentType: 'context_menu_checkbox_item',
                shouldStartInteraction: false,
                value: true,
            });
            // have context menu disappear
            act(() => {
                screen.getByText('Trigger').blur();
            });
            expect(screen.queryAllByRole('menuitemcheckbox')).toHaveLength(0);
            // click again to open the menu
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(screen.queryAllByRole('menuitemcheckbox')).toHaveLength(1);
            // each open triggers new view events
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onView',
                componentId: 'context_menu_checkbox_item_test',
                componentType: 'context_menu_checkbox_item',
                shouldStartInteraction: false,
            });
        });
        it('emits value change for radio group', async () => {
            const onValueChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsxs(ContextMenu.RadioGroup, { componentId: "context_menu_radio_group_test", valueHasNoPii: true, onValueChange: onValueChangeSpy, children: [_jsx(ContextMenu.RadioItem, { value: "one", children: "Radio Item 1" }), _jsx(ContextMenu.RadioItem, { value: "two", children: "Radio Item 2" })] }) }));
            expect(onValueChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
            expect(onValueChangeSpy).toHaveBeenCalledWith('one');
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: 'one',
            });
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(eventCallback).toHaveBeenCalledTimes(3);
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 2' }));
            expect(onValueChangeSpy).toHaveBeenCalledWith('two');
            expect(eventCallback).toHaveBeenCalledTimes(4);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: 'two',
            });
            // have context menu disappear
            act(() => {
                screen.getByText('Trigger').blur();
            });
            expect(screen.queryAllByRole('menuitemradio')).toHaveLength(0);
            // click again to open the menu
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(screen.queryAllByRole('menuitemradio')).toHaveLength(2);
            // each open triggers new view events
            expect(eventCallback).toBeCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onView',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
            });
        });
        it('emits value change event without value for radio group when valueHasNoPii is not set', async () => {
            const onValueChangeSpy = jest.fn();
            render(_jsx(CommonComponent, { children: _jsxs(ContextMenu.RadioGroup, { componentId: "context_menu_radio_group_test", onValueChange: onValueChangeSpy, children: [_jsx(ContextMenu.RadioItem, { value: "one", children: "Radio Item 1" }), _jsx(ContextMenu.RadioItem, { value: "two", children: "Radio Item 2" })] }) }));
            expect(onValueChangeSpy).not.toHaveBeenCalled();
            expect(eventCallback).not.toHaveBeenCalled();
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
            });
            await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
            expect(onValueChangeSpy).toHaveBeenCalledWith('one');
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onValueChange',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
                value: undefined,
            });
            // have context menu disappear
            act(() => {
                screen.getByText('Trigger').blur();
            });
            expect(screen.queryAllByRole('menuitemradio')).toHaveLength(0);
            // click again to open the menu
            fireEvent.contextMenu(screen.getByText('Trigger'));
            expect(screen.queryAllByRole('menuitemradio')).toHaveLength(2);
            // each open triggers new view events
            expect(eventCallback).toBeCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onView',
                componentId: 'context_menu_radio_group_test',
                componentType: 'context_menu_radio_group',
                shouldStartInteraction: false,
            });
        });
    });
});
//# sourceMappingURL=ContextMenu.test.js.map