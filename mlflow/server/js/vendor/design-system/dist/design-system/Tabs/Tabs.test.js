import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, beforeEach, jest } from '@jest/globals';
import { act, render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Tabs } from '.';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider/DesignSystemEventProviderTestUtils';
import { setupSafexTesting } from '../utils/safex';
describe('Tabs', () => {
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const UncontrolledTabs = ({ valueHasNoPii }) => (_jsxs(Tabs.Root, { componentId: "TABS_TEST", defaultValue: "tab1", valueHasNoPii: valueHasNoPii, children: [_jsxs(Tabs.List, { children: [_jsx(Tabs.Trigger, { value: "tab1", children: "Tab 1" }), _jsx(Tabs.Trigger, { value: "tab2", children: "Tab 2" }), _jsx(Tabs.Trigger, { value: "tab3", children: "Tab 3" })] }), _jsx(Tabs.Content, { value: "tab1", children: "Tab 1 Content" }), _jsx(Tabs.Content, { value: "tab2", children: "Tab 2 Content" }), _jsx(Tabs.Content, { value: "tab3", children: "Tab 3 Content" })] }));
    const ControlledTabs = ({ valueHasNoPii }) => {
        const [tabs, setTabs] = useState([
            { value: 'tab1', title: 'Tab 1', content: 'Tab 1 Content' },
            { value: 'tab2', title: 'Tab 2', content: 'Tab 2 Content' },
        ]);
        const [activeTab, setActiveTab] = useState('tab1');
        const [nextTabNumber, setNextTabNumber] = useState(3);
        return (_jsxs(Tabs.Root, { componentId: "TABS_TEST", value: activeTab, onValueChange: setActiveTab, valueHasNoPii: valueHasNoPii, children: [_jsx(Tabs.List, { addButtonProps: {
                        onClick: () => {
                            const newTab = {
                                value: `tab${nextTabNumber}`,
                                title: `Tab ${nextTabNumber}`,
                                content: `Tab ${nextTabNumber} Content`,
                            };
                            setTabs(tabs.concat(newTab));
                            setActiveTab(newTab.value);
                            setNextTabNumber(nextTabNumber + 1);
                        },
                    }, children: tabs.map((tab) => (_jsx(Tabs.Trigger, { value: tab.value, onClose: (value) => {
                            const newTabs = tabs.filter((tab) => tab.value !== value);
                            setTabs(newTabs);
                            if (activeTab === value && newTabs.length > 0) {
                                setActiveTab(newTabs[0].value);
                            }
                        }, children: tab.title }, tab.value))) }), tabs.map((tab) => (_jsx(Tabs.Content, { value: tab.value, children: tab.content }, tab.value)))] }));
    };
    it('renders a set of tabs', async () => {
        render(_jsx(UncontrolledTabs, {}));
        const tabList = screen.getByRole('tablist');
        const tabs = within(tabList).getAllByRole('tab');
        expect(tabs).toHaveLength(3);
        expect(tabs[0]).toHaveTextContent('Tab 1');
        expect(tabs[1]).toHaveTextContent('Tab 2');
        expect(tabs[2]).toHaveTextContent('Tab 3');
        expect(screen.getByText('Tab 1 Content')).toBeInTheDocument();
        expect(screen.queryByText('Tab 2 Content')).not.toBeInTheDocument();
        await userEvent.click(tabs[1]);
        expect(screen.getByText('Tab 2 Content')).toBeInTheDocument();
        expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();
    });
    it('disabled tabs cannot be selected', async () => {
        render(_jsxs(Tabs.Root, { componentId: "TABS_TEST", defaultValue: "tab1", children: [_jsxs(Tabs.List, { children: [_jsx(Tabs.Trigger, { value: "tab1", children: "Tab 1" }), _jsx(Tabs.Trigger, { value: "tab2", disabled: true, children: "Tab 2" })] }), _jsx(Tabs.Content, { value: "tab1", children: "Tab 1 Content" }), _jsx(Tabs.Content, { value: "tab2", children: "Tab 2 Content" })] }));
        const tabList = screen.getByRole('tablist');
        const tabs = within(tabList).getAllByRole('tab');
        expect(tabs[1]).toBeDisabled();
    });
    it('tabs can be added and closed', async () => {
        render(_jsx(ControlledTabs, {}));
        expect(screen.queryByText('Tab 1 Content')).toBeInTheDocument();
        const addButton = screen.getByRole('button', { name: 'Add tab' });
        expect(addButton).toBeInTheDocument();
        await userEvent.click(addButton);
        let tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
        expect(tabs).toHaveLength(3);
        expect(screen.getByText('Tab 3 Content')).toBeInTheDocument();
        expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();
        act(() => {
            tabs[2].focus();
        });
        await userEvent.keyboard('{Delete}');
        tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
        expect(tabs).toHaveLength(2);
        expect(screen.getByText('Tab 1 Content')).toBeInTheDocument();
        expect(screen.queryByText('Tab 3 Content')).not.toBeInTheDocument();
        await userEvent.pointer({ target: tabs[0], keys: '[MouseMiddle]' });
        tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
        expect(tabs).toHaveLength(1);
        expect(screen.getByText('Tab 2 Content')).toBeInTheDocument();
        expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();
    });
    describe('Analytics Events', () => {
        const { setSafex } = setupSafexTesting();
        beforeEach(() => {
            setSafex({
                'databricks.fe.observability.defaultComponentView.button': true,
                'databricks.fe.observability.defaultComponentView.tabs': true,
            });
            // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
            window.IntersectionObserver = undefined;
        });
        it('uncontrolled tabs emit value change events without value', async () => {
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(UncontrolledTabs, {}) }));
            await waitFor(() => {
                expect(screen.getByRole('tablist')).toBeVisible();
            });
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: undefined,
            });
            const tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
            await userEvent.click(tabs[1]);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onValueChange',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: undefined,
            });
        });
        it('uncontrolled tabs emit value change events with value', async () => {
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(UncontrolledTabs, { valueHasNoPii: true }) }));
            await waitFor(() => {
                expect(screen.getByRole('tablist')).toBeVisible();
            });
            expect(eventCallback).toHaveBeenCalledTimes(1);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: 'tab1',
            });
            const tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
            await userEvent.click(tabs[1]);
            expect(eventCallback).toHaveBeenCalledWith({
                eventType: 'onValueChange',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: 'tab2',
            });
        });
        it('controlled tabs emit value change and on click events', async () => {
            render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(ControlledTabs, { valueHasNoPii: true }) }));
            await waitFor(() => {
                expect(screen.getByRole('tablist')).toBeVisible();
            });
            expect(eventCallback).toHaveBeenCalledTimes(2);
            expect(eventCallback).toHaveBeenNthCalledWith(1, {
                eventType: 'onView',
                componentId: 'TABS_TEST.add_tab',
                componentType: 'button',
                componentSubType: undefined,
                shouldStartInteraction: false,
            });
            expect(eventCallback).toHaveBeenNthCalledWith(2, {
                eventType: 'onView',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: 'tab1',
            });
            const addButton = screen.getByRole('button', { name: 'Add tab' });
            await userEvent.click(addButton);
            expect(eventCallback).toHaveBeenCalledTimes(3);
            expect(eventCallback).toHaveBeenNthCalledWith(3, {
                eventType: 'onClick',
                componentId: 'TABS_TEST.add_tab',
                componentType: 'button',
                shouldStartInteraction: true,
                isInteractionSubject: true,
                value: undefined,
                event: expect.any(Object),
            });
            let tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
            await userEvent.click(tabs[0]);
            expect(eventCallback).toHaveBeenCalledTimes(4);
            expect(eventCallback).toHaveBeenNthCalledWith(4, {
                eventType: 'onValueChange',
                componentId: 'TABS_TEST',
                componentType: 'tabs',
                componentSubType: undefined,
                shouldStartInteraction: false,
                value: 'tab1',
            });
            const closeIcon = within(tabs[0]).getByLabelText('Press delete to close the tab');
            await userEvent.click(closeIcon);
            expect(eventCallback).toHaveBeenCalledTimes(5);
            expect(eventCallback).toHaveBeenNthCalledWith(5, {
                eventType: 'onClick',
                componentId: 'TABS_TEST.close_tab',
                componentType: 'button',
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: undefined,
            });
            tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
            act(() => {
                tabs[0].focus();
            });
            await userEvent.keyboard('{Delete}');
            expect(eventCallback).toHaveBeenCalledTimes(6);
            expect(eventCallback).toHaveBeenNthCalledWith(6, {
                eventType: 'onClick',
                componentId: 'TABS_TEST.close_tab',
                componentType: 'button',
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: undefined,
            });
            tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
            await userEvent.pointer({ target: tabs[0], keys: '[MouseMiddle]' });
            expect(eventCallback).toHaveBeenCalledTimes(7);
            expect(eventCallback).toHaveBeenNthCalledWith(7, {
                eventType: 'onClick',
                componentId: 'TABS_TEST.close_tab',
                componentType: 'button',
                shouldStartInteraction: true,
                value: undefined,
                event: expect.any(Object),
                isInteractionSubject: undefined,
            });
        });
    });
});
//# sourceMappingURL=Tabs.test.js.map