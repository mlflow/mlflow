import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { screen, render, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Accordion } from './index';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Accordion onChange event emits correct values', () => {
    const eventCallback = jest.fn();
    const onChangeSpy = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const TestAccordion = ({ displayMode, valueHasNoPii, }) => {
        return (_jsxs(Accordion, { displayMode: displayMode, componentId: "accordion_test", valueHasNoPii: valueHasNoPii, onChange: onChangeSpy, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], children: [_jsx(Accordion.Panel, { header: "Section 1", children: "foo" }, "1"), _jsx(Accordion.Panel, { header: "Section 2", children: "bar" }, "2")] }));
    };
    it('emits accordion event with empty value when valueHasNoPii is false', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TestAccordion, { displayMode: "single" }) }));
        expect(eventCallback).not.toHaveBeenCalled();
        expect(onChangeSpy).not.toHaveBeenCalled();
        const sections = screen.getAllByRole('tab');
        const section1 = sections[0];
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onChangeSpy).toHaveBeenCalledWith('1');
    });
    it('emits accordion events where the accordion only allows one panel open at a time', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TestAccordion, { displayMode: "single", valueHasNoPii: true }) }));
        expect(eventCallback).not.toHaveBeenCalled();
        expect(onChangeSpy).not.toHaveBeenCalled();
        // Retrieve the sections to click on and verify that the sections are not expanded.
        const sections = screen.getAllByRole('tab');
        const section1 = sections[0];
        const section2 = sections[1];
        expect(section1).toBeInTheDocument();
        expect(section2).toBeInTheDocument();
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
        // Click on the first section and verify that it is expanded.
        await userEvent.click(section1);
        expect(section1).toHaveAttribute('aria-expanded', 'true');
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '1',
        });
        expect(onChangeSpy).toHaveBeenCalledWith('1');
        // Close the first section and verify that it is no longer expanded.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(onChangeSpy).toHaveBeenCalledWith(undefined);
        // Click on the second section and verify that it is expanded.
        await userEvent.click(section2);
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '2',
        });
        expect(onChangeSpy).toHaveBeenCalledWith('2');
        expect(section2).toHaveAttribute('aria-expanded', 'true');
        // Click on the first section and verify that the second section is closed and the first section is open.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '1',
        });
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(onChangeSpy).toHaveBeenCalledWith('1');
        expect(section1).toHaveAttribute('aria-expanded', 'true');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
    });
    it('emits accordion events where the accordion allows multiple sections open at a time', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TestAccordion, { displayMode: "multiple", valueHasNoPii: true }) }));
        expect(eventCallback).not.toHaveBeenCalled();
        expect(onChangeSpy).not.toHaveBeenCalled();
        // Retrieve the sections to click on and verify that the sections are not expanded.
        const sections = screen.getAllByRole('button');
        const section1 = sections[0];
        const section2 = sections[1];
        expect(section1).toBeInTheDocument();
        expect(section2).toBeInTheDocument();
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
        // Click on the first section and verify that it is expanded.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '["1"]',
        });
        expect(onChangeSpy).toHaveBeenCalledWith(['1']);
        // Close the first section and verify that it is not expanded.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '[]',
        });
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        // Click on the second section and verify that it is expanded.
        await userEvent.click(section2);
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '["2"]',
        });
        expect(onChangeSpy).toHaveBeenCalledWith(['2']);
        // Click on the first section and verify that both sections are expanded.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(section1).toHaveAttribute('aria-expanded', 'true');
        expect(section2).toHaveAttribute('aria-expanded', 'true');
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '["2","1"]',
        });
        // Close both sections and verify that both sections are no longer expanded.
        await userEvent.click(section1);
        await userEvent.click(section2);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: '[]',
        });
        expect(onChangeSpy).toHaveBeenCalledWith([]);
        expect(eventCallback).toHaveBeenCalledTimes(6);
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
    });
});
describe('Accordion emits default onView event', () => {
    const eventCallback = jest.fn();
    const onChangeSpy = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const { setSafex } = setupSafexTesting();
    const TestAccordion = () => {
        return (_jsxs(Accordion, { componentId: "accordion_test", displayMode: "single", onChange: onChangeSpy, children: [_jsx(Accordion.Panel, { header: "Section 1", children: "foo" }, "1"), _jsx(Accordion.Panel, { header: "Section 2", children: "bar" }, "2")] }));
    };
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.accordion': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('emits one onView event', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(TestAccordion, {}) }));
        await waitFor(() => {
            expect(screen.getByText('Section 1')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
            value: undefined,
        });
        expect(onChangeSpy).not.toHaveBeenCalled();
        // Retrieve the sections to click on and verify that the sections are not expanded.
        const sections = screen.getAllByRole('tab');
        const section1 = sections[0];
        const section2 = sections[1];
        expect(section1).toBeInTheDocument();
        expect(section2).toBeInTheDocument();
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
        // Click on the first section and verify that it is expanded.
        await userEvent.click(section1);
        expect(section1).toHaveAttribute('aria-expanded', 'true');
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
        });
        expect(onChangeSpy).toHaveBeenCalledTimes(1);
        // Close the first section and verify that it is no longer expanded.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
        });
        expect(section1).toHaveAttribute('aria-expanded', 'false');
        expect(onChangeSpy).toHaveBeenCalledTimes(2);
        // Click on the second section and verify that it is expanded.
        await userEvent.click(section2);
        expect(eventCallback).toHaveBeenCalledTimes(4);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
        });
        expect(onChangeSpy).toHaveBeenCalledTimes(3);
        expect(section2).toHaveAttribute('aria-expanded', 'true');
        // Click on the first section and verify that the second section is closed and the first section is open.
        await userEvent.click(section1);
        expect(eventCallback).toHaveBeenNthCalledWith(5, {
            eventType: 'onValueChange',
            componentId: 'accordion_test',
            componentType: 'accordion',
            shouldStartInteraction: false,
        });
        expect(eventCallback).toHaveBeenCalledTimes(5);
        expect(onChangeSpy).toHaveBeenCalledTimes(4);
        expect(section1).toHaveAttribute('aria-expanded', 'true');
        expect(section2).toHaveAttribute('aria-expanded', 'false');
    });
});
//# sourceMappingURL=Accordion.test.js.map