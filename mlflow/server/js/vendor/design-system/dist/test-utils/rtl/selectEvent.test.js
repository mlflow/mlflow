import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { expect, describe, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { selectEvent } from '.';
import { LegacySelect, DesignSystemProvider } from '../../design-system';
import { selectClasses } from '../common';
function renderSelect(extraProps) {
    return render(_jsx(DesignSystemProvider, { children: _jsxs(LegacySelect, { "aria-label": "Label", "data-testid": "dubois-select", value: "pie", ...extraProps, children: [_jsx(LegacySelect.Option, { value: "pie", children: "Pie" }), _jsx(LegacySelect.Option, { value: "bar", children: "Bar" }), _jsx(LegacySelect.Option, { value: "line", children: _jsx("div", { children: "Line" }) }), _jsx(LegacySelect.Option, { value: "bubble", children: "Bubble" })] }) }));
}
function renderMultiSelect(extraProps) {
    return render(_jsx(DesignSystemProvider, { children: _jsxs(LegacySelect, { allowClear: true, "aria-label": "Label", "data-testid": "dubois-select", mode: "multiple", value: ['pie', 'bar'], ...extraProps, children: [_jsx(LegacySelect.Option, { value: "pie", children: "Pie" }), _jsx(LegacySelect.Option, { value: "bar", children: "Bar" }), _jsx(LegacySelect.Option, { value: "line", children: _jsx("div", { children: "Line" }) }), _jsx(LegacySelect.Option, { value: "bubble", children: "Bubble" })] }) }));
}
function renderTagSelect(extraProps) {
    return render(_jsx(DesignSystemProvider, { children: _jsx(LegacySelect, { "aria-label": "Label", "data-testid": "dubois-select", mode: "tags", ...extraProps }) }));
}
describe.each([
    ['by test id', () => screen.getByTestId('dubois-select')],
    ['by role', () => screen.getByRole('combobox', { name: 'Label' })],
])('Select, queried by %s', (_, getter) => {
    it('should open and close a select', async () => {
        renderSelect();
        await selectEvent.openMenu(getter());
        expect(screen.getByTestId('dubois-select')).toHaveClass(selectClasses.open);
        await selectEvent.closeMenu(getter());
        expect(screen.getByTestId('dubois-select')).not.toHaveClass(selectClasses.open);
    });
    it('should select an option', async () => {
        const onChange = jest.fn();
        renderSelect({ onChange });
        await selectEvent.singleSelect(getter(), 'Bar');
        expect(onChange).toHaveBeenCalledWith('bar', expect.anything());
    });
    it('should select an option by regex', async () => {
        const onChange = jest.fn();
        renderSelect({ onChange });
        await selectEvent.singleSelect(getter(), /lin/i);
        expect(onChange).toHaveBeenCalledWith('line', expect.anything());
    });
    it('should select multiple options', async () => {
        const onChange = jest.fn();
        renderMultiSelect({ onChange });
        await selectEvent.multiSelect(getter(), ['Line']);
        expect(onChange).toHaveBeenCalledWith(['pie', 'bar', 'line'], expect.anything());
    });
    it('should select multiple options by regex', async () => {
        const onChange = jest.fn();
        renderMultiSelect({ onChange });
        await selectEvent.multiSelect(getter(), [/lin/i]);
        expect(onChange).toHaveBeenCalledWith(['pie', 'bar', 'line'], expect.anything());
    });
    it('should remove option from multi select', async () => {
        const onChange = jest.fn();
        renderMultiSelect({ onChange });
        await selectEvent.removeMultiSelectOption(getter(), 'Pie');
        // Expect onChange to be called with the only item still selected
        expect(onChange).toHaveBeenCalledWith(['bar'], expect.anything());
    });
    it('should clear all options', async () => {
        const onChange = jest.fn();
        renderMultiSelect({ onChange });
        await selectEvent.clearAll(getter());
        expect(onChange).toHaveBeenCalledWith([], expect.anything());
    });
    it('should get all options for single select', async () => {
        renderSelect();
        expect(await selectEvent.getAllOptions(getter())).toEqual(['Pie', 'Bar', 'Line', 'Bubble']);
    });
    it('should get all options for multi select', async () => {
        renderMultiSelect();
        expect(await selectEvent.getAllOptions(getter())).toEqual(['Pie', 'Bar', 'Line', 'Bubble']);
    });
    it('should create new options for tag select', async () => {
        const onChange = jest.fn();
        renderTagSelect({ onChange });
        await selectEvent.createNewOption(getter(), 'New option 1');
        expect(onChange).toHaveBeenCalledWith(['New option 1'], expect.anything());
        await selectEvent.createNewOption(getter(), 'New option 2');
        expect(onChange).toHaveBeenCalledWith(['New option 1', 'New option 2'], expect.anything());
    });
});
//# sourceMappingURL=selectEvent.test.js.map