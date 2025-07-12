import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, jest } from '@jest/globals';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
// eslint-disable-next-line @databricks/no-restricted-imports-regexp
import { mount, configure as configureEnzyme } from 'enzyme';
import { expect } from '@databricks/config-jest/enzyme';
import { selectEvent } from '.';
import { LegacySelect, DesignSystemProvider } from '../../design-system';
import { selectClasses } from '../common';
configureEnzyme({ adapter: new Adapter() });
function renderSelect(extraProps) {
    return mount(_jsx(DesignSystemProvider, { children: _jsxs(LegacySelect, { "data-testid": "dubois-select", value: "pie", ...extraProps, children: [_jsx(LegacySelect.Option, { value: "pie", children: "Pie" }), _jsx(LegacySelect.Option, { value: "bar", children: "Bar" }), _jsx(LegacySelect.Option, { value: "line", children: "Line" }), _jsx(LegacySelect.Option, { value: "bubble", children: "Bubble" })] }) }));
}
function renderMultiSelect(extraProps) {
    return mount(_jsx(DesignSystemProvider, { children: _jsxs(LegacySelect, { allowClear: true, "data-testid": "dubois-select", mode: "multiple", value: ['pie', 'bar'], ...extraProps, children: [_jsx(LegacySelect.Option, { value: "pie", children: "Pie" }), _jsx(LegacySelect.Option, { value: "bar", children: "Bar" }), _jsx(LegacySelect.Option, { value: "line", children: "Line" }), _jsx(LegacySelect.Option, { value: "bubble", children: "Bubble" })] }) }));
}
function renderTagSelect(extraProps) {
    return mount(_jsx(DesignSystemProvider, { children: _jsx(LegacySelect, { "aria-label": "Label", "data-testid": "dubois-select", mode: "tags", ...extraProps }) }));
}
describe('selectEvent', () => {
    it('should open and close a select', async () => {
        const wrapper = renderSelect();
        await selectEvent.openMenu(() => wrapper.find({ 'data-testid': 'dubois-select' }));
        expect(wrapper.find({ 'data-testid': 'dubois-select' }).hostNodes()).toHaveClassName(selectClasses.open);
        await selectEvent.closeMenu(() => wrapper.find({ 'data-testid': 'dubois-select' }));
        expect(wrapper.find({ 'data-testid': 'dubois-select' }).hostNodes()).not.toHaveClassName(selectClasses.open);
    });
    it('should select an option', async () => {
        const onChange = jest.fn();
        const wrapper = renderSelect({ onChange });
        await selectEvent.singleSelect(() => wrapper.find({ 'data-testid': 'dubois-select' }), 'Bar');
        expect(onChange).toHaveBeenCalledWith('bar', expect.anything());
    });
    it('should select multiple options', async () => {
        const onChange = jest.fn();
        const wrapper = renderMultiSelect({ onChange });
        await selectEvent.multiSelect(() => wrapper.find({ 'data-testid': 'dubois-select' }), ['Line']);
        expect(onChange).toHaveBeenCalledWith(['pie', 'bar', 'line'], expect.anything());
    });
    it('should remove option from multi select', async () => {
        const onChange = jest.fn();
        const wrapper = renderMultiSelect({ onChange });
        selectEvent.removeMultiSelectOption(() => wrapper.find({ 'data-testid': 'dubois-select' }), 'Pie');
        // Expect onChange to be called with the only item still selected
        expect(onChange).toHaveBeenCalledWith(['bar'], expect.anything());
    });
    it('should clear all options', async () => {
        const onChange = jest.fn();
        const wrapper = renderMultiSelect({ onChange });
        selectEvent.clearAll(() => wrapper.find({ 'data-testid': 'dubois-select' }));
        expect(onChange).toHaveBeenCalledWith([], expect.anything());
    });
    it('should get all options for single select', async () => {
        const wrapper = renderSelect();
        expect(await selectEvent.getAllOptions(() => wrapper.find({ 'data-testid': 'dubois-select' }))).toEqual([
            'Pie',
            'Bar',
            'Line',
            'Bubble',
        ]);
    });
    it('should get all options for multi select', async () => {
        const wrapper = renderMultiSelect();
        expect(await selectEvent.getAllOptions(() => wrapper.find({ 'data-testid': 'dubois-select' }))).toEqual([
            'Pie',
            'Bar',
            'Line',
            'Bubble',
        ]);
    });
    it('should create new options for tag select', async () => {
        const onChange = jest.fn();
        const wrapper = renderTagSelect({ onChange });
        await selectEvent.createNewOption(() => wrapper.find({ 'data-testid': 'dubois-select' }), 'New option 1');
        expect(onChange).toHaveBeenCalledWith(['New option 1'], expect.anything());
        await selectEvent.createNewOption(() => wrapper.find({ 'data-testid': 'dubois-select' }), 'New option 2');
        expect(onChange).toHaveBeenCalledWith(['New option 1', 'New option 2'], expect.anything());
    });
});
//# sourceMappingURL=selectEvent.test.js.map