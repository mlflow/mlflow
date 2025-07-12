import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { render } from '@testing-library/react';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
describe('Dialog Combobox - Content', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('renders', () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByLabelText, queryByLabelText } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxcontent.test.tsx_17", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { children: _jsx(DialogComboboxOptionList, { children: options.map((option, key) => (_jsx(DialogComboboxOptionListSelectItem, { value: option }, key))) }) })] }));
        const content = getByLabelText('example filter options');
        expect(content).toBeVisible();
        expect(content.getAttribute('aria-busy')).toBeFalsy();
        expect(queryByLabelText('Loading')).toBeFalsy();
    });
    it('renders loading', () => {
        const options = ['value 1', 'value 2', 'value 3'];
        const { getByLabelText, queryByText } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxcontent.test.tsx_38", label: "example filter", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { loading: true, children: _jsx(DialogComboboxOptionList, { children: options.map((option, key) => (_jsx(DialogComboboxOptionListSelectItem, { value: option }, key))) }) })] }));
        const content = getByLabelText('example filter options');
        expect(content).toBeVisible();
        expect(content.getAttribute('aria-busy')).toBeTruthy();
        expect(queryByText('Loading')).toBeTruthy();
    });
    it("doesn't render outside DialogCombobox", () => {
        jest.spyOn(console, 'error').mockImplementation(() => { });
        expect(() => render(_jsx(DialogComboboxContent, { "aria-label": "Buttons container" }))).toThrowError('`DialogComboboxContent` must be used within `DialogCombobox`');
    });
});
//# sourceMappingURL=DialogComboboxContent.test.js.map