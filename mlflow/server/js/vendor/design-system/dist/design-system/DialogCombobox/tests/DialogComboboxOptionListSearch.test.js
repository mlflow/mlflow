import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { render, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSearch } from '../DialogComboboxOptionListSearch';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';
const items = [
    {
        key: '1',
        value: 'Alpha',
    },
    {
        key: '2',
        value: 'Beta',
    },
    {
        key: '3',
        value: 'Charlie',
    },
    {
        key: '4',
        value: 'Delta',
    },
];
describe('Dialog Combobox - Search', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('correctly filters DialogComboboxOptionListSelectItem children', () => {
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistsearch.test.tsx_38", label: "Owner", open: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { minWidth: 250, children: _jsx(DialogComboboxOptionList, { children: _jsx(DialogComboboxOptionListSearch, { children: items.map((item) => (_jsx(DialogComboboxOptionListSelectItem, { value: item.value }, item.key))) }) }) })] }));
        const input = getByRole('searchbox');
        userEvent.type(input, 'A');
        // eslint-disable-next-line testing-library/await-async-utils -- FEINF-3005
        waitFor(() => {
            expect(queryAllByRole('option')).toHaveLength(1);
        });
    });
    it('correctly filters DialogComboboxOptionListCheckboxItem children', () => {
        const { getByRole, queryAllByRole } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistsearch.test.tsx_62", label: "Owner", open: true, multiSelect: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { minWidth: 250, children: _jsx(DialogComboboxOptionList, { children: _jsx(DialogComboboxOptionListSearch, { children: items.map((item) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: item.value }, item.key))) }) }) })] }));
        const input = getByRole('searchbox');
        userEvent.type(input, 'A');
        // eslint-disable-next-line testing-library/await-async-utils -- FEINF-3005
        waitFor(() => {
            expect(queryAllByRole('option')).toHaveLength(1);
        });
    });
    it('renders the search controls if passed', () => {
        const { getByText } = render(_jsxs(DialogCombobox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistsearch.test.tsx_97", label: "Owner", open: true, multiSelect: true, children: [_jsx(DialogComboboxTrigger, {}), _jsx(DialogComboboxContent, { minWidth: 250, children: _jsx(DialogComboboxOptionList, { children: _jsx(DialogComboboxOptionListSearch, { rightSearchControls: _jsx("span", { children: "Right Search Control" }), children: items.map((item) => (_jsx(DialogComboboxOptionListCheckboxItem, { value: item.value }, item.key))) }) }) })] }));
        expect(getByText('Right Search Control')).toBeInTheDocument();
    });
});
//# sourceMappingURL=DialogComboboxOptionListSearch.test.js.map