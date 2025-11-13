import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useTypeaheadComboboxContext } from './hooks';
import { SectionHeader } from '../_shared_/Combobox';
export const TypeaheadComboboxSectionHeader = ({ children, ...props }) => {
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxSectionHeader` must be used within `TypeaheadComboboxMenu`');
    }
    return _jsx(SectionHeader, { ...props, children: children });
};
//# sourceMappingURL=TypeaheadComboboxSectionHeader.js.map