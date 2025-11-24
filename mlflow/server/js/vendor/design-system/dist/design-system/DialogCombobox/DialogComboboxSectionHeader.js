import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { SectionHeader } from '../_shared_/Combobox';
export const DialogComboboxSectionHeader = ({ children, ...props }) => {
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxSectionHeader` must be used within `DialogCombobox`');
    }
    return _jsx(SectionHeader, { ...props, children: children });
};
//# sourceMappingURL=DialogComboboxSectionHeader.js.map