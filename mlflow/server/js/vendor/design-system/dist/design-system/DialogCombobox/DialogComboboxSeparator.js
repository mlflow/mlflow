import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { Separator } from '../_shared_/Combobox';
export const DialogComboboxSeparator = (props) => {
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxSeparator` must be used within `DialogCombobox`');
    }
    return _jsx(Separator, { ...props });
};
//# sourceMappingURL=DialogComboboxSeparator.js.map