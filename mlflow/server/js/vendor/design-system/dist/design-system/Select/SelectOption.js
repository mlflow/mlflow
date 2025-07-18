import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { DialogComboboxOptionListSelectItem } from '../DialogCombobox';
import { useDialogComboboxContext } from '../DialogCombobox/hooks/useDialogComboboxContext';
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectOption = forwardRef((props, ref) => {
    const { value } = useDialogComboboxContext();
    return _jsx(DialogComboboxOptionListSelectItem, { checked: value && value[0] === props.value, ...props, ref: ref });
});
//# sourceMappingURL=SelectOption.js.map