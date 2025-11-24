import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { DialogComboboxContent, DialogComboboxOptionList } from '../DialogCombobox';
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectContent = forwardRef(({ children, minWidth = 150, ...restProps }, ref) => {
    return (_jsx(DialogComboboxContent, { minWidth: minWidth, ...restProps, ref: ref, children: _jsx(DialogComboboxOptionList, { children: children }) }));
});
//# sourceMappingURL=SelectContent.js.map