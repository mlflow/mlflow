import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { DialogComboboxTrigger } from '../DialogCombobox';
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectTrigger = forwardRef((props, ref) => {
    const { children, ...restProps } = props;
    return (_jsx(DialogComboboxTrigger, { allowClear: false, ...restProps, ref: ref, children: children }));
});
//# sourceMappingURL=SelectTrigger.js.map