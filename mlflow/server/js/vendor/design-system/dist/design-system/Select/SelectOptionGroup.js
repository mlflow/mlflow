import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { DialogComboboxSectionHeader } from '../DialogCombobox';
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const SelectOptionGroup = (props) => {
    const { name, children, ...restProps } = props;
    return (_jsxs(_Fragment, { children: [_jsx(DialogComboboxSectionHeader, { ...restProps, children: name }), children] }));
};
//# sourceMappingURL=SelectOptionGroup.js.map