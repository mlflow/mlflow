import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { SelectContextProvider } from './providers/SelectContext';
import { DialogCombobox } from '../DialogCombobox';
/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
export const Select = (props) => {
    const { children, placeholder, value, label, ...restProps } = props;
    return (_jsx(SelectContextProvider, { value: { isSelect: true, placeholder }, children: _jsx(DialogCombobox, { label: label, value: value ? [value] : [], ...restProps, children: children }) }));
};
//# sourceMappingURL=Select.js.map