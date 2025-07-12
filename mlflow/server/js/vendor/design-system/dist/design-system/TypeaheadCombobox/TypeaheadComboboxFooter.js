import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { getFooterStyles } from '../_shared_/Combobox';
export const DuboisTypeaheadComboboxFooter = ({ children, ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxFooter` must be used within `TypeaheadComboboxMenu`');
    }
    return (_jsx("div", { ...restProps, css: getFooterStyles(theme), children: children }));
};
DuboisTypeaheadComboboxFooter.defaultProps = {
    _type: 'TypeaheadComboboxFooter',
};
export const TypeaheadComboboxFooter = DuboisTypeaheadComboboxFooter;
//# sourceMappingURL=TypeaheadComboboxFooter.js.map