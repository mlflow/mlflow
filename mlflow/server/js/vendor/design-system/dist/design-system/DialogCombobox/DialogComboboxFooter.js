import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDesignSystemTheme } from '../Hooks';
import { getFooterStyles } from '../_shared_/Combobox';
export const DialogComboboxFooter = ({ children, ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxFooter` must be used within `DialogCombobox`');
    }
    return (_jsx("div", { ...restProps, css: getFooterStyles(theme), children: children }));
};
//# sourceMappingURL=DialogComboboxFooter.js.map