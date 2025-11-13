import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { TypeaheadComboboxToggleButton } from './TypeaheadComboboxToggleButton';
import { useDesignSystemTheme } from '../Hooks';
import { ClearSelectionButton } from '../_shared_/Combobox/ClearSelectionButton';
export const TypeaheadComboboxControls = ({ getDownshiftToggleButtonProps, showClearSelectionButton, showComboboxToggleButton = true, handleClear, disabled, }) => {
    const { theme } = useDesignSystemTheme();
    return (_jsxs("div", { css: {
            position: 'absolute',
            top: theme.spacing.sm,
            right: 7,
            height: 16,
            zIndex: 1,
        }, children: [showClearSelectionButton && (_jsx(ClearSelectionButton, { onClick: handleClear, css: {
                    pointerEvents: 'all',
                    verticalAlign: 'text-top',
                } })), showComboboxToggleButton && (_jsx(TypeaheadComboboxToggleButton, { ...getDownshiftToggleButtonProps(), disabled: disabled }))] }));
};
//# sourceMappingURL=TypeaheadComboboxControls.js.map