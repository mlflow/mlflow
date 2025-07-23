import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { PlusIcon } from '../Icon';
import { getComboboxOptionItemWrapperStyles } from '../_shared_/Combobox';
import { importantify } from '../utils/css-utils';
export const DialogComboboxAddButton = ({ children, ...restProps }) => {
    const { theme } = useDesignSystemTheme();
    const { isInsideDialogCombobox, componentId } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxAddButton` must be used within `DialogCombobox`');
    }
    return (_jsx(Button, { componentId: `${componentId ? componentId : 'design_system.dialogcombobox'}.add_option`, ...restProps, type: "tertiary", className: "combobox-footer-add-button", css: {
            ...getComboboxOptionItemWrapperStyles(theme),
            ...css(importantify({
                width: '100%',
                padding: 0,
                display: 'flex',
                alignItems: 'center',
                borderRadius: 0,
                '&:focus': {
                    background: theme.colors.actionTertiaryBackgroundHover,
                    outline: 'none',
                },
            })),
        }, icon: _jsx(PlusIcon, {}), children: children }));
};
//# sourceMappingURL=DialogComboboxAddButton.js.map