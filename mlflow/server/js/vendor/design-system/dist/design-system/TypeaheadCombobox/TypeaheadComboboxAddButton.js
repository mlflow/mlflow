import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { forwardRef } from 'react';
import { useTypeaheadComboboxContext } from './hooks/useTypeaheadComboboxContext';
import { Button } from '../Button';
import { useDesignSystemTheme } from '../Hooks';
import { PlusIcon } from '../Icon';
import { getComboboxOptionItemWrapperStyles } from '../_shared_/Combobox';
import { importantify } from '../utils/css-utils';
export const TypeaheadComboboxAddButton = forwardRef(({ children, ...restProps }, forwardedRef) => {
    const { theme } = useDesignSystemTheme();
    const { isInsideTypeaheadCombobox, componentId } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxAddButton` must be used within `TypeaheadCombobox`');
    }
    return (_jsx(Button, { ...restProps, componentId: `${componentId}.add_option`, type: "tertiary", onClick: (event) => {
            event.stopPropagation();
            restProps.onClick?.(event);
        }, onMouseUp: (event) => {
            event.stopPropagation();
            restProps.onMouseUp?.(event);
        }, className: "combobox-footer-add-button", css: {
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
        }, icon: _jsx(PlusIcon, {}), ref: forwardedRef, children: children }));
});
//# sourceMappingURL=TypeaheadComboboxAddButton.js.map