import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDialogComboboxOptionListContext } from './hooks/useDialogComboboxOptionListContext';
import { getKeyboardNavigationFunctions, dialogComboboxLookAheadKeyDown, getDialogComboboxOptionLabelWidth, } from './shared';
import { Checkbox } from '../Checkbox';
import { useDesignSystemTheme } from '../Hooks';
import { InfoIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { getComboboxOptionItemWrapperStyles, getInfoIconStyles, getCheckboxStyles } from '../_shared_/Combobox';
const DuboisDialogComboboxOptionListCheckboxItem = forwardRef(({ value, checked, indeterminate, onChange, children, disabledReason, _TYPE, ...props }, ref) => {
    const { theme } = useDesignSystemTheme();
    const { textOverflowMode, contentWidth, disableMouseOver, setDisableMouseOver } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList, setLookAhead, lookAhead } = useDialogComboboxOptionListContext();
    if (!isInsideDialogComboboxOptionList) {
        throw new Error('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
    }
    const handleSelect = (e) => {
        if (onChange) {
            onChange(value, e);
        }
    };
    let content = children ?? value;
    if (props.disabled && disabledReason) {
        content = (_jsxs("div", { css: { display: 'flex' }, children: [_jsx("div", { children: content }), _jsx("div", { children: _jsx(LegacyTooltip, { title: disabledReason, placement: "right", children: _jsx("span", { css: getInfoIconStyles(theme), children: _jsx(InfoIcon, { "aria-label": "Disabled status information", "aria-hidden": "false" }) }) }) })] }));
    }
    return (_jsx("div", { ref: ref, role: "option", "aria-selected": indeterminate ? false : checked, css: [getComboboxOptionItemWrapperStyles(theme)], ...props, onClick: (e) => {
            if (props.disabled) {
                e.preventDefault();
            }
            else {
                handleSelect(e);
            }
        }, tabIndex: -1, ...getKeyboardNavigationFunctions(handleSelect, {
            onKeyDown: props.onKeyDown,
            onMouseEnter: props.onMouseEnter,
            onDefaultKeyDown: (e) => dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead),
            disableMouseOver,
            setDisableMouseOver,
        }), children: _jsx(Checkbox, { componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxoptionlistcheckboxitem.tsx_86", disabled: props.disabled, isChecked: indeterminate ? null : checked, css: [
                getCheckboxStyles(theme, textOverflowMode),
                contentWidth
                    ? {
                        '& > span:last-of-type': {
                            width: getDialogComboboxOptionLabelWidth(theme, contentWidth),
                        },
                    }
                    : {},
            ], tabIndex: -1, 
            // Needed because Antd handles keyboard inputs as clicks
            onClick: (e) => {
                e.stopPropagation();
                handleSelect(e);
            }, children: _jsx("div", { css: { maxWidth: '100%' }, children: content }) }) }));
});
DuboisDialogComboboxOptionListCheckboxItem.defaultProps = {
    _TYPE: 'DialogComboboxOptionListCheckboxItem',
};
export const DialogComboboxOptionListCheckboxItem = DuboisDialogComboboxOptionListCheckboxItem;
//# sourceMappingURL=DialogComboboxOptionListCheckboxItem.js.map