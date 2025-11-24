import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { useDialogComboboxOptionListContext } from './hooks/useDialogComboboxOptionListContext';
import { dialogComboboxLookAheadKeyDown, getKeyboardNavigationFunctions } from './shared';
import { useDesignSystemTheme } from '../Hooks';
import { CheckIcon, InfoIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { useSelectContext } from '../Select/hooks/useSelectContext';
import { getComboboxOptionItemWrapperStyles, getInfoIconStyles, getSelectItemWithHintColumnStyles, getHintColumnStyles, getComboboxOptionLabelStyles, } from '../_shared_/Combobox';
const DuboisDialogComboboxOptionListSelectItem = forwardRef(({ value, checked, disabledReason, onChange, hintColumn, hintColumnWidthPercent = 50, children, _TYPE, icon, dangerouslyHideCheck, ...props }, ref) => {
    const { theme } = useDesignSystemTheme();
    const { stayOpenOnSelection, isOpen, setIsOpen, value: existingValue, contentWidth, textOverflowMode, scrollToSelectedElement, disableMouseOver, setDisableMouseOver, } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList, lookAhead, setLookAhead } = useDialogComboboxOptionListContext();
    const { isSelect } = useSelectContext();
    if (!isInsideDialogComboboxOptionList) {
        throw new Error('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
    }
    const itemRef = useRef(null);
    const prevCheckedRef = useRef(checked);
    useImperativeHandle(ref, () => itemRef.current);
    useEffect(() => {
        if (scrollToSelectedElement && isOpen) {
            // Check if checked didn't change since the last update, otherwise the popover is still open and we don't need to scroll
            if (checked && prevCheckedRef.current === checked) {
                // Wait for the popover to render and scroll to the selected element's position
                const interval = setInterval(() => {
                    if (itemRef.current) {
                        itemRef.current?.scrollIntoView?.({
                            behavior: 'smooth',
                            block: 'center',
                        });
                        clearInterval(interval);
                    }
                }, 50);
                return () => clearInterval(interval);
            }
            prevCheckedRef.current = checked;
        }
        return;
    }, [isOpen, scrollToSelectedElement, checked]);
    const handleSelect = (e) => {
        if (onChange) {
            if (isSelect) {
                onChange({ value, label: typeof children === 'string' ? children : value }, e);
                if (existingValue?.includes(value)) {
                    setIsOpen(false);
                }
                return;
            }
            onChange(value, e);
            // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
            if (!stayOpenOnSelection && existingValue?.includes(value)) {
                setIsOpen(false);
            }
        }
    };
    let content = children ?? value;
    if (props.disabled && disabledReason) {
        content = (_jsxs("div", { css: { display: 'flex' }, children: [_jsx("div", { children: content }), _jsx(LegacyTooltip, { title: disabledReason, placement: "right", children: _jsx("span", { css: getInfoIconStyles(theme), children: _jsx(InfoIcon, { "aria-label": "Disabled status information", "aria-hidden": "false" }) }) })] }));
    }
    return (_jsxs("div", { ref: itemRef, css: [
            getComboboxOptionItemWrapperStyles(theme),
            {
                '&:focus': {
                    background: theme.colors.actionTertiaryBackgroundHover,
                    outline: 'none',
                },
            },
        ], ...props, onClick: (e) => {
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
        }), role: "option", "aria-selected": checked, children: [!dangerouslyHideCheck &&
                (checked ? (_jsx(CheckIcon, { css: { paddingTop: 2, color: theme.colors.textSecondary } })) : (_jsx("div", { style: { width: 16, flexShrink: 0 } }))), _jsxs("label", { css: getComboboxOptionLabelStyles({
                    theme,
                    dangerouslyHideCheck,
                    textOverflowMode,
                    contentWidth,
                    hasHintColumn: Boolean(hintColumn),
                }), children: [icon && (_jsx("span", { style: {
                            position: 'relative',
                            top: 1,
                            marginRight: theme.spacing.sm,
                            color: theme.colors.textSecondary,
                        }, children: icon })), hintColumn ? (_jsxs("span", { css: getSelectItemWithHintColumnStyles(hintColumnWidthPercent), children: [content, _jsx("span", { css: getHintColumnStyles(theme, Boolean(props.disabled), textOverflowMode), children: hintColumn })] })) : (content)] })] }));
});
DuboisDialogComboboxOptionListSelectItem.defaultProps = {
    _TYPE: 'DialogComboboxOptionListSelectItem',
};
export const DialogComboboxOptionListSelectItem = DuboisDialogComboboxOptionListSelectItem;
export { getComboboxOptionItemWrapperStyles, getComboboxOptionLabelStyles } from '../_shared_/Combobox';
//# sourceMappingURL=DialogComboboxOptionListSelectItem.js.map