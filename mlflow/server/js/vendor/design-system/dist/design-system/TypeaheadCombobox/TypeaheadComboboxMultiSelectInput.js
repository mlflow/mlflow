import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { forwardRef, useState, useRef, useLayoutEffect, useEffect } from 'react';
import { CountBadge } from './CountBadge';
import { TypeaheadComboboxControls } from './TypeaheadComboboxControls';
import { TypeaheadComboboxSelectedItem } from './TypeaheadComboboxSelectedItem';
import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { LegacyTooltip } from '../LegacyTooltip';
import { getValidationStateColor, useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getContainerStyles = (theme, validationState, width, maxHeight, disabled, useNewBorderColors) => {
    const validationColor = getValidationStateColor(theme, validationState);
    return css({
        cursor: 'text',
        display: 'inline-block',
        verticalAlign: 'top',
        border: `1px solid ${useNewBorderColors ? theme.colors.actionDefaultBorderDefault : theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        minHeight: 32,
        height: 'auto',
        minWidth: 0,
        ...(width ? { width } : {}),
        ...(maxHeight ? { maxHeight } : {}),
        padding: '5px 52px 5px 12px',
        position: 'relative',
        overflow: 'auto',
        textOverflow: 'ellipsis',
        '&:hover': {
            border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
        },
        '&:focus-within': {
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineWidth: 2,
            outlineOffset: -2,
            outlineStyle: 'solid',
            boxShadow: 'none',
            borderColor: 'transparent',
        },
        '&&': {
            ...(validationState && { borderColor: validationColor }),
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover,
            },
            '&:focus': {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                boxShadow: 'none',
                borderColor: 'transparent',
            },
            ...(disabled && {
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: theme.colors.actionDisabledBackground,
                cursor: 'not-allowed',
                outline: 'none',
                '&:hover': {
                    border: `1px solid ${theme.colors.actionDisabledBorder}`,
                },
                '&:focus-within': {
                    outline: 'none',
                    borderColor: theme.colors.actionDisabledBorder,
                },
            }),
        },
    });
};
const getContentWrapperStyles = () => {
    return css({
        display: 'flex',
        flex: 'auto',
        flexWrap: 'wrap',
        maxWidth: '100%',
        position: 'relative',
    });
};
const getInputWrapperStyles = () => {
    return css({
        display: 'inline-flex',
        position: 'relative',
        maxWidth: '100%',
        alignSelf: 'auto',
        flex: 'none',
    });
};
const getInputStyles = (theme) => {
    return css({
        lineHeight: 20,
        height: 24,
        margin: 0,
        padding: 0,
        appearance: 'none',
        cursor: 'auto',
        width: '100%',
        backgroundColor: 'transparent',
        color: theme.colors.textPrimary,
        '&, &:hover, &:focus-visible': {
            border: 'none',
            outline: 'none',
        },
        '&::placeholder': {
            color: theme.colors.textPlaceholder,
        },
    });
};
export const TypeaheadComboboxMultiSelectInput = forwardRef(({ comboboxState, multipleSelectionState, selectedItems, setSelectedItems, getSelectedItemLabel, allowClear = true, showTagAfterValueCount = 20, width, maxHeight, placeholder, validationState, showComboboxToggleButton, disableTooltip = false, ...restProps }, ref) => {
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxMultiSelectInput` must be used within `TypeaheadCombobox`');
    }
    const { getInputProps, getToggleButtonProps, toggleMenu, inputValue, setInputValue } = comboboxState;
    const { getSelectedItemProps, getDropdownProps, reset, removeSelectedItem } = multipleSelectionState;
    const { ref: downshiftRef, ...downshiftProps } = getInputProps(getDropdownProps({}, { suppressRefError: true }));
    const { floatingUiRefs, setInputWidth: setContextInputWidth, inputWidth: contextInputWidth, } = useTypeaheadComboboxContext();
    const containerRef = useRef(null);
    const mergedContainerRef = useMergeRefs([containerRef, floatingUiRefs?.setReference]);
    const itemsRef = useRef(null);
    const measureRef = useRef(null);
    const innerRef = useRef(null);
    const mergedInputRef = useMergeRefs([ref, innerRef, downshiftRef]);
    const { theme } = useDesignSystemTheme();
    const { useNewBorderColors } = useDesignSystemSafexFlags();
    const [inputWidth, setInputWidth] = useState(0);
    const shouldShowCountBadge = selectedItems.length > showTagAfterValueCount;
    const [showTooltip, setShowTooltip] = useState(shouldShowCountBadge);
    const selectedItemsToRender = selectedItems.slice(0, showTagAfterValueCount);
    const handleClick = () => {
        if (!restProps.disabled) {
            innerRef.current?.focus();
            toggleMenu();
        }
    };
    const handleClear = () => {
        setInputValue('');
        reset();
        setSelectedItems([]);
    };
    // We measure width and set to the input immediately
    useLayoutEffect(() => {
        if (measureRef?.current) {
            const measuredWidth = measureRef.current.scrollWidth;
            setInputWidth(measuredWidth);
        }
    }, [measureRef?.current?.scrollWidth, selectedItems?.length]);
    // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
    useEffect(() => {
        // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
        if (floatingUiRefs?.domReference) {
            const width = floatingUiRefs.domReference.current?.getBoundingClientRect().width ?? 0;
            // Only update context width when the input width updated
            if (width !== contextInputWidth) {
                setContextInputWidth?.(width);
            }
        }
    }, [floatingUiRefs?.domReference, setContextInputWidth, contextInputWidth]);
    // Determine whether to show tooltip
    useEffect(() => {
        let isPartiallyHidden = false;
        if (itemsRef.current && containerRef.current) {
            const { clientHeight: innerHeight } = itemsRef.current;
            const { clientHeight: outerHeight } = containerRef.current;
            isPartiallyHidden = innerHeight > outerHeight;
        }
        setShowTooltip(!disableTooltip && (shouldShowCountBadge || isPartiallyHidden));
    }, [shouldShowCountBadge, itemsRef.current?.clientHeight, containerRef.current?.clientHeight, disableTooltip]);
    const content = (_jsxs("div", { ...addDebugOutlineIfEnabled(), onClick: handleClick, ref: mergedContainerRef, css: getContainerStyles(theme, validationState, width, maxHeight, restProps.disabled, useNewBorderColors), tabIndex: restProps.disabled ? -1 : 0, children: [_jsxs("div", { ref: itemsRef, css: getContentWrapperStyles(), children: [selectedItemsToRender?.map((selectedItemForRender, index) => (_jsx(TypeaheadComboboxSelectedItem, { label: getSelectedItemLabel(selectedItemForRender), item: selectedItemForRender, getSelectedItemProps: getSelectedItemProps, removeSelectedItem: removeSelectedItem, disabled: restProps.disabled }, `selected-item-${index}`))), shouldShowCountBadge && (_jsx(CountBadge, { countStartAt: showTagAfterValueCount, totalCount: selectedItems.length, role: "status", "aria-label": "Selected options count", disabled: restProps.disabled })), _jsxs("div", { css: getInputWrapperStyles(), children: [_jsx("input", { ...downshiftProps, ref: mergedInputRef, css: [getInputStyles(theme), { width: inputWidth }], placeholder: selectedItems?.length ? undefined : placeholder, "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined, ...restProps }), _jsxs("span", { ref: measureRef, "aria-hidden": true, css: { visibility: 'hidden', whiteSpace: 'pre', position: 'absolute' }, children: [innerRef.current?.value ? innerRef.current.value : placeholder, "\u00A0"] })] })] }), _jsx(TypeaheadComboboxControls, { getDownshiftToggleButtonProps: getToggleButtonProps, showComboboxToggleButton: showComboboxToggleButton, showClearSelectionButton: allowClear && (Boolean(inputValue) || (selectedItems && selectedItems.length > 0)) && !restProps.disabled, handleClear: handleClear, disabled: restProps.disabled })] }));
    if (showTooltip && selectedItems.length > 0) {
        return (_jsx(LegacyTooltip, { title: selectedItems.map((item) => getSelectedItemLabel(item)).join(', '), children: content }));
    }
    return content;
});
//# sourceMappingURL=TypeaheadComboboxMultiSelectInput.js.map