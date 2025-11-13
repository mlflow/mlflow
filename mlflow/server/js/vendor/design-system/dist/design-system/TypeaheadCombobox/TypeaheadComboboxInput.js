import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { forwardRef, useEffect, useRef } from 'react';
import { TypeaheadComboboxControls } from './TypeaheadComboboxControls';
import { useTypeaheadComboboxContext } from './hooks';
import { useDesignSystemTheme } from '../Hooks';
import { Input } from '../Input';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getContainerStyles = () => {
    return css({
        display: 'flex',
        position: 'relative',
    });
};
const getInputStyles = (theme, showComboboxToggleButton, useNewBorderRadii, useNewBorderColors) => css({
    paddingRight: showComboboxToggleButton ? 52 : 26,
    width: '100%',
    minWidth: 72,
    ...(useNewBorderRadii && {
        borderRadius: theme.borders.borderRadiusSm,
    }),
    ...(useNewBorderColors && {
        borderColor: theme.colors.actionDefaultBorderDefault,
    }),
    '&:disabled': {
        borderColor: theme.colors.actionDisabledBorder,
        backgroundColor: theme.colors.actionDisabledBackground,
        color: theme.colors.actionDisabledText,
    },
    '&:not(:disabled)': {
        backgroundColor: 'transparent',
    },
});
export const TypeaheadComboboxInput = forwardRef(({ comboboxState, allowClear = true, showComboboxToggleButton = true, formOnChange, onClick, clearInputValueOnFocus = false, ...restProps }, ref) => {
    const { isInsideTypeaheadCombobox, floatingUiRefs, setInputWidth, inputWidth } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxInput` must be used within `TypeaheadCombobox`');
    }
    const { getInputProps, getToggleButtonProps, toggleMenu, inputValue, setInputValue, reset, isOpen, selectedItem, componentId, } = comboboxState;
    const { ref: downshiftRef, ...downshiftProps } = getInputProps({}, { suppressRefError: true });
    const mergedRef = useMergeRefs([ref, downshiftRef]);
    const { theme } = useDesignSystemTheme();
    const { useNewBorderRadii, useNewBorderColors } = useDesignSystemSafexFlags();
    const handleClick = (e) => {
        onClick?.(e);
        toggleMenu();
    };
    const previousInputValue = useRef(null);
    useEffect(() => {
        if (!clearInputValueOnFocus) {
            return;
        }
        // If the input is open and has value, clear the input value
        if (isOpen && !previousInputValue.current) {
            previousInputValue.current = {
                selectedItem: selectedItem,
                inputValue: inputValue,
            };
            setInputValue('');
        }
        // If the input is closed and the input value was cleared, restore the input value
        if (!isOpen && previousInputValue.current) {
            // Only restore the input value if the selected item is the same as the previous selected item
            if (previousInputValue.current.selectedItem === selectedItem) {
                setInputValue(previousInputValue.current.inputValue);
            }
            previousInputValue.current = null;
        }
    }, [isOpen, inputValue, setInputValue, previousInputValue, clearInputValueOnFocus, selectedItem]);
    const handleClear = () => {
        setInputValue('');
        reset();
        formOnChange?.(null);
    };
    // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
    useEffect(() => {
        // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
        if (floatingUiRefs?.domReference) {
            const width = floatingUiRefs.domReference.current?.getBoundingClientRect().width ?? 0;
            // Only update context width when the input width updated
            if (width !== inputWidth) {
                setInputWidth?.(width);
            }
        }
    }, [floatingUiRefs?.domReference, setInputWidth, inputWidth]);
    return (_jsxs("div", { ref: floatingUiRefs?.setReference, css: getContainerStyles(), className: restProps.className, ...addDebugOutlineIfEnabled(), children: [_jsx(Input, { componentId: componentId ? `${componentId}.input` : 'design_system.typeahead_combobox.input', ref: mergedRef, ...downshiftProps, "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined, onClick: handleClick, css: getInputStyles(theme, showComboboxToggleButton, useNewBorderRadii, useNewBorderColors), ...restProps }), _jsx(TypeaheadComboboxControls, { getDownshiftToggleButtonProps: getToggleButtonProps, showClearSelectionButton: allowClear && Boolean(inputValue) && !restProps.disabled, showComboboxToggleButton: showComboboxToggleButton, handleClear: handleClear, disabled: restProps.disabled })] }));
});
//# sourceMappingURL=TypeaheadComboboxInput.js.map