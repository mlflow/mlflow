import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import React, { Children, forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxOptionListContextProvider } from './providers/DialogComboboxOptionListContext';
import { highlightFirstNonDisabledOption } from './shared';
import { EmptyResults, LoadingSpinner } from '../_shared_/Combobox';
export const DialogComboboxOptionList = forwardRef(({ children, loading, loadingDescription = 'DialogComboboxOptionList', withProgressiveLoading, ...restProps }, forwardedRef) => {
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    const ref = useRef(null);
    useImperativeHandle(forwardedRef, () => ref.current);
    const [lookAhead, setLookAhead] = useState('');
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxOptionList` must be used within `DialogCombobox`');
    }
    const lookAheadTimeout = useRef(null);
    useEffect(() => {
        if (lookAheadTimeout.current) {
            clearTimeout(lookAheadTimeout.current);
        }
        lookAheadTimeout.current = setTimeout(() => {
            setLookAhead('');
        }, 1500);
        return () => {
            if (lookAheadTimeout.current) {
                clearTimeout(lookAheadTimeout.current);
            }
        };
    }, [lookAhead]);
    useEffect(() => {
        if (loading && !withProgressiveLoading) {
            return;
        }
        const optionItems = ref.current?.querySelectorAll('[role="option"]');
        const hasTabIndexedOption = Array.from(optionItems ?? []).some((optionItem) => {
            return optionItem.getAttribute('tabindex') === '0';
        });
        if (!hasTabIndexedOption) {
            const firstOptionItem = optionItems?.[0];
            if (firstOptionItem) {
                highlightFirstNonDisabledOption(firstOptionItem, 'start');
            }
        }
    }, [loading, withProgressiveLoading]);
    const handleOnMouseEnter = (event) => {
        const target = event.target;
        if (target) {
            const options = target.hasAttribute('data-combobox-option-list')
                ? target.querySelectorAll('[role="option"]')
                : target?.closest('[data-combobox-option-list="true"]')?.querySelectorAll('[role="option"]');
            if (options) {
                options.forEach((option) => option.removeAttribute('data-highlighted'));
            }
        }
    };
    return (_jsx("div", { ref: ref, "aria-busy": loading, "data-combobox-option-list": "true", css: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            width: '100%',
        }, onMouseEnter: handleOnMouseEnter, ...restProps, children: _jsx(DialogComboboxOptionListContextProvider, { value: { isInsideDialogComboboxOptionList: true, lookAhead, setLookAhead }, children: loading ? (withProgressiveLoading ? (_jsxs(_Fragment, { children: [children, _jsx(LoadingSpinner, { "aria-label": "Loading", alt: "Loading spinner", loadingDescription: loadingDescription })] })) : (_jsx(LoadingSpinner, { "aria-label": "Loading", alt: "Loading spinner", loadingDescription: loadingDescription }))) : children && Children.toArray(children).some((child) => React.isValidElement(child)) ? (children) : (_jsx(EmptyResults, {})) }) }));
});
//# sourceMappingURL=DialogComboboxOptionList.js.map