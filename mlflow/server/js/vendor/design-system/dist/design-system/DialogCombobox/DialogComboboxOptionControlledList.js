import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { DialogComboboxOptionList } from './DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from './DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSearch } from './DialogComboboxOptionListSearch';
import { DialogComboboxOptionListSelectItem } from './DialogComboboxOptionListSelectItem';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxOptionListContextProvider } from './providers/DialogComboboxOptionListContext';
import { highlightOption } from './shared';
import { EmptyResults, LoadingSpinner } from '../_shared_/Combobox';
export const DialogComboboxOptionControlledList = forwardRef(({ options, onChange, loading, loadingDescription = 'DialogComboboxOptionControlledList', withProgressiveLoading, withSearch, showAllOption, allOptionLabel = 'All', ...restProps }, forwardedRef) => {
    const { isInsideDialogCombobox, multiSelect, value, setValue, setIsControlled } = useDialogComboboxContext();
    const [lookAhead, setLookAhead] = useState('');
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxOptionControlledList` must be used within `DialogCombobox`');
    }
    const lookAheadTimeout = useRef(null);
    const ref = useRef(null);
    useImperativeHandle(forwardedRef, () => ref.current);
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
                highlightOption(firstOptionItem, undefined, false);
            }
        }
    }, [loading, withProgressiveLoading]);
    const isOptionChecked = options.reduce((acc, option) => {
        acc[option] = value?.includes(option);
        return acc;
    }, {});
    const handleUpdate = (updatedValue) => {
        setIsControlled(true);
        let newValue = [];
        if (multiSelect) {
            if (value.find((item) => item === updatedValue)) {
                newValue = value.filter((item) => item !== updatedValue);
            }
            else {
                newValue = [...value, updatedValue];
            }
        }
        else {
            newValue = [updatedValue];
        }
        setValue(newValue);
        isOptionChecked[updatedValue] = !isOptionChecked[updatedValue];
        if (onChange) {
            onChange(newValue);
        }
    };
    const handleSelectAll = () => {
        setIsControlled(true);
        if (value.length === options.length) {
            setValue([]);
            options.forEach((option) => {
                isOptionChecked[option] = false;
            });
            if (onChange) {
                onChange([]);
            }
        }
        else {
            setValue(options);
            options.forEach((option) => {
                isOptionChecked[option] = true;
            });
            if (onChange) {
                onChange(options);
            }
        }
    };
    const renderedOptions = (_jsxs(_Fragment, { children: [showAllOption && multiSelect && (_jsx(DialogComboboxOptionListCheckboxItem, { value: "all", onChange: handleSelectAll, checked: value.length === options.length, indeterminate: Boolean(value.length) && value.length !== options.length, children: allOptionLabel })), options && options.length > 0 ? (options.map((option, key) => multiSelect ? (_jsx(DialogComboboxOptionListCheckboxItem, { value: option, checked: isOptionChecked[option], onChange: handleUpdate, children: option }, key)) : (_jsx(DialogComboboxOptionListSelectItem, { value: option, checked: isOptionChecked[option], onChange: handleUpdate, children: option }, key)))) : (_jsx(EmptyResults, {}))] }));
    const optionList = (_jsx(DialogComboboxOptionList, { children: withSearch ? (_jsx(DialogComboboxOptionListSearch, { hasWrapper: true, children: renderedOptions })) : (renderedOptions) }));
    return (_jsx("div", { ref: ref, "aria-busy": loading, css: { display: 'flex', flexDirection: 'column', alignItems: 'flex-start', width: '100%' }, ...restProps, children: _jsx(DialogComboboxOptionListContextProvider, { value: { isInsideDialogComboboxOptionList: true, lookAhead, setLookAhead }, children: _jsx(_Fragment, { children: loading ? (withProgressiveLoading ? (_jsxs(_Fragment, { children: [optionList, _jsx(LoadingSpinner, { "aria-label": "Loading", alt: "Loading spinner", loadingDescription: loadingDescription })] })) : (_jsx(LoadingSpinner, { "aria-label": "Loading", alt: "Loading spinner", loadingDescription: loadingDescription }))) : (optionList) }) }) }));
});
//# sourceMappingURL=DialogComboboxOptionControlledList.js.map