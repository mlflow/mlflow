import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
/** @jsxImportSource @emotion/react */
import { css } from '@emotion/react';
import { useCallback, useMemo, useState } from 'react';
import { ListboxInput } from './ListboxInput';
import { ListboxOptions } from './ListboxOptions';
import { ListboxRoot, useListboxContext } from './ListboxRoot';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, useDesignSystemTheme, useNotifyOnFirstView, } from '../../design-system';
const ListboxContent = ({ options, filterValue, setFilterValue, filterInputPlaceholder, onSelect, ariaLabel, includeFilterInput, filterInputEmptyMessage, listBoxDivRef, }) => {
    const [highlightedValue, setHighlightedValue] = useState();
    const { listboxId } = useListboxContext();
    const designSystemTheme = useDesignSystemTheme();
    const filteredOptions = useMemo(() => {
        if (!filterValue)
            return options;
        const lowerFilter = filterValue.toLowerCase();
        return options.filter((option) => option.value.toLowerCase().includes(lowerFilter) || option.label.toLowerCase().includes(lowerFilter));
    }, [filterValue, options]);
    return (_jsxs("div", { css: css({
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
        }), ref: listBoxDivRef, children: [includeFilterInput && (_jsxs(_Fragment, { children: [_jsx(ListboxInput, { value: filterValue, onChange: setFilterValue, placeholder: filterInputPlaceholder, "aria-controls": listboxId, "aria-activedescendant": highlightedValue ? `${listboxId}-${highlightedValue}` : undefined, options: filteredOptions }), filteredOptions.length === 0 && filterInputEmptyMessage && (_jsx("div", { "aria-live": "polite", role: "status", css: {
                            color: designSystemTheme.theme.colors.textSecondary,
                            textAlign: 'center',
                            padding: '6px 12px',
                            width: '100%',
                            boxSizing: 'border-box',
                        }, children: filterInputEmptyMessage }))] })), _jsx(ListboxOptions, { options: filteredOptions, onSelect: onSelect, onHighlight: setHighlightedValue, "aria-label": ariaLabel })] }));
};
export const Listbox = ({ options, onSelect, includeFilterInput, filterInputEmptyMessage, initialSelectedValue, filterInputPlaceholder, 'aria-label': ariaLabel, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], valueHasNoPii, className, }) => {
    const [filterValue, setFilterValue] = useState('');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Listbox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const handleSelect = useCallback((value) => {
        eventContext.onValueChange(value);
        onSelect?.(value);
    }, [eventContext, onSelect]);
    const { elementRef: listBoxDivRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: initialSelectedValue,
    });
    return (_jsx(ListboxRoot, { className: className, onSelect: handleSelect, initialSelectedValue: initialSelectedValue, listBoxDivRef: listBoxDivRef, children: _jsx(ListboxContent, { options: options, filterValue: filterValue, setFilterValue: setFilterValue, filterInputPlaceholder: filterInputPlaceholder, onSelect: handleSelect, ariaLabel: ariaLabel, includeFilterInput: includeFilterInput, filterInputEmptyMessage: filterInputEmptyMessage, listBoxDivRef: listBoxDivRef }) }));
};
//# sourceMappingURL=index.js.map