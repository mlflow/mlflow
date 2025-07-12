import { useCombobox, useMultipleSelection, } from 'downshift';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../../DesignSystemEventProvider';
import { safex } from '../../utils/safex';
const mapItemsToString = (items, itemToString) => {
    return JSON.stringify(items.map(itemToString));
};
export const TypeaheadComboboxStateChangeTypes = useCombobox.stateChangeTypes;
export const TypeaheadComboboxMultiSelectStateChangeTypes = useMultipleSelection.stateChangeTypes;
export function useComboboxState({ allItems, items, itemToString, onIsOpenChange, allowNewValue = false, formValue, formOnChange, formOnBlur, componentId, valueHasNoPii, analyticsEvents, matcher, preventUnsetOnBlur = false, ...props }) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.typeaheadCombobox', false);
    const getFilteredItems = useCallback((inputValue) => {
        const lowerCasedInputValue = inputValue?.toLowerCase() ?? '';
        // If the input is empty or if there is no matcher supplied, do not filter
        return allItems.filter((item) => !inputValue || !matcher || matcher(item, lowerCasedInputValue));
    }, [allItems, matcher]);
    const [inputValue, setInputValue] = useState();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const itemToStringWithDefaultToString = useCallback((item) => {
        return item ? (itemToString ? itemToString(item) : item.toString()) : '';
    }, [itemToString]);
    const prevAllItems = useRef(allItems);
    useEffect(() => {
        // When allItems changes, re-apply filter so users don't see stale items values in the dropdown box
        if (!props.multiSelect &&
            (allItems.length !== prevAllItems.current.length ||
                // Avoid redundant or endless updates by checking individual elements as allItems may have an unstable reference.
                allItems.some((item, index) => itemToStringWithDefaultToString(item) !== itemToStringWithDefaultToString(prevAllItems.current[index])))) {
            props.setItems(getFilteredItems(inputValue));
            prevAllItems.current = allItems;
        }
    }, [allItems, inputValue, getFilteredItems, props, itemToStringWithDefaultToString]);
    const comboboxState = {
        ...useCombobox({
            onIsOpenChange: onIsOpenChange,
            onInputValueChange: ({ inputValue }) => {
                if (inputValue !== undefined) {
                    setInputValue(inputValue);
                    props.setInputValue?.(inputValue);
                }
                if (!props.multiSelect) {
                    props.setItems(getFilteredItems(inputValue));
                }
            },
            items: items,
            itemToString: itemToStringWithDefaultToString,
            defaultHighlightedIndex: props.multiSelect ? 0 : undefined, // after selection for multiselect, highlight the first item.
            scrollIntoView: () => { }, // disabling scroll because floating-ui repositions the menu
            selectedItem: props.multiSelect ? null : formValue, // useMultipleSelection will handle the item selection for multiselect
            stateReducer(state, actionAndChanges) {
                const { changes, type } = actionAndChanges;
                switch (type) {
                    case useCombobox.stateChangeTypes.InputBlur:
                        if (preventUnsetOnBlur) {
                            return changes;
                        }
                        if (!props.multiSelect) {
                            // If allowNewValue is true, register the input's current value on blur
                            if (allowNewValue) {
                                const newInputValue = state.inputValue === '' ? null : state.inputValue;
                                formOnChange?.(newInputValue);
                                formOnBlur?.(newInputValue);
                            }
                            else {
                                // If allowNewValue is false, clear value on blur
                                formOnChange?.(null);
                                formOnBlur?.(null);
                            }
                        }
                        else {
                            formOnBlur?.(state.selectedItem);
                        }
                        return changes;
                    case useCombobox.stateChangeTypes.InputKeyDownEnter:
                    case useCombobox.stateChangeTypes.ItemClick:
                        formOnChange?.(changes.selectedItem);
                        return {
                            ...changes,
                            highlightedIndex: props.multiSelect ? state.highlightedIndex : 0, // on multiselect keep the highlighted index unchanged.
                            isOpen: props.multiSelect ? true : false, // for multiselect, keep the menu open after selection.
                        };
                    default:
                        return changes;
                }
            },
            onStateChange: (args) => {
                const { type, selectedItem: newSelectedItem, inputValue: newInputValue } = args;
                props.onStateChange?.(args);
                if (props.multiSelect) {
                    switch (type) {
                        case useCombobox.stateChangeTypes.InputKeyDownEnter:
                        case useCombobox.stateChangeTypes.ItemClick:
                        case useCombobox.stateChangeTypes.InputBlur:
                            if (newSelectedItem) {
                                props.setSelectedItems([...props.selectedItems, newSelectedItem]);
                                props.setInputValue('');
                                formOnBlur?.([...props.selectedItems, newSelectedItem]);
                            }
                            break;
                        case useCombobox.stateChangeTypes.InputChange:
                            props.setInputValue(newInputValue ?? '');
                            break;
                        case useCombobox.stateChangeTypes.FunctionReset:
                            eventContext.onValueChange?.('[]');
                            break;
                        default:
                            break;
                    }
                    // Unselect when clicking selected item
                    if (newSelectedItem && props.selectedItems.includes(newSelectedItem)) {
                        const newSelectedItems = props.selectedItems.filter((item) => item !== newSelectedItem);
                        props.setSelectedItems(newSelectedItems);
                        eventContext.onValueChange?.(mapItemsToString(newSelectedItems, itemToStringWithDefaultToString));
                    }
                    else if (newSelectedItem) {
                        eventContext.onValueChange?.(mapItemsToString([...props.selectedItems, newSelectedItem], itemToStringWithDefaultToString));
                    }
                }
                else if (newSelectedItem) {
                    eventContext.onValueChange?.(itemToStringWithDefaultToString(newSelectedItem));
                }
                else if (type === useCombobox.stateChangeTypes.FunctionReset) {
                    eventContext.onValueChange?.('');
                }
            },
            initialInputValue: props.initialInputValue,
            initialSelectedItem: props.initialSelectedItem,
        }),
        componentId,
        analyticsEvents,
        valueHasNoPii,
        onView: eventContext.onView,
        firstOnViewValue: props.multiSelect
            ? mapItemsToString(props.selectedItems, itemToStringWithDefaultToString)
            : itemToStringWithDefaultToString(props.initialSelectedItem ?? null),
    };
    return comboboxState;
}
export function useMultipleSelectionState(selectedItems, setSelectedItems, { componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange], valueHasNoPii, itemToString, }) {
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    return useMultipleSelection({
        selectedItems,
        onStateChange({ selectedItems: newSelectedItems, type }) {
            switch (type) {
                case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownBackspace:
                case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownDelete:
                case useMultipleSelection.stateChangeTypes.DropdownKeyDownBackspace:
                case useMultipleSelection.stateChangeTypes.FunctionRemoveSelectedItem:
                    setSelectedItems(newSelectedItems || []);
                    break;
                default:
                    break;
            }
            const itemToStringWithDefaultToString = itemToString ?? ((item) => item?.toString() ?? '');
            eventContext.onValueChange?.(mapItemsToString(newSelectedItems || [], itemToStringWithDefaultToString));
        },
    });
}
//# sourceMappingURL=downshiftHookWrappers.js.map