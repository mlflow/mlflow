import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import * as Popover from '@radix-ui/react-popover';
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { DialogComboboxContextProvider } from './providers/DialogComboboxContext';
import { ComponentFinderContext, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { safex } from '../utils/safex';
export const DialogCombobox = ({ children, label, id, value = [], open, emptyText, scrollToSelectedElement = true, rememberLastScrollPosition = false, componentId, analyticsEvents, valueHasNoPii, onOpenChange, ...props }) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dialogCombobox', false);
    // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
    // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
    const [isControlled, setIsControlled] = useState(false);
    const [selectedValue, setSelectedValue] = useState(value);
    const [isOpen, setIsOpenState] = useState(Boolean(open));
    const setIsOpen = useCallback((isOpen) => {
        setIsOpenState(isOpen);
        onOpenChange?.(isOpen);
    }, [setIsOpenState, onOpenChange]);
    const [contentWidth, setContentWidth] = useState();
    const [textOverflowMode, setTextOverflowMode] = useState('multiline');
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const [disableMouseOver, setDisableMouseOver] = useState(false);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DialogCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const setSelectedValueWrapper = useCallback((newValue) => {
        eventContext.onValueChange(JSON.stringify(newValue));
        setSelectedValue(newValue);
    }, [eventContext]);
    useEffect(() => {
        if (((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value) ||
            (selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i) => v === value[i]))) {
            return;
        }
        if (!isControlled) {
            setSelectedValueWrapper(value);
        }
    }, [value, isControlled, selectedValue, setSelectedValueWrapper]);
    return (_jsx(DialogComboboxContextProvider, { value: {
            id,
            label,
            value: selectedValue,
            setValue: setSelectedValueWrapper,
            setIsControlled,
            contentWidth,
            setContentWidth,
            textOverflowMode,
            setTextOverflowMode,
            isInsideDialogCombobox: true,
            multiSelect: props.multiSelect,
            stayOpenOnSelection: props.stayOpenOnSelection,
            isOpen,
            setIsOpen,
            emptyText,
            scrollToSelectedElement,
            rememberLastScrollPosition,
            componentId,
            analyticsEvents,
            valueHasNoPii,
            disableMouseOver,
            setDisableMouseOver,
            onView: eventContext.onView,
        }, children: _jsx(Root, { open: open !== undefined ? open : isOpen, ...props, children: _jsx(ComponentFinderContext.Provider, { value: { dataComponentProps: eventContext.dataComponentProps }, children: children }) }) }));
};
const Root = (props) => {
    const { children, stayOpenOnSelection, multiSelect, ...restProps } = props;
    const { value, setIsOpen, onView } = useDialogComboboxContext();
    const firstView = useRef(true);
    useEffect(() => {
        if (firstView.current) {
            onView(value);
            firstView.current = false;
        }
    }, [onView, value]);
    const handleOpenChange = (open) => {
        setIsOpen(open);
    };
    useEffect(() => {
        if (!stayOpenOnSelection && (typeof stayOpenOnSelection === 'boolean' || !multiSelect)) {
            setIsOpen(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [value, stayOpenOnSelection, multiSelect]); // Don't trigger when setIsOpen changes.
    return (_jsx(Popover.Root, { onOpenChange: handleOpenChange, ...restProps, children: children }));
};
//# sourceMappingURL=DialogCombobox.js.map