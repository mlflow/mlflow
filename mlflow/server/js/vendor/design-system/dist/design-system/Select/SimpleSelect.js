import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import React, { createContext, forwardRef, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState, } from 'react';
import { Select, SelectContent, SelectOption, SelectOptionGroup, SelectTrigger } from '.';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { useDialogComboboxContext } from '../DialogCombobox/hooks/useDialogComboboxContext';
import { useNotifyOnFirstView } from '../utils';
import { safex } from '../utils/safex';
const SimpleSelectContext = createContext(undefined);
const getSelectedOption = (children, value) => {
    const childArray = React.Children.toArray(children);
    for (const child of childArray) {
        if (React.isValidElement(child)) {
            if (child.type === SimpleSelectOption && child.props.value === value) {
                return child;
            }
            if (child.props.children) {
                const nestedOption = getSelectedOption(child.props.children, value);
                if (nestedOption) {
                    return nestedOption;
                }
            }
        }
    }
    return undefined;
};
const getSelectedOptionLabel = (children, value) => {
    const selectedOption = getSelectedOption(children, value);
    if (React.isValidElement(selectedOption)) {
        return selectedOption.props.children;
    }
    return '';
};
/**
 * This is the future `Select` component which simplifies the API of the current Select primitives.
 * It is temporarily named `SimpleSelect` pending cleanup.
 */
export const SimpleSelect = forwardRef(({ defaultValue, name, placeholder, children, contentProps, onChange, onOpenChange, id, label, value, validationState, forceCloseOnEscape, componentId, analyticsEvents, valueHasNoPii, ...rest }, ref) => {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.simpleSelect', false);
    const [defaultLabel] = useState(() => {
        if (value) {
            return getSelectedOptionLabel(children, value);
        }
        return '';
    });
    const innerRef = useRef(null);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    useImperativeHandle(ref, () => innerRef.current, []);
    const previousExternalValue = useRef(value);
    const [internalValue, setInternalValue] = useState(value);
    const [selectedLabel, setSelectedLabel] = useState(defaultLabel);
    const isControlled = value !== undefined;
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.SimpleSelect,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
    });
    const { elementRef: simpleSelectRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: value ?? defaultValue,
    });
    // Controlled state setup.
    useEffect(() => {
        if (value !== undefined && value !== previousExternalValue.current) {
            setInternalValue(value);
            previousExternalValue.current = value;
        }
    }, [value]);
    // Uncontrolled state setup.
    useEffect(() => {
        if (isControlled) {
            return;
        }
        // Set initial state.
        const element = innerRef.current;
        const initialValue = defaultValue || element?.value || '';
        setInternalValue(initialValue);
        previousExternalValue.current = initialValue;
    }, [isControlled, defaultValue, value]);
    // Separately update the label when the value changes; this responds
    // to either the controlled or uncontrolled useEffects above.
    useEffect(() => {
        setSelectedLabel(getSelectedOptionLabel(children, internalValue || ''));
    }, [internalValue, children]);
    // Handles controlled state, and propagates changes to the input element.
    const handleChange = (newValue) => {
        eventContext.onValueChange(newValue);
        innerRef.current?.setAttribute('value', newValue || '');
        setInternalValue(newValue);
        setSelectedLabel(getSelectedOptionLabel(children, newValue));
        if (onChange) {
            onChange({
                target: {
                    name,
                    type: 'select',
                    value: newValue,
                },
                type: 'change',
            });
        }
    };
    return (_jsx(SimpleSelectContext.Provider, { value: { value: internalValue, onChange: handleChange }, children: _jsx(Select
        // SimpleSelect emits its own value change events rather than emitting them from the underlying
        // DialogCombobox due to how SimpleSelect sets its initial state. The Select componentId is explicitly
        // set to undefined to prevent it from emitting events if the componentId prop is required in the future.
        , { 
            // SimpleSelect emits its own value change events rather than emitting them from the underlying
            // DialogCombobox due to how SimpleSelect sets its initial state. The Select componentId is explicitly
            // set to undefined to prevent it from emitting events if the componentId prop is required in the future.
            componentId: undefined, value: internalValue, placeholder: placeholder, label: label ?? rest['aria-label'], id: id, children: _jsxs(SimpleSelectContentWrapper, { onOpenChange: onOpenChange, children: [_jsx(SelectTrigger, { ref: simpleSelectRef, ...rest, validationState: validationState, onClear: () => {
                            handleChange('');
                        }, id: id, value: internalValue, ...eventContext.dataComponentProps, children: selectedLabel || placeholder }), _jsx("input", { type: "hidden", ref: innerRef }), _jsx(SelectContent, { forceCloseOnEscape: forceCloseOnEscape, ...contentProps, children: children })] }) }) }));
});
// This component is used to propagate the open state of the DialogCombobox to the SimpleSelect.
// We don't directly pass through `onOpenChange` since it's tied into the actual state; `SimpleSelect` merely
// needs to communicate via the optional prop if the dropdown is open or not and doesn't need to control it.
const SimpleSelectContentWrapper = ({ children, onOpenChange }) => {
    const { isOpen } = useDialogComboboxContext();
    useEffect(() => {
        if (onOpenChange) {
            onOpenChange(Boolean(isOpen));
        }
    }, [isOpen, onOpenChange]);
    return _jsx(_Fragment, { children: children });
};
export const SimpleSelectOption = forwardRef(({ value, children, ...rest }, ref) => {
    const context = useContext(SimpleSelectContext);
    if (!context) {
        throw new Error('SimpleSelectOption must be used within a SimpleSelect');
    }
    const { onChange } = context;
    return (_jsx(SelectOption, { ...rest, ref: ref, value: value, onChange: ({ value }) => {
            onChange(value);
        }, children: children }));
});
export const SimpleSelectOptionGroup = ({ children, label, ...props }) => {
    const context = useContext(SimpleSelectContext);
    if (!context) {
        throw new Error('SimpleSelectOptionGroup must be used within a SimpleSelect');
    }
    return (_jsx(SelectOptionGroup, { ...props, name: label, children: children }));
};
//# sourceMappingURL=SimpleSelect.js.map