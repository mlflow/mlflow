import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import React, { useCallback, useMemo, useRef, useState } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../../design-system/DesignSystemEventProvider';
import { LoadingState } from '../../design-system/LoadingState';
const FormContext = React.createContext({ componentId: undefined, isSubmitting: false, formRef: undefined });
export const useFormContext = () => React.useContext(FormContext);
/**
 * Form is a wrapper around the form element that allows us to track the active element before the form is submitted.
 * This is useful for analytics purposes to know what component was last focused when the form was sent for submission.
 *
 * NOTE: Form component cannot be nested.
 */
export const Form = ({ children, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnSubmit], onSubmit, shouldStartInteraction, ...otherProps }) => {
    const formContext = useFormContext();
    const formRef = useRef(null);
    const eventRef = useRef(null);
    if (formContext.componentId !== undefined) {
        throw new Error('DuBois Form component cannot be nested');
    }
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Form,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction,
    });
    const [isSubmitting, setIsSubmitting] = useState(false);
    const onSubmitHandler = useCallback(async (e) => {
        try {
            setIsSubmitting(true);
            eventContext.onSubmit({
                event: e,
                initialState: undefined,
                finalState: undefined,
                referrerComponent: extractReferrerComponent(),
            });
            await onSubmit(e);
        }
        finally {
            setIsSubmitting(false);
        }
    }, [eventContext, onSubmit]);
    const contextValue = useMemo(() => ({ componentId, isSubmitting, eventRef, formRef }), [componentId, isSubmitting]);
    return (_jsx(FormContext.Provider, { value: contextValue, children: _jsxs("form", { onSubmit: onSubmitHandler, ...otherProps, ref: formRef, ...eventContext.dataComponentProps, children: [isSubmitting && _jsx(LoadingState, { description: componentId }), children] }) }));
};
/**
 * Form is a wrapper around the Form DuBois component to better integrate with React Hook Form and to track the active
 * element before the form is submitted. This is useful for analytics purposes to know what component was last focused
 * when the form was sent for submission, along with being able to pass in the initial and final values of the form.
 *
 * Internal implementation details:
 * The form is wrapped in a Form component to share implementation details along with associating the HTML form
 * submission event with interaction details. The Form component does not emit analytics events, but the RHF
 * component will emit the events once the RHF lifecycle events occur so we can relay the form property values.
 *
 * NOTE: Form components cannot be nested.
 */
export const RhfForm = ({ handleSubmit, handleValid, handleInvalid, processFinalPropertyValues, children, componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnSubmit], shouldStartInteraction, initialState, ...otherProps }) => {
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Form,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction,
    });
    const onValidWrapper = useCallback(async (data, e) => {
        const refinedData = processFinalPropertyValues ? processFinalPropertyValues(data) : data;
        eventContext.onSubmit({
            event: e,
            initialState,
            finalState: refinedData,
            referrerComponent: extractReferrerComponent(),
        });
        await handleValid(refinedData, e);
    }, [eventContext, initialState, handleValid, processFinalPropertyValues]);
    const onInvalidWrapper = useCallback(async (errors, e) => {
        eventContext.onSubmit({
            event: e,
            initialState,
            finalState: undefined,
            referrerComponent: extractReferrerComponent(),
        });
        if (handleInvalid) {
            await handleInvalid(errors, e);
        }
    }, [eventContext, initialState, handleInvalid]);
    const onSubmitWrapper = useCallback(async (e) => {
        await handleSubmit(onValidWrapper, onInvalidWrapper)(e);
    }, [handleSubmit, onInvalidWrapper, onValidWrapper]);
    return (_jsx(Form, { componentId: componentId, analyticsEvents: [], onSubmit: onSubmitWrapper, ...otherProps, children: children }));
};
function extractReferrerComponent() {
    const activeElement = document.activeElement;
    const { componentType, componentId } = activeElement?.dataset ?? {};
    if (componentType && componentId) {
        return {
            type: componentType,
            id: componentId,
        };
    }
    return undefined;
}
//# sourceMappingURL=Form.js.map