import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Input as AntDInput } from 'antd';
import { forwardRef, useCallback, useEffect, useMemo, useRef } from 'react';
import { getInputEmotionStyles } from './Input';
import { useFormContext } from '../../development/Form';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useCallbackOnEnter } from '../utils/useCallbackOnEnter';
import { useCallbackOnceUntilReset } from '../utils/useCallbackOnceUntilReset';
import { useChainEventHandlers } from '../utils/useChainEventHandlers';
export const TextArea = forwardRef(function TextArea({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, componentId, analyticsEvents, onChange, onFocus, onKeyDown, onCompositionStart, onCompositionEnd, allowFormSubmitOnEnter = false, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.textArea', false);
    const formContext = useFormContext();
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TextArea,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: false,
    });
    const firstView = useRef(true);
    useEffect(() => {
        if (firstView.current) {
            eventContext.onView();
            firstView.current = false;
        }
    }, [eventContext]);
    // Prevents multiple onValueChange triggers until after a focus event resets it.
    const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } = useCallbackOnceUntilReset(eventContext.onValueChange);
    const handleChange = useCallback((e) => {
        sendAnalyticsEventOncePerFocus();
        onChange?.(e);
    }, [onChange, sendAnalyticsEventOncePerFocus]);
    const handleFocus = useCallback((e) => {
        resetSendAnalyticsEventOnFocus();
        onFocus?.(e);
    }, [onFocus, resetSendAnalyticsEventOnFocus]);
    // Callback used to submit the parent form
    // This is used when allowFormSubmitOnEnter is true
    const handleSubmitForm = useCallback((event) => {
        if (!formContext.formRef?.current)
            return;
        event.preventDefault();
        formContext.formRef.current.requestSubmit();
    }, [formContext.formRef]);
    // Event handlers for submitting the form on Enter
    // This hook takes meta keys into account for platform-specific behavior
    const submitHandlers = useCallbackOnEnter({
        callback: handleSubmitForm,
        allowBasicEnter: allowFormSubmitOnEnter, // Conditionally allow form submission on Enter
        allowPlatformEnter: true, // Always allow form submission on platform Enter
    });
    // Chains the event handlers from useCallbackOnEnter together with potential event handlers from props
    const onKeyDownChain = useChainEventHandlers(useMemo(() => ({
        handlers: [submitHandlers.onKeyDown, onKeyDown],
        stopOnDefaultPrevented: true, // If the form is submitted, do not call the next handler
    }), [onKeyDown, submitHandlers.onKeyDown]));
    const onCompositionStartChain = useChainEventHandlers(useMemo(() => ({ handlers: [submitHandlers.onCompositionStart, onCompositionStart] }), [submitHandlers.onCompositionStart, onCompositionStart]));
    const onCompositionEndChain = useChainEventHandlers(useMemo(() => ({ handlers: [submitHandlers.onCompositionEnd, onCompositionEnd] }), [submitHandlers.onCompositionEnd, onCompositionEnd]));
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDInput.TextArea, { ...addDebugOutlineIfEnabled(), ref: ref, autoComplete: autoComplete, css: [
                getInputEmotionStyles(classNamePrefix, theme, { validationState, useNewShadows, useNewBorderRadii }),
                dangerouslyAppendEmotionCSS,
            ], onChange: handleChange, onFocus: handleFocus, onKeyDown: onKeyDownChain, onCompositionStart: onCompositionStartChain, onCompositionEnd: onCompositionEndChain, "data-component-type": DesignSystemEventProviderComponentTypes.TextArea, "data-component-id": componentId, ...props, ...dangerouslySetAntdProps }) }));
});
//# sourceMappingURL=TextArea.js.map