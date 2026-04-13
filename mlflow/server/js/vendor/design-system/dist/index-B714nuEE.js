import { jsx } from '@emotion/react/jsx-runtime';
import { css } from '@emotion/react';
import { Input as Input$2 } from 'antd';
import { u as useDesignSystemTheme, D as DesignSystemAntDConfigProvider, a as addDebugOutlineIfEnabled, g as getValidationStateColor, b as getAnimationCss, s as safex, c as serverSideSafe, d as useFormContext, e as DesignSystemEventProviderAnalyticsEventTypes, f as useDesignSystemEventComponentCallbacks, h as DesignSystemEventProviderComponentTypes, L as LockIcon, i as importantify } from './Tabs-Z65utToH.js';
import React__default, { useMemo, useRef, useCallback, forwardRef, useEffect } from 'react';
import '@ant-design/icons';

/**
 * ES-1267895 When entering composed characters, we should not submit forms on Enter.
 * For instance Japanese characters are composed and we should not submit forms when
 * the user is still composing the characters.
 *
 * This hook provides a reusable way to invoke a callback on Enter,
 * but not when composing characters.
 * This can be used to invoke a form submission callback when Enter is pressed.
 * @param callback VoidFunction to call when the Enter is pressed
 * @param allowBasicEnter If true, the callback will be invoked when Enter is pressed without any modifiers
 * @param allowPlatformEnter If true, the callback will be invoked when Enter is pressed with the platform modifier (CMD on Mac, CTRL on Windows)
 * @returns Object with onKeyDown, onCompositionEnd, and onCompositionStart event handlers
 *
 * @example
 * ```tsx
 * const handleSubmit = (event: React.KeyboardEvent) => {
 *  event.preventDefault();
 * // Submit the form
 * };
 * const eventHandlers = useCallbackOnEnter({
 *   callback: handleSubmit,
 *   allowBasicEnter: true,
 *   allowPlatformEnter: true,
 * })
 * return <input {...eventHandlers} />;
 * ```
 */ const useCallbackOnEnter = ({ callback, allowBasicEnter, allowPlatformEnter })=>{
    const isMacOs = useMemo(()=>navigator.userAgent.includes('Mac'), []);
    // Keeping track of whether we are composing characters
    // This is stored in a ref so that it can be accessed in the onKeyDown event handler
    // without causing a re-renders
    const isComposing = useRef(false);
    // Handler for when the composition starts
    const onCompositionStart = useCallback(()=>{
        isComposing.current = true;
    }, []);
    // Handler for when the composition ends
    const onCompositionEnd = useCallback(()=>{
        isComposing.current = false;
    }, []);
    // Handler for when a key is pressed
    // Used to submit the form when Enter is pressed
    const onKeyDown = useCallback((event)=>{
        // Only invoke the callback on Enter
        if (event.key !== 'Enter') return;
        // Do not submit on Enter if user is composing characters
        if (isComposing.current) return;
        // Do not submit on Enter if both are false
        if (!allowBasicEnter && !allowPlatformEnter) return;
        // Check if the event is a valid Enter press
        const basicEnter = allowBasicEnter && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;
        const platformEnter = allowPlatformEnter && (isMacOs ? event.metaKey : event.ctrlKey);
        const isValidEnterPress = basicEnter || platformEnter;
        // Submit the form if the Enter press is valid
        if (isValidEnterPress) callback(event);
    }, [
        allowBasicEnter,
        allowPlatformEnter,
        callback,
        isMacOs
    ]);
    return {
        onKeyDown,
        onCompositionEnd,
        onCompositionStart
    };
};

const getInputGroupStyling = (clsPrefix, theme, buttonSide)=>{
    const inputClass = `.${clsPrefix}-input`;
    const buttonClass = `.${clsPrefix}-btn`;
    const buttonOnRight = buttonSide === 'right';
    return /*#__PURE__*/ css({
        display: 'inline-flex !important',
        width: 'auto',
        // This is a hack to achieve (0, 3, 0) specificity, so that
        // these styles ALWAYS override the (0,2,0) specificity
        // styles with !important present in Input.
        [`&& > ${inputClass}`]: {
            flexGrow: 1,
            // Square off the inner edge (the edge adjacent to the button).
            ...buttonOnRight ? {
                borderTopRightRadius: '0px !important',
                borderBottomRightRadius: '0px !important'
            } : {
                borderTopLeftRadius: '0px !important',
                borderBottomLeftRadius: '0px !important'
            },
            '&:disabled': {
                border: 'none',
                background: theme.colors.actionDisabledBackground,
                '&:hover': {
                    borderRight: `1px solid ${theme.colors.actionDisabledBorder} !important`
                }
            },
            '&[data-validation]': {
                marginRight: 0
            }
        },
        [`& > ${buttonClass}`]: {
            boxShadow: 'none !important',
            // Square off the inner edge (the edge adjacent to the input).
            ...buttonOnRight ? {
                borderTopLeftRadius: '0px !important',
                borderBottomLeftRadius: '0px !important'
            } : {
                borderTopRightRadius: '0px !important',
                borderBottomRightRadius: '0px !important',
                // In button-left mode the input (later in DOM) renders on top of the button's
                // right border by default. Elevate the button on hover so its right border
                // hover state is not hidden behind the input.
                '&:hover': {
                    position: 'relative',
                    zIndex: 2
                }
            }
        },
        [`& > ${buttonClass} > span`]: {
            verticalAlign: 'middle'
        },
        [`& > ${buttonClass}:disabled, & > ${buttonClass}:disabled:hover`]: {
            borderLeft: `1px solid ${theme.colors.actionDisabledBorder} !important`,
            backgroundColor: `${theme.colors.actionDisabledBackground} !important`,
            color: `${theme.colors.actionDisabledText} !important`
        }
    });
};
const Group = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, compact = true, buttonSide = 'right', ...props })=>{
    const { classNamePrefix, theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Input$2.Group, {
            ...addDebugOutlineIfEnabled(),
            css: [
                getInputGroupStyling(classNamePrefix, theme, buttonSide),
                dangerouslyAppendEmotionCSS
            ],
            compact: compact,
            ...props,
            ...dangerouslySetAntdProps
        })
    });
};

/**
 * A React custom hook that allows a callback function to be executed exactly once until it is explicitly reset.
 *
 * Usage:
 *
 * const originalCallback = () => { console.log('originalCallback'); }
 * const { callbackOnceUntilReset, reset } = useCallbackOnceUntilReset(originalCallback);
 *
 * // To execute the callback
 * callbackOnceUntilReset(); // Prints 'originalCallback'
 * callbackOnceUntilReset(); // No effect for further calls
 * reset();
 * callbackOnceUntilReset(); // Prints 'originalCallback' again
 */ const useCallbackOnceUntilReset = (callback)=>{
    const canTriggerRef = useRef(true);
    const reset = useCallback(()=>{
        canTriggerRef.current = true;
    }, []);
    const callbackOnceUntilReset = useCallback(()=>{
        if (canTriggerRef.current) {
            callback();
            canTriggerRef.current = false;
        }
    }, [
        callback
    ]);
    return {
        callbackOnceUntilReset,
        reset
    };
};

/**
 * This hook provides a way to chain event handlers together.
 * Optionally, it can stop calling the next handler if the event has been defaultPrevented.
 * @param handlers Array of event handlers to chain together. Optional handlers are allowed for convenience.
 * @param stopOnDefaultPrevented If true, the next handler will not be called if the event has been defaultPrevented
 * @returns A function that will call each handler in the order they are provided
 * @example
 * ```tsx
 * const onClick = useChainEventHandlers({ handlers: [onClick1, onClick2] });
 * return <button onClick={onClick} />;
 */ const useChainEventHandlers = (props)=>{
    const { handlers, stopOnDefaultPrevented } = props;
    return useCallback((event)=>{
        // Loop over each handler in succession
        for (const handler of handlers){
            // Break if the event has been defaultPrevented and stopOnDefaultPrevented is true
            if (stopOnDefaultPrevented && event.defaultPrevented) return;
            // Call the handler if it exists
            handler?.(event);
        }
    }, [
        handlers,
        stopOnDefaultPrevented
    ]);
};

const SMALL_INPUT_HEIGHT = 24;
const getInputStyles = (clsPrefix, theme, { validationState, type, hasValue, locked, size }, { useFocusWithin = false })=>{
    const inputClass = `.${clsPrefix}-input`;
    const affixClass = `.${clsPrefix}-input-affix-wrapper`;
    const affixClassDisabled = `.${clsPrefix}-input-affix-wrapper-disabled`;
    const affixClassFocused = `.${clsPrefix}-input-affix-wrapper-focused`;
    const clearIcon = `.${clsPrefix}-input-clear-icon`;
    const prefixIcon = `.${clsPrefix}-input-prefix`;
    const suffixIcon = `.${clsPrefix}-input-suffix`;
    const validationColor = getValidationStateColor(theme, validationState);
    const focusSpecifier = useFocusWithin ? 'focus-within' : 'focus';
    const styles = {
        '&&': {
            fontSize: theme.typography.fontSizeBase,
            lineHeight: theme.typography.lineHeightBase,
            ...size === 'small' ? {
                height: SMALL_INPUT_HEIGHT
            } : {
                minHeight: theme.general.heightSm
            },
            ...size === 'small' && {
                paddingLeft: theme.spacing.sm,
                paddingRight: theme.spacing.sm
            },
            borderColor: theme.colors.actionDefaultBorderDefault,
            ...validationState && {
                borderColor: validationColor
            },
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
            },
            [`&:${focusSpecifier}`]: {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                borderColor: 'transparent'
            },
            '&:focus-visible': {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                borderColor: 'transparent'
            },
            [`& + .${clsPrefix}-form-message`]: {
                marginTop: theme.spacing.sm
            }
        },
        [`&${inputClass}, ${inputClass}`]: {
            backgroundColor: 'transparent',
            borderRadius: theme.borders.borderRadiusSm,
            '&:disabled': {
                backgroundColor: theme.colors.actionDisabledBackground,
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder
            },
            '&::placeholder': {
                color: theme.colors.textPlaceholder
            }
        },
        [`&${affixClass}`]: {
            backgroundColor: 'transparent',
            fontSize: theme.typography.fontSizeBase,
            lineHeight: theme.typography.lineHeightBase,
            paddingTop: 5,
            paddingBottom: 5,
            ...size === 'small' ? {
                height: SMALL_INPUT_HEIGHT
            } : {
                minHeight: theme.general.heightSm
            },
            '::before': {
                lineHeight: theme.typography.lineHeightBase
            },
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
            },
            [`input.${clsPrefix}-input`]: {
                fontSize: 'inherit',
                borderRadius: 0
            }
        },
        [`&${affixClassDisabled}`]: {
            backgroundColor: theme.colors.actionDisabledBackground
        },
        [`&${affixClassFocused}`]: {
            boxShadow: 'none',
            [`&&, &:${focusSpecifier}`]: {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                boxShadow: 'none',
                borderColor: 'transparent'
            }
        },
        [clearIcon]: {
            fontSize: theme.typography.fontSizeSm
        },
        [prefixIcon]: {
            marginRight: theme.spacing.sm,
            color: theme.colors.textSecondary
        },
        [suffixIcon]: {
            marginLeft: theme.spacing.sm,
            color: theme.colors.textSecondary,
            ...!hasValue && type === 'number' && {
                display: 'none'
            }
        },
        ...locked ? {
            [`&${affixClass}`]: {
                backgroundColor: theme.colors.backgroundSecondary
            },
            [`input.${clsPrefix}-input`]: {
                backgroundColor: `${theme.colors.backgroundSecondary} !important`,
                color: `${theme.colors.textPrimary} !important`
            }
        } : {},
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return styles;
};
const getInputEmotionStyles = (clsPrefix, theme, { validationState, type, hasValue, locked, size })=>{
    const styles = getInputStyles(clsPrefix, theme, {
        validationState,
        type,
        hasValue,
        locked,
        size
    }, {});
    return /*#__PURE__*/ css(importantify(styles));
};
const Input$1 = /*#__PURE__*/ forwardRef(function Input({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, onChange, onClear, onFocus, onPressEnter, onWheel, onCompositionStart, onCompositionEnd, componentId, shouldPreventFormSubmission, analyticsEvents, readOnly, locked, size = 'middle', ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.input', false);
    const disableNumberInputWheelValueChange = serverSideSafe('databricks.fe.designsystem.disableNumberInputWheelValueChange', false);
    const formContext = useFormContext();
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const [hasValue, setHasValue] = React__default.useState(props.value !== undefined && props.value !== null && props.value !== '');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Input,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: false
    });
    const firstView = useRef(true);
    useEffect(()=>{
        if (firstView.current) {
            eventContext.onView();
            firstView.current = false;
        }
    }, [
        eventContext
    ]);
    // Prevents multiple onValueChange triggers until after a focus event resets it.
    const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } = useCallbackOnceUntilReset(eventContext.onValueChange);
    const handleChange = useCallback((e)=>{
        sendAnalyticsEventOncePerFocus();
        // If the input is cleared, call the onClear handler, but only
        // if the event is not an input event -- which is the case when you click the
        // ant-provided (X) button.
        if (!e.target.value && e.nativeEvent instanceof InputEvent === false && onClear) {
            onClear?.();
            setHasValue(false);
        } else {
            onChange?.(e);
            setHasValue(Boolean(e.target.value));
        }
    }, [
        onChange,
        onClear,
        sendAnalyticsEventOncePerFocus
    ]);
    const handleFocus = useCallback((e)=>{
        resetSendAnalyticsEventOnFocus();
        onFocus?.(e);
    }, [
        onFocus,
        resetSendAnalyticsEventOnFocus
    ]);
    const handleWheel = useCallback((e)=>{
        /*
        For additional context, read CJ-72625, PR 1693195 and https://github.com/facebook/react/issues/32156

        The code below fixes an issue when the following conditions are met:
          - An input with type = "number" is focused
          - The mouse is inside the input
          - The user scrolls the mouse wheel

        This can result in unexpected behavior where the input value changes, even though the user might intend
        to scroll on the page.

        React adds a wheel event listener to the input element. On Chrome, when an input with type = "number"
        has a wheel event listener attached, when receiving a WheelEvent, it will also update the input value.
        On Firefox, the input value is not updated.

        The fix here is to blur the input when all the above conditions are met, preventing the input value
        from being changed. See https://stackoverflow.com/questions/63224459/disable-scrolling-on-input-type-number-in-react.
      */ if (props.type === 'number' && !onWheel && disableNumberInputWheelValueChange) {
            e.currentTarget.blur();
            return;
        }
        onWheel?.(e);
    }, [
        disableNumberInputWheelValueChange,
        onWheel,
        props.type
    ]);
    const onFormSubmit = useCallback((e)=>{
        if (!formContext.formRef?.current) return;
        e.preventDefault();
        formContext.formRef.current.requestSubmit();
    }, [
        formContext.formRef
    ]);
    const submitHandlers = useCallbackOnEnter({
        callback: onFormSubmit,
        allowBasicEnter: !shouldPreventFormSubmission,
        allowPlatformEnter: !shouldPreventFormSubmission
    });
    const onPressEnterChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onKeyDown,
                onPressEnter
            ],
            stopOnDefaultPrevented: false
        }), [
        submitHandlers.onKeyDown,
        onPressEnter
    ]));
    const onCompositionStartChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onCompositionStart,
                onCompositionStart
            ]
        }), [
        submitHandlers.onCompositionStart,
        onCompositionStart
    ]));
    const onCompositionEndChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onCompositionEnd,
                onCompositionEnd
            ]
        }), [
        submitHandlers.onCompositionEnd,
        onCompositionEnd
    ]));
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Input$2, {
            ...addDebugOutlineIfEnabled(),
            autoComplete: autoComplete,
            "data-validation": validationState,
            ref: ref,
            css: [
                getInputEmotionStyles(classNamePrefix, theme, {
                    validationState,
                    type: props.type,
                    hasValue,
                    locked,
                    size
                }),
                dangerouslyAppendEmotionCSS
            ],
            onChange: handleChange,
            onFocus: handleFocus,
            onPressEnter: onPressEnterChain,
            onWheel: handleWheel,
            onCompositionStart: onCompositionStartChain,
            onCompositionEnd: onCompositionEndChain,
            ...props,
            readOnly: locked || readOnly,
            suffix: locked ? /*#__PURE__*/ jsx(LockIcon, {}) : props.suffix,
            ...dangerouslySetAntdProps,
            ...eventContext.dataComponentProps
        })
    });
});

const Password = /*#__PURE__*/ forwardRef(function Password({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, ...props }, ref) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Input$2.Password, {
            ...addDebugOutlineIfEnabled(),
            visibilityToggle: false,
            ref: ref,
            autoComplete: autoComplete,
            css: [
                getInputEmotionStyles(classNamePrefix, theme, {
                    validationState
                }),
                dangerouslyAppendEmotionCSS
            ],
            ...props,
            ...dangerouslySetAntdProps
        })
    });
});

const TextArea = /*#__PURE__*/ forwardRef(function TextArea({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, componentId, analyticsEvents, onChange, onFocus, onKeyDown, onCompositionStart, onCompositionEnd, allowFormSubmitOnEnter = false, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.textArea', false);
    const formContext = useFormContext();
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TextArea,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: false
    });
    const firstView = useRef(true);
    useEffect(()=>{
        if (firstView.current) {
            eventContext.onView();
            firstView.current = false;
        }
    }, [
        eventContext
    ]);
    // Prevents multiple onValueChange triggers until after a focus event resets it.
    const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } = useCallbackOnceUntilReset(eventContext.onValueChange);
    const handleChange = useCallback((e)=>{
        sendAnalyticsEventOncePerFocus();
        onChange?.(e);
    }, [
        onChange,
        sendAnalyticsEventOncePerFocus
    ]);
    const handleFocus = useCallback((e)=>{
        resetSendAnalyticsEventOnFocus();
        onFocus?.(e);
    }, [
        onFocus,
        resetSendAnalyticsEventOnFocus
    ]);
    // Callback used to submit the parent form
    // This is used when allowFormSubmitOnEnter is true
    const handleSubmitForm = useCallback((event)=>{
        if (!formContext.formRef?.current) return;
        event.preventDefault();
        formContext.formRef.current.requestSubmit();
    }, [
        formContext.formRef
    ]);
    // Event handlers for submitting the form on Enter
    // This hook takes meta keys into account for platform-specific behavior
    const submitHandlers = useCallbackOnEnter({
        callback: handleSubmitForm,
        allowBasicEnter: allowFormSubmitOnEnter,
        allowPlatformEnter: true
    });
    // Chains the event handlers from useCallbackOnEnter together with potential event handlers from props
    const onKeyDownChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onKeyDown,
                onKeyDown
            ],
            stopOnDefaultPrevented: true
        }), [
        onKeyDown,
        submitHandlers.onKeyDown
    ]));
    const onCompositionStartChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onCompositionStart,
                onCompositionStart
            ]
        }), [
        submitHandlers.onCompositionStart,
        onCompositionStart
    ]));
    const onCompositionEndChain = useChainEventHandlers(useMemo(()=>({
            handlers: [
                submitHandlers.onCompositionEnd,
                onCompositionEnd
            ]
        }), [
        submitHandlers.onCompositionEnd,
        onCompositionEnd
    ]));
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Input$2.TextArea, {
            ...addDebugOutlineIfEnabled(),
            ref: ref,
            autoComplete: autoComplete,
            css: [
                getInputEmotionStyles(classNamePrefix, theme, {
                    validationState
                }),
                dangerouslyAppendEmotionCSS
            ],
            onChange: handleChange,
            onFocus: handleFocus,
            onKeyDown: onKeyDownChain,
            onCompositionStart: onCompositionStartChain,
            onCompositionEnd: onCompositionEndChain,
            "data-component-type": DesignSystemEventProviderComponentTypes.TextArea,
            "data-component-id": componentId,
            ...props,
            ...dangerouslySetAntdProps
        })
    });
});

// Properly creates the namespace and dot-notation components with correct types.
const InputNamespace = /* #__PURE__ */ Object.assign(Input$1, {
    TextArea,
    Password,
    Group
});
const Input = InputNamespace;

export { Input as I, useCallbackOnEnter as a, getInputStyles as g, useCallbackOnceUntilReset as u };
//# sourceMappingURL=index-B714nuEE.js.map
