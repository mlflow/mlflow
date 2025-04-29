import React__default, { forwardRef, useMemo, useRef, useCallback } from 'react';
import { I as Icon, u as useDesignSystemTheme, a as useDesignSystemSafexFlags, D as DesignSystemAntDConfigProvider, b as addDebugOutlineIfEnabled, g as getValidationStateColor, c as getAnimationCss, d as DesignSystemEventProviderAnalyticsEventTypes, e as useFormContext, f as useDesignSystemEventComponentCallbacks, h as DesignSystemEventProviderComponentTypes, i as importantify } from './Stepper-ae2Kr5tN.js';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';
import { css } from '@emotion/react';
import { Input as Input$2 } from 'antd';

function SvgClockIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 4v4c0 .199.079.39.22.53l2 2 1.06-1.06-1.78-1.78V4z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
      clipRule: "evenodd"
    })]
  });
}
const ClockIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgClockIcon
  });
});
ClockIcon.displayName = 'ClockIcon';

function SvgMegaphoneIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12.197 1.243A.75.75 0 0 1 12.75 1h1.5a.75.75 0 0 1 .75.75v10.5a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1-.553-.243l-.892-.973A5.5 5.5 0 0 0 8 10.051V13a2 2 0 1 1-4 0v-3a3 3 0 0 1 0-6h3.25a5.5 5.5 0 0 0 4.055-1.784zM6.5 8.5v-3H4a1.5 1.5 0 1 0 0 3zm-1 1.5v3a.5.5 0 0 0 1 0v-3zm6.911.77A7 7 0 0 0 8 8.54V5.46a7 7 0 0 0 4.411-2.23l.669-.73h.42v9h-.42z",
      clipRule: "evenodd"
    })
  });
}
const MegaphoneIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMegaphoneIcon
  });
});
MegaphoneIcon.displayName = 'MegaphoneIcon';

function SvgSearchIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#SearchIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SearchIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSearchIcon
  });
});
SearchIcon.displayName = 'SearchIcon';

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
 */
const useCallbackOnEnter = _ref => {
  let {
    callback,
    allowBasicEnter,
    allowPlatformEnter
  } = _ref;
  const isMacOs = useMemo(() => navigator.userAgent.includes('Mac'), []);
  // Keeping track of whether we are composing characters
  // This is stored in a ref so that it can be accessed in the onKeyDown event handler
  // without causing a re-renders
  const isComposing = useRef(false);

  // Handler for when the composition starts
  const onCompositionStart = useCallback(() => {
    isComposing.current = true;
  }, []);

  // Handler for when the composition ends
  const onCompositionEnd = useCallback(() => {
    isComposing.current = false;
  }, []);

  // Handler for when a key is pressed
  // Used to submit the form when Enter is pressed
  const onKeyDown = useCallback(event => {
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
  }, [allowBasicEnter, allowPlatformEnter, callback, isMacOs]);
  return {
    onKeyDown,
    onCompositionEnd,
    onCompositionStart
  };
};

const getInputGroupStyling = (clsPrefix, theme, useNewShadows, useNewBorderRadii) => {
  const inputClass = `.${clsPrefix}-input`;
  const buttonClass = `.${clsPrefix}-btn`;
  return /*#__PURE__*/css({
    display: 'inline-flex !important',
    width: 'auto',
    [`& > ${inputClass}`]: {
      flexGrow: 1,
      ...(useNewBorderRadii && {
        borderTopRightRadius: '0px !important',
        borderBottomRightRadius: '0px !important'
      }),
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
    ...(useNewShadows && {
      [`& > ${buttonClass}`]: {
        boxShadow: 'none !important'
      }
    }),
    ...(useNewBorderRadii && {
      [`& > ${buttonClass}`]: {
        borderTopLeftRadius: '0px !important',
        borderBottomLeftRadius: '0px !important'
      }
    }),
    [`& > ${buttonClass} > span`]: {
      verticalAlign: 'middle'
    },
    [`& > ${buttonClass}:disabled, & > ${buttonClass}:disabled:hover`]: {
      borderLeft: `1px solid ${theme.colors.actionDisabledBorder} !important`,
      backgroundColor: `${theme.colors.actionDisabledBackground} !important`,
      color: `${theme.colors.actionDisabledText} !important`
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getInputGroupStyling;");
};
const Group = _ref => {
  let {
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    compact = true,
    ...props
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    useNewShadows,
    useNewBorderRadii
  } = useDesignSystemSafexFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.Group, {
      ...addDebugOutlineIfEnabled(),
      css: [getInputGroupStyling(classNamePrefix, theme, useNewShadows, useNewBorderRadii), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Group;"],
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
 */
const useCallbackOnceUntilReset = callback => {
  const canTriggerRef = useRef(true);
  const reset = useCallback(() => {
    canTriggerRef.current = true;
  }, []);
  const callbackOnceUntilReset = useCallback(() => {
    if (canTriggerRef.current) {
      callback();
      canTriggerRef.current = false;
    }
  }, [callback]);
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
 */
const useChainEventHandlers = props => {
  const {
    handlers,
    stopOnDefaultPrevented
  } = props;
  return useCallback(event => {
    // Loop over each handler in succession
    for (const handler of handlers) {
      // Break if the event has been defaultPrevented and stopOnDefaultPrevented is true
      if (stopOnDefaultPrevented && event.defaultPrevented) return;
      // Call the handler if it exists
      handler === null || handler === void 0 || handler(event);
    }
  }, [handlers, stopOnDefaultPrevented]);
};

const getInputStyles = (clsPrefix, theme, _ref, _ref2) => {
  let {
    validationState,
    type,
    hasValue,
    useNewShadows,
    useNewFormUISpacing,
    useNewBorderRadii
  } = _ref;
  let {
    useFocusWithin = false
  } = _ref2;
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
      lineHeight: theme.typography.lineHeightBase,
      minHeight: theme.general.heightSm,
      borderColor: theme.colors.actionDefaultBorderDefault,
      ...(validationState && {
        borderColor: validationColor
      }),
      '&:hover': {
        borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
      },
      [`&:${focusSpecifier}`]: {
        outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        ...(!useNewShadows && {
          boxShadow: 'none'
        }),
        borderColor: 'transparent'
      },
      '&:focus-visible': {
        outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        ...(!useNewShadows && {
          boxShadow: 'none'
        }),
        borderColor: 'transparent'
      },
      ...(useNewFormUISpacing && {
        [`& + .${clsPrefix}-form-message`]: {
          marginTop: theme.spacing.sm
        }
      })
    },
    [`&${inputClass}, ${inputClass}`]: {
      backgroundColor: 'transparent',
      ...(useNewBorderRadii && {
        borderRadius: theme.borders.borderRadiusSm
      }),
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
      lineHeight: theme.typography.lineHeightBase,
      paddingTop: 5,
      paddingBottom: 5,
      minHeight: theme.general.heightSm,
      '::before': {
        lineHeight: theme.typography.lineHeightBase
      },
      '&:hover': {
        borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
      },
      [`input.${clsPrefix}-input`]: {
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
      ...(!hasValue && type === 'number' && {
        display: 'none'
      })
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return styles;
};
const getInputEmotionStyles = (clsPrefix, theme, _ref3) => {
  let {
    validationState,
    type,
    hasValue,
    useNewShadows,
    useNewBorderRadii
  } = _ref3;
  const styles = getInputStyles(clsPrefix, theme, {
    validationState,
    type,
    hasValue,
    useNewShadows,
    useNewBorderRadii
  }, {});
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getInputEmotionStyles;");
};
const Input$1 = /*#__PURE__*/forwardRef(function Input(_ref4, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    onChange,
    onClear,
    onFocus,
    onPressEnter,
    onCompositionStart,
    onCompositionEnd,
    componentId,
    shouldPreventFormSubmission,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    ...props
  } = _ref4;
  const formContext = useFormContext();
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const [hasValue, setHasValue] = React__default.useState(props.value !== undefined && props.value !== null && props.value !== '');
  const {
    useNewShadows,
    useNewBorderRadii
  } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Input,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: false
  });

  // Prevents multiple onValueChange triggers until after a focus event resets it.
  const {
    callbackOnceUntilReset: sendAnalyticsEventOncePerFocus,
    reset: resetSendAnalyticsEventOnFocus
  } = useCallbackOnceUntilReset(eventContext.onValueChange);
  const handleChange = useCallback(e => {
    sendAnalyticsEventOncePerFocus();
    // If the input is cleared, call the onClear handler, but only
    // if the event is not an input event -- which is the case when you click the
    // ant-provided (X) button.
    if (!e.target.value && e.nativeEvent instanceof InputEvent === false && onClear) {
      onClear === null || onClear === void 0 || onClear();
      setHasValue(false);
    } else {
      onChange === null || onChange === void 0 || onChange(e);
      setHasValue(Boolean(e.target.value));
    }
  }, [onChange, onClear, sendAnalyticsEventOncePerFocus]);
  const handleFocus = useCallback(e => {
    resetSendAnalyticsEventOnFocus();
    onFocus === null || onFocus === void 0 || onFocus(e);
  }, [onFocus, resetSendAnalyticsEventOnFocus]);
  const onFormSubmit = useCallback(e => {
    var _formContext$formRef;
    if (!((_formContext$formRef = formContext.formRef) !== null && _formContext$formRef !== void 0 && _formContext$formRef.current)) return;
    e.preventDefault();
    formContext.formRef.current.requestSubmit();
  }, [formContext.formRef]);
  const submitHandlers = useCallbackOnEnter({
    callback: onFormSubmit,
    allowBasicEnter: !shouldPreventFormSubmission,
    allowPlatformEnter: !shouldPreventFormSubmission
  });
  const onPressEnterChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onKeyDown, onPressEnter],
    stopOnDefaultPrevented: false
  }), [submitHandlers.onKeyDown, onPressEnter]));
  const onCompositionStartChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onCompositionStart, onCompositionStart]
  }), [submitHandlers.onCompositionStart, onCompositionStart]));
  const onCompositionEndChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onCompositionEnd, onCompositionEnd]
  }), [submitHandlers.onCompositionEnd, onCompositionEnd]));
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2, {
      ...addDebugOutlineIfEnabled(),
      autoComplete: autoComplete,
      "data-validation": validationState,
      ref: ref,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState,
        type: props.type,
        hasValue,
        useNewShadows,
        useNewBorderRadii
      }), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Input;"],
      onChange: handleChange,
      onFocus: handleFocus,
      onPressEnter: onPressEnterChain,
      onCompositionStart: onCompositionStartChain,
      onCompositionEnd: onCompositionEndChain,
      ...props,
      ...dangerouslySetAntdProps,
      ...eventContext.dataComponentProps
    })
  });
});

const Password = /*#__PURE__*/forwardRef(function Password(_ref, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    ...props
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    useNewShadows,
    useNewBorderRadii
  } = useDesignSystemSafexFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.Password, {
      ...addDebugOutlineIfEnabled(),
      visibilityToggle: false,
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState,
        useNewShadows,
        useNewBorderRadii
      }), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Password;"],
      ...props,
      ...dangerouslySetAntdProps
    })
  });
});

const TextArea = /*#__PURE__*/forwardRef(function TextArea(_ref, ref) {
  let {
    validationState,
    autoComplete = 'off',
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    onChange,
    onFocus,
    onKeyDown,
    onCompositionStart,
    onCompositionEnd,
    allowFormSubmitOnEnter = false,
    ...props
  } = _ref;
  const formContext = useFormContext();
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    useNewShadows,
    useNewBorderRadii
  } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TextArea,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: false
  });

  // Prevents multiple onValueChange triggers until after a focus event resets it.
  const {
    callbackOnceUntilReset: sendAnalyticsEventOncePerFocus,
    reset: resetSendAnalyticsEventOnFocus
  } = useCallbackOnceUntilReset(eventContext.onValueChange);
  const handleChange = useCallback(e => {
    sendAnalyticsEventOncePerFocus();
    onChange === null || onChange === void 0 || onChange(e);
  }, [onChange, sendAnalyticsEventOncePerFocus]);
  const handleFocus = useCallback(e => {
    resetSendAnalyticsEventOnFocus();
    onFocus === null || onFocus === void 0 || onFocus(e);
  }, [onFocus, resetSendAnalyticsEventOnFocus]);

  // Callback used to submit the parent form
  // This is used when allowFormSubmitOnEnter is true
  const handleSubmitForm = useCallback(event => {
    var _formContext$formRef;
    if (!((_formContext$formRef = formContext.formRef) !== null && _formContext$formRef !== void 0 && _formContext$formRef.current)) return;
    event.preventDefault();
    formContext.formRef.current.requestSubmit();
  }, [formContext.formRef]);
  // Event handlers for submitting the form on Enter
  // This hook takes meta keys into account for platform-specific behavior
  const submitHandlers = useCallbackOnEnter({
    callback: handleSubmitForm,
    allowBasicEnter: allowFormSubmitOnEnter,
    // Conditionally allow form submission on Enter
    allowPlatformEnter: true // Always allow form submission on platform Enter
  });

  // Chains the event handlers from useCallbackOnEnter together with potential event handlers from props
  const onKeyDownChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onKeyDown, onKeyDown],
    stopOnDefaultPrevented: true // If the form is submitted, do not call the next handler
  }), [onKeyDown, submitHandlers.onKeyDown]));
  const onCompositionStartChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onCompositionStart, onCompositionStart]
  }), [submitHandlers.onCompositionStart, onCompositionStart]));
  const onCompositionEndChain = useChainEventHandlers(useMemo(() => ({
    handlers: [submitHandlers.onCompositionEnd, onCompositionEnd]
  }), [submitHandlers.onCompositionEnd, onCompositionEnd]));
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.TextArea, {
      ...addDebugOutlineIfEnabled(),
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState,
        useNewShadows,
        useNewBorderRadii
      }), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:TextArea;"],
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
const InputNamespace = /* #__PURE__ */Object.assign(Input$1, {
  TextArea,
  Password,
  Group
});
const Input = InputNamespace;

export { ClockIcon as C, Input as I, MegaphoneIcon as M, SearchIcon as S, getInputStyles as g, useCallbackOnEnter as u };
//# sourceMappingURL=index-B32Lva7u.js.map
