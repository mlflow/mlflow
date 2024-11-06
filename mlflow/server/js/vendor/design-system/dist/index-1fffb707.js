import React__default, { forwardRef, useRef, useCallback, useMemo } from 'react';
import { I as Icon, a as useDesignSystemTheme, D as DesignSystemAntDConfigProvider, h as addDebugOutlineIfEnabled, t as getValidationStateColor, g as getAnimationCss, b as DesignSystemEventProviderAnalyticsEventTypes, s as safex, c as useDesignSystemEventComponentCallbacks, d as DesignSystemEventProviderComponentTypes, q as importantify, a7 as useFormContext } from './Typography-24f4450b.js';
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
var ClockIcon$1 = ClockIcon;

function SvgMegaphoneIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 18 18",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M16.25 2a.75.75 0 0 0-1.248-.56l-4.287 3.81H4A2.75 2.75 0 0 0 1.25 8v2A2.75 2.75 0 0 0 4 12.75h1.75V16a.75.75 0 0 0 1.5 0v-3.25h3.465l4.287 3.81A.75.75 0 0 0 16.25 16zm-4.752 4.56 3.252-2.89v10.66l-3.252-2.89a.75.75 0 0 0-.498-.19H4c-.69 0-1.25-.56-1.25-1.25V8c0-.69.56-1.25 1.25-1.25h7a.75.75 0 0 0 .498-.19",
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
var MegaphoneIcon$1 = MegaphoneIcon;

const getInputGroupStyling = (clsPrefix, theme) => {
  const inputClass = `.${clsPrefix}-input`;
  const buttonClass = `.${clsPrefix}-btn`;
  return /*#__PURE__*/css({
    display: 'inline-flex !important',
    width: 'auto',
    [`& > ${inputClass}`]: {
      flexGrow: 1,
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
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.Group, {
      ...addDebugOutlineIfEnabled(),
      css: [getInputGroupStyling(classNamePrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Group;"],
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

const getInputStyles = (clsPrefix, theme, _ref, _ref2) => {
  let {
    validationState,
    type,
    hasValue
  } = _ref;
  let {
    useTransparent = false,
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
        outlineColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent'
      },
      '&:focus-visible': {
        outlineColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent'
      }
    },
    [`&${inputClass}, ${inputClass}`]: {
      ...(useTransparent && {
        backgroundColor: 'transparent'
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
      ...(useTransparent && {
        backgroundColor: 'transparent'
      }),
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
        outlineColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundDefault,
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
const getInputEmotionStyles = (clsPrefix, theme, _ref3, useTransparent) => {
  let {
    validationState,
    type,
    hasValue
  } = _ref3;
  const styles = getInputStyles(clsPrefix, theme, {
    validationState,
    type,
    hasValue
  }, {
    useTransparent
  });
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
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    ...props
  } = _ref4;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const [hasValue, setHasValue] = React__default.useState(props.value !== undefined && props.value !== null && props.value !== '');
  const useTransparent = safex('databricks.fe.designsystem.useTransparentInput', false);
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
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2, {
      ...addDebugOutlineIfEnabled(),
      autoComplete: autoComplete,
      "data-validation": validationState,
      ref: ref,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState,
        type: props.type,
        hasValue
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Input;"],
      onChange: handleChange,
      onFocus: handleFocus,
      ...props,
      ...dangerouslySetAntdProps,
      "data-component-type": DesignSystemEventProviderComponentTypes.Input,
      "data-component-id": componentId
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
  const useTransparent = safex('databricks.fe.designsystem.useTransparentInput', false);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.Password, {
      ...addDebugOutlineIfEnabled(),
      visibilityToggle: false,
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Password;"],
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
  const isMacOs = useMemo(() => navigator.userAgent.includes('Mac'), []);
  const isCompositing = useRef(false);
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const useTransparent = safex('databricks.fe.designsystem.useTransparentInput', false);
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
  const onKeyDownWrapper = useCallback(event => {
    var _formContext$formRef;
    // Trigger a form submission if its within a form context and the enter combination is pressed and is not in composition mode.
    const basicEnter = event.key === 'Enter' && allowFormSubmitOnEnter && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;
    const platformEnter = event.key === 'Enter' && (isMacOs ? event.metaKey : event.ctrlKey);
    if ((_formContext$formRef = formContext.formRef) !== null && _formContext$formRef !== void 0 && _formContext$formRef.current && (basicEnter || platformEnter) && !isCompositing.current) {
      event.preventDefault();
      formContext.formRef.current.requestSubmit();
      return;
    }
    onKeyDown === null || onKeyDown === void 0 || onKeyDown(event);
  }, [allowFormSubmitOnEnter, formContext.formRef, isMacOs, onKeyDown]);
  const onCompositionStartWrapper = useCallback(event => {
    isCompositing.current = true;
    onCompositionStart === null || onCompositionStart === void 0 || onCompositionStart(event);
  }, [onCompositionStart]);
  const onCompositionEndWrapper = useCallback(event => {
    isCompositing.current = false;
    onCompositionEnd === null || onCompositionEnd === void 0 || onCompositionEnd(event);
  }, [onCompositionEnd]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Input$2.TextArea, {
      ...addDebugOutlineIfEnabled(),
      ref: ref,
      autoComplete: autoComplete,
      css: [getInputEmotionStyles(classNamePrefix, theme, {
        validationState
      }, useTransparent), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:TextArea;"],
      onChange: handleChange,
      onFocus: handleFocus,
      onKeyDown: onKeyDownWrapper,
      onCompositionStart: onCompositionStartWrapper,
      onCompositionEnd: onCompositionEndWrapper,
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

export { ClockIcon$1 as C, Input as I, MegaphoneIcon$1 as M, getInputStyles as g };
//# sourceMappingURL=index-1fffb707.js.map
