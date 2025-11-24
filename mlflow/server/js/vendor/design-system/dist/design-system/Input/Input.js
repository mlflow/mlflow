import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import React, { forwardRef, useCallback, useEffect, useMemo, useRef } from 'react';
import { useFormContext } from '../../development/Form/Form';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { LockIcon } from '../Icon';
import { useDesignSystemSafexFlags } from '../utils';
import { getValidationStateColor, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useCallbackOnEnter } from '../utils/useCallbackOnEnter';
import { useCallbackOnceUntilReset } from '../utils/useCallbackOnceUntilReset';
import { useChainEventHandlers } from '../utils/useChainEventHandlers';
export const getInputStyles = (clsPrefix, theme, { validationState, type, hasValue, useNewShadows, useNewFormUISpacing, useNewBorderRadii, locked, }, { useFocusWithin = false }) => {
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
            ...(validationState && { borderColor: validationColor }),
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover,
            },
            [`&:${focusSpecifier}`]: {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                ...(!useNewShadows && {
                    boxShadow: 'none',
                }),
                borderColor: 'transparent',
            },
            '&:focus-visible': {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                ...(!useNewShadows && {
                    boxShadow: 'none',
                }),
                borderColor: 'transparent',
            },
            ...(useNewFormUISpacing && {
                [`& + .${clsPrefix}-form-message`]: {
                    marginTop: theme.spacing.sm,
                },
            }),
        },
        [`&${inputClass}, ${inputClass}`]: {
            backgroundColor: 'transparent',
            ...(useNewBorderRadii && {
                borderRadius: theme.borders.borderRadiusSm,
            }),
            '&:disabled': {
                backgroundColor: theme.colors.actionDisabledBackground,
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder,
            },
            '&::placeholder': {
                color: theme.colors.textPlaceholder,
            },
        },
        [`&${affixClass}`]: {
            backgroundColor: 'transparent',
            lineHeight: theme.typography.lineHeightBase,
            paddingTop: 5,
            paddingBottom: 5,
            minHeight: theme.general.heightSm,
            '::before': {
                lineHeight: theme.typography.lineHeightBase,
            },
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover,
            },
            [`input.${clsPrefix}-input`]: {
                borderRadius: 0,
            },
        },
        [`&${affixClassDisabled}`]: {
            backgroundColor: theme.colors.actionDisabledBackground,
        },
        [`&${affixClassFocused}`]: {
            boxShadow: 'none',
            [`&&, &:${focusSpecifier}`]: {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                boxShadow: 'none',
                borderColor: 'transparent',
            },
        },
        [clearIcon]: {
            fontSize: theme.typography.fontSizeSm,
        },
        [prefixIcon]: {
            marginRight: theme.spacing.sm,
            color: theme.colors.textSecondary,
        },
        [suffixIcon]: {
            marginLeft: theme.spacing.sm,
            color: theme.colors.textSecondary,
            ...(!hasValue &&
                type === 'number' && {
                display: 'none',
            }),
        },
        ...(locked
            ? {
                [`&${affixClass}`]: {
                    backgroundColor: theme.colors.backgroundSecondary,
                },
                [`input.${clsPrefix}-input`]: {
                    backgroundColor: `${theme.colors.backgroundSecondary} !important`,
                    color: `${theme.colors.textPrimary} !important`,
                },
            }
            : {}),
        ...getAnimationCss(theme.options.enableAnimation),
    };
    return styles;
};
export const getInputEmotionStyles = (clsPrefix, theme, { validationState, type, hasValue, useNewShadows, useNewBorderRadii, locked, }) => {
    const styles = getInputStyles(clsPrefix, theme, { validationState, type, hasValue, useNewShadows, useNewBorderRadii, locked }, {});
    return css(importantify(styles));
};
export const Input = forwardRef(function Input({ validationState, autoComplete = 'off', dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, onChange, onClear, onFocus, onPressEnter, onCompositionStart, onCompositionEnd, componentId, shouldPreventFormSubmission, analyticsEvents, readOnly, locked, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.input', false);
    const formContext = useFormContext();
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const [hasValue, setHasValue] = React.useState(props.value !== undefined && props.value !== null && props.value !== '');
    const { useNewShadows, useNewBorderRadii } = useDesignSystemSafexFlags();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Input,
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
        // If the input is cleared, call the onClear handler, but only
        // if the event is not an input event -- which is the case when you click the
        // ant-provided (X) button.
        if (!e.target.value && e.nativeEvent instanceof InputEvent === false && onClear) {
            onClear?.();
            setHasValue(false);
        }
        else {
            onChange?.(e);
            setHasValue(Boolean(e.target.value));
        }
    }, [onChange, onClear, sendAnalyticsEventOncePerFocus]);
    const handleFocus = useCallback((e) => {
        resetSendAnalyticsEventOnFocus();
        onFocus?.(e);
    }, [onFocus, resetSendAnalyticsEventOnFocus]);
    const onFormSubmit = useCallback((e) => {
        if (!formContext.formRef?.current)
            return;
        e.preventDefault();
        formContext.formRef.current.requestSubmit();
    }, [formContext.formRef]);
    const submitHandlers = useCallbackOnEnter({
        callback: onFormSubmit,
        allowBasicEnter: !shouldPreventFormSubmission,
        allowPlatformEnter: !shouldPreventFormSubmission,
    });
    const onPressEnterChain = useChainEventHandlers(useMemo(() => ({ handlers: [submitHandlers.onKeyDown, onPressEnter], stopOnDefaultPrevented: false }), [submitHandlers.onKeyDown, onPressEnter]));
    const onCompositionStartChain = useChainEventHandlers(useMemo(() => ({ handlers: [submitHandlers.onCompositionStart, onCompositionStart] }), [submitHandlers.onCompositionStart, onCompositionStart]));
    const onCompositionEndChain = useChainEventHandlers(useMemo(() => ({ handlers: [submitHandlers.onCompositionEnd, onCompositionEnd] }), [submitHandlers.onCompositionEnd, onCompositionEnd]));
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDInput, { ...addDebugOutlineIfEnabled(), autoComplete: autoComplete, "data-validation": validationState, ref: ref, css: [
                getInputEmotionStyles(classNamePrefix, theme, {
                    validationState,
                    type: props.type,
                    hasValue,
                    useNewShadows,
                    useNewBorderRadii,
                    locked,
                }),
                dangerouslyAppendEmotionCSS,
            ], onChange: handleChange, onFocus: handleFocus, onPressEnter: onPressEnterChain, onCompositionStart: onCompositionStartChain, onCompositionEnd: onCompositionEndChain, ...props, readOnly: locked || readOnly, suffix: locked ? _jsx(LockIcon, {}) : props.suffix, ...dangerouslySetAntdProps, ...eventContext.dataComponentProps }) }));
});
//# sourceMappingURL=Input.js.map