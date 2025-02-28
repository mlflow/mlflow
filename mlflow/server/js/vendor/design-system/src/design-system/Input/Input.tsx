import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { Input as AntDInput } from 'antd';
import React, { forwardRef, useCallback, useMemo } from 'react';

import type { InputProps } from './common';
import { useFormContext } from '../../development/Form/Form';
import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import type { ValidationState } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { getValidationStateColor, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useCallbackOnEnter } from '../utils/useCallbackOnEnter';
import { useCallbackOnceUntilReset } from '../utils/useCallbackOnceUntilReset';
import { useChainEventHandlers } from '../utils/useChainEventHandlers';

interface InputStylesOptions {
  useFocusWithin?: boolean;
}

export const getInputStyles = (
  clsPrefix: string,
  theme: Theme,
  {
    validationState,
    type,
    hasValue,
    useNewShadows,
    useNewFormUISpacing,
  }: {
    validationState?: ValidationState;
    type?: string;
    hasValue?: boolean;
    useNewShadows?: boolean;
    useNewFormUISpacing?: boolean;
  },
  { useFocusWithin = false }: InputStylesOptions,
): CSSObject => {
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

    ...getAnimationCss(theme.options.enableAnimation),
  };

  return styles;
};

export const getInputEmotionStyles = (
  clsPrefix: string,
  theme: Theme,
  {
    validationState,
    type,
    hasValue,
    useNewShadows,
  }: { validationState?: ValidationState; type?: string; hasValue?: boolean; useNewShadows: boolean },
): SerializedStyles => {
  const styles = getInputStyles(clsPrefix, theme, { validationState, type, hasValue, useNewShadows }, {});
  return css(importantify(styles));
};

export const Input = forwardRef<AntDInput, InputProps>(function Input(
  {
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
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    ...props
  },
  ref,
) {
  const formContext = useFormContext();
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const [hasValue, setHasValue] = React.useState(
    props.value !== undefined && props.value !== null && props.value !== '',
  );

  const { useNewShadows } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Input,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: false,
  });

  // Prevents multiple onValueChange triggers until after a focus event resets it.
  const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } =
    useCallbackOnceUntilReset(eventContext.onValueChange);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
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
    },
    [onChange, onClear, sendAnalyticsEventOncePerFocus],
  );

  const handleFocus = useCallback(
    (e: React.FocusEvent<HTMLInputElement>) => {
      resetSendAnalyticsEventOnFocus();
      onFocus?.(e);
    },
    [onFocus, resetSendAnalyticsEventOnFocus],
  );

  const onFormSubmit = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (!formContext.formRef?.current) return;
      e.preventDefault();
      formContext.formRef.current.requestSubmit();
    },
    [formContext.formRef],
  );

  const submitHandlers = useCallbackOnEnter<HTMLInputElement>({
    callback: onFormSubmit,
    allowBasicEnter: true,
    allowPlatformEnter: true,
  });

  const onPressEnterChain = useChainEventHandlers(
    useMemo(
      () => ({ handlers: [submitHandlers.onKeyDown, onPressEnter], stopOnDefaultPrevented: false }),
      [submitHandlers.onKeyDown, onPressEnter],
    ),
  );

  const onCompositionStartChain = useChainEventHandlers(
    useMemo(
      () => ({ handlers: [submitHandlers.onCompositionStart, onCompositionStart] }),
      [submitHandlers.onCompositionStart, onCompositionStart],
    ),
  );

  const onCompositionEndChain = useChainEventHandlers(
    useMemo(
      () => ({ handlers: [submitHandlers.onCompositionEnd, onCompositionEnd] }),
      [submitHandlers.onCompositionEnd, onCompositionEnd],
    ),
  );

  return (
    <DesignSystemAntDConfigProvider>
      <AntDInput
        {...addDebugOutlineIfEnabled()}
        autoComplete={autoComplete}
        data-validation={validationState}
        ref={ref}
        css={[
          getInputEmotionStyles(classNamePrefix, theme, { validationState, type: props.type, hasValue, useNewShadows }),
          dangerouslyAppendEmotionCSS,
        ]}
        onChange={handleChange}
        onFocus={handleFocus}
        onPressEnter={onPressEnterChain}
        onCompositionStart={onCompositionStartChain}
        onCompositionEnd={onCompositionEndChain}
        {...props}
        {...dangerouslySetAntdProps}
        {...eventContext.dataComponentProps}
      />
    </DesignSystemAntDConfigProvider>
  );
});
