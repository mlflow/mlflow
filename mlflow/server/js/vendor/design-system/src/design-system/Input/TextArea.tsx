import { Input as AntDInput } from 'antd';
import type { TextAreaRef } from 'antd/lib/input/TextArea';
import { forwardRef, useCallback, useMemo } from 'react';

import { getInputEmotionStyles } from './Input';
import type { TextAreaProps } from './common';
import { useFormContext } from '../../development/Form';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useCallbackOnEnter } from '../utils/useCallbackOnEnter';
import { useCallbackOnceUntilReset } from '../utils/useCallbackOnceUntilReset';
import { useChainEventHandlers } from '../utils/useChainEventHandlers';

export const TextArea = forwardRef<TextAreaRef, TextAreaProps>(function TextArea(
  {
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
  },
  ref,
) {
  const formContext = useFormContext();
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TextArea,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: false,
  });

  // Prevents multiple onValueChange triggers until after a focus event resets it.
  const { callbackOnceUntilReset: sendAnalyticsEventOncePerFocus, reset: resetSendAnalyticsEventOnFocus } =
    useCallbackOnceUntilReset(eventContext.onValueChange);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      sendAnalyticsEventOncePerFocus();
      onChange?.(e);
    },
    [onChange, sendAnalyticsEventOncePerFocus],
  );

  const handleFocus = useCallback(
    (e: React.FocusEvent<HTMLTextAreaElement>) => {
      resetSendAnalyticsEventOnFocus();
      onFocus?.(e);
    },
    [onFocus, resetSendAnalyticsEventOnFocus],
  );

  // Callback used to submit the parent form
  // This is used when allowFormSubmitOnEnter is true
  const handleSubmitForm = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (!formContext.formRef?.current) return;

      event.preventDefault();
      formContext.formRef.current.requestSubmit();
    },
    [formContext.formRef],
  );
  // Event handlers for submitting the form on Enter
  // This hook takes meta keys into account for platform-specific behavior
  const submitHandlers = useCallbackOnEnter({
    callback: handleSubmitForm,
    allowBasicEnter: allowFormSubmitOnEnter, // Conditionally allow form submission on Enter
    allowPlatformEnter: true, // Always allow form submission on platform Enter
  });

  // Chains the event handlers from useCallbackOnEnter together with potential event handlers from props
  const onKeyDownChain = useChainEventHandlers(
    useMemo(
      () => ({
        handlers: [submitHandlers.onKeyDown, onKeyDown],
        stopOnDefaultPrevented: true, // If the form is submitted, do not call the next handler
      }),
      [onKeyDown, submitHandlers.onKeyDown],
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
      <AntDInput.TextArea
        {...addDebugOutlineIfEnabled()}
        ref={ref}
        autoComplete={autoComplete}
        css={[
          getInputEmotionStyles(classNamePrefix, theme, { validationState, useNewShadows }),
          dangerouslyAppendEmotionCSS,
        ]}
        onChange={handleChange}
        onFocus={handleFocus}
        onKeyDown={onKeyDownChain}
        onCompositionStart={onCompositionStartChain}
        onCompositionEnd={onCompositionEndChain}
        data-component-type={DesignSystemEventProviderComponentTypes.TextArea}
        data-component-id={componentId}
        {...props}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
});
