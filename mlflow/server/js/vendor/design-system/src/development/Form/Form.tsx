import type { FormEvent } from 'react';
import React, { useCallback, useMemo, useRef, useState } from 'react';

import type { ReferrerComponentType } from '../../design-system';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../../design-system/DesignSystemEventProvider';
import { LoadingState } from '../../design-system/LoadingState';
import type { AnalyticsEventPropsWithStartInteraction } from '../../design-system/types';

const FormContext = React.createContext<{
  componentId?: string;
  isSubmitting: boolean;
  formRef?: React.MutableRefObject<HTMLFormElement | null>;
}>({ componentId: undefined, isSubmitting: false, formRef: undefined });

export const useFormContext = () => React.useContext(FormContext);

export type FormProps = {
  onSubmit: (event: FormEvent) => Promise<void>;
} & AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnSubmit> &
  Omit<React.ComponentProps<'form'>, 'onSubmit'>;

/**
 * Form is a wrapper around the form element that allows us to track the active element before the form is submitted.
 * This is useful for analytics purposes to know what component was last focused when the form was sent for submission.
 *
 * NOTE: Form component cannot be nested.
 */
export const Form = ({
  children,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnSubmit],
  onSubmit,
  shouldStartInteraction,
  ...otherProps
}: FormProps) => {
  const formContext = useFormContext();
  const formRef = useRef<HTMLFormElement | null>(null);
  const eventRef = useRef<React.UIEvent | null>(null);
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

  const onSubmitHandler = useCallback<React.FormEventHandler<HTMLFormElement>>(
    async (e) => {
      try {
        setIsSubmitting(true);
        eventContext.onSubmit(e, extractReferrerComponent());
        await onSubmit(e);
      } finally {
        setIsSubmitting(false);
      }
    },
    [eventContext, onSubmit],
  );
  const contextValue = useMemo(() => ({ componentId, isSubmitting, eventRef, formRef }), [componentId, isSubmitting]);

  return (
    <FormContext.Provider value={contextValue}>
      <form onSubmit={onSubmitHandler} {...otherProps} ref={formRef} {...eventContext.dataComponentProps}>
        {isSubmitting && <LoadingState description={componentId} />}
        {children}
      </form>
    </FormContext.Provider>
  );
};

function extractReferrerComponent(): ReferrerComponentType | undefined {
  const activeElement = document.activeElement as HTMLOrSVGElement | null;
  const { componentType, componentId } = activeElement?.dataset ?? {};
  if (componentType && componentId) {
    return {
      type: componentType as DesignSystemEventProviderComponentTypes,
      id: componentId,
    };
  }

  return undefined;
}
