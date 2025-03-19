import type { FormEvent } from 'react';
import React from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system/DesignSystemEventProvider';
import type { AnalyticsEventPropsWithStartInteraction } from '../../design-system/types';
export declare const useFormContext: () => {
    componentId?: string;
    isSubmitting: boolean;
    formRef?: React.MutableRefObject<HTMLFormElement | null>;
};
export type FormProps = {
    onSubmit: (event: FormEvent) => Promise<void>;
} & AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnSubmit> & Omit<React.ComponentProps<'form'>, 'onSubmit'>;
/**
 * Form is a wrapper around the form element that allows us to track the active element before the form is submitted.
 * This is useful for analytics purposes to know what component was last focused when the form was sent for submission.
 *
 * NOTE: Form component cannot be nested.
 */
export declare const Form: ({ children, componentId, analyticsEvents, onSubmit, shouldStartInteraction, ...otherProps }: FormProps) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Form.d.ts.map