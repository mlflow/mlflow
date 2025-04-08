import type { FormEvent } from 'react';
import React from 'react';
import type { DeepPartial, FieldValues, UseFormHandleSubmit, SubmitHandler, SubmitErrorHandler } from 'react-hook-form';
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
export type RhfFormProps<T extends FieldValues> = {
    /**
     * @param handleSubmit - The handleSubmit function from react-hook-form
     */
    handleSubmit: UseFormHandleSubmit<T>;
    /**
     * @param handleValid - The onValid callback for handleSubmit from react-hook-form; this is called when the form is
     * valid according to react-hook-form and after the form is submitted
     */
    handleValid: SubmitHandler<T>;
    /**
     * @param handleInvalid - The onInvalid callback for handleSubmit from react-hook-form; this is called when the form is
     * invalid according to react-hook-form and after the form is submitted
     */
    handleInvalid?: SubmitErrorHandler<T>;
    /**
     * @param initialState - The values of the form when it was first rendered with its default values
     */
    initialState?: DeepPartial<T>;
    /**
     * @param processFinalPropertyValues - A function that can be used to process the data that is passed to the
     * final_values property in the FORM_SUBMIT event
     */
    processFinalPropertyValues?: (data: T) => T;
} & AnalyticsEventPropsWithStartInteraction<DesignSystemEventProviderAnalyticsEventTypes.OnSubmit> & Omit<React.ComponentProps<'form'>, 'onSubmit'>;
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
export declare const RhfForm: <T extends FieldValues>({ handleSubmit, handleValid, handleInvalid, processFinalPropertyValues, children, componentId, analyticsEvents, shouldStartInteraction, initialState, ...otherProps }: RhfFormProps<T>) => import("@emotion/react/jsx-runtime").JSX.Element;
//# sourceMappingURL=Form.d.ts.map