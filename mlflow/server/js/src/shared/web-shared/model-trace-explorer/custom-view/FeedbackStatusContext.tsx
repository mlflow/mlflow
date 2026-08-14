import { createContext, useContext, type Provider } from 'react';

/**
 * Host-provided bridge for the interactive feedback primitives.
 *
 * Two distinct needs, deliberately split:
 *  - FeedbackThumbsUpDownButtons logs ONE assessment on click and owns its own
 *    `useCreateAssessment` hook (mirroring AssessmentCreateForm), so it only
 *    needs the current `traceId` from here to build the payload. Giving each
 *    thumb its own hook keeps separate react-query MutationObservers, so two
 *    thumbs clicked in quick succession never clobber each other's callbacks.
 *  - FeedbackSubmit flushes SEVERAL staged dimensions at once, which no single
 *    `useCreateAssessment` form models, so the host owns the staged buffer and
 *    exposes `submitStagedFeedback` to flush it.
 *
 * Provided around the `A2uiSurface`; React context propagates into the rendered
 * catalog components.
 */
export interface FeedbackStatusContextValue {
  // True once a host bridge is mounted. When false, the primitives fall back to
  // their reflect-only behavior (no assessment is logged).
  enabled: boolean;
  // The current trace id, so a thumb can build its CreateAssessment payload.
  traceId: string;
  // Whether there is staged feedback with a submittable value ready to submit.
  // A submit owns exactly one form, identified by `formId`: it counts only the
  // entries staged under that same `formId`. Reactive: flips as RadioGroup /
  // FeedbackInputText inputs stage or clear.
  hasStagedFeedback: (formId?: string) => boolean;
  // Returns a staged field for a logical feedback entry. Controls use it to
  // preserve in-progress values when assessment data refreshes and causes the
  // A2UI surface to re-bind without changing its template.
  getStagedFeedbackValue: (
    entry: { name: string; spanId?: string; formId?: string },
    field?: 'value' | 'rationale',
  ) => string | undefined;
  // Flushes staged RadioGroup / FeedbackInputText feedback for the given form.
  // Only entries sharing this `formId` are submitted, so one form's submit never
  // flushes another form's staged feedback. Failed dimensions stay staged for
  // retry; rejects when any dimension fails so partial success is not presented
  // as a fully successful form submission.
  submitStagedFeedback: (formId?: string) => Promise<{ submitted: number }>;
  // Increments after this logical feedback entry is persisted. Staged controls
  // use it to clear their visible/data-model values together with the host
  // buffer, without resetting another form or span's input.
  getFeedbackResetVersion: (entry: { name: string; spanId?: string; formId?: string }) => number;
}

const FeedbackStatusContext = createContext<FeedbackStatusContextValue>({
  enabled: false,
  traceId: '',
  hasStagedFeedback: () => false,
  getStagedFeedbackValue: () => undefined,
  submitStagedFeedback: () => Promise.resolve({ submitted: 0 }),
  getFeedbackResetVersion: () => 0,
});

export const FeedbackStatusProvider: Provider<FeedbackStatusContextValue> = FeedbackStatusContext.Provider;

export const useFeedbackStatus = (): FeedbackStatusContextValue => useContext(FeedbackStatusContext);
