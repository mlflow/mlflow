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
  // Flushes staged RadioGroup / FeedbackInputText feedback for the given form.
  // Only entries sharing this `formId` are submitted, so one form's submit never
  // flushes another form's staged feedback. Failed dimensions stay staged for
  // retry; rejects only when nothing succeeded.
  submitStagedFeedback: (formId?: string) => Promise<{ submitted: number }>;
}

const FeedbackStatusContext = createContext<FeedbackStatusContextValue>({
  enabled: false,
  traceId: '',
  hasStagedFeedback: () => false,
  submitStagedFeedback: () => Promise.resolve({ submitted: 0 }),
});

export const FeedbackStatusProvider: Provider<FeedbackStatusContextValue> = FeedbackStatusContext.Provider;

export const useFeedbackStatus = (): FeedbackStatusContextValue => useContext(FeedbackStatusContext);
