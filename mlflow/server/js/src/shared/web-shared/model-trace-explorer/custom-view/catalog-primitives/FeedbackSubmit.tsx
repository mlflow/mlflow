import { useEffect, useRef, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Button, CheckIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { asString } from '../catalogPrimitiveUtils';
import { useFeedbackStatus } from '../FeedbackStatusContext';

type SubmitStatus = 'idle' | 'pending' | 'success' | 'error';

/**
 * Schema (API) for the FeedbackSubmit primitive: a button that flushes the
 * staged feedback belonging to its form. The host logs one assessment per
 * staged dimension and this button reflects the real request lifecycle.
 */
const FeedbackSubmitApi = {
  name: 'FeedbackSubmit',
  schema: z
    .object({
      label: DynamicStringSchema.describe('Button text. Defaults to "Submit feedback".').optional(),
      formId: DynamicStringSchema.describe(
        'REQUIRED. Which form this button submits. It flushes every RadioGroup / FeedbackInputText sharing this same "formId" (across any spans), and only those. Every feedback form needs a "formId" — even a single-form view (its controls and submit share one id). Give each form a distinct "formId" when a view has more than one submit button.',
      ),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const FeedbackSubmit: ReactComponentImplementation = createComponentImplementation(
  FeedbackSubmitApi,
  ({ props }) => {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const { hasStagedFeedback, submitStagedFeedback } = useFeedbackStatus();
    const formId = typeof props.formId === 'string' && props.formId ? props.formId : undefined;
    const [status, setStatus] = useState<SubmitStatus>('idle');

    const resetTimerRef = useRef<number>();
    useEffect(() => () => window.clearTimeout(resetTimerRef.current), []);

    const label = props.label
      ? asString(props.label)
      : intl.formatMessage({
          defaultMessage: 'Submit feedback',
          description: 'Default label for the button that submits staged feedback in a custom trace view',
        });
    const weight = typeof props.weight === 'number' ? props.weight : undefined;

    const submit = () => {
      if (status === 'pending' || !hasStagedFeedback(formId)) {
        return;
      }
      window.clearTimeout(resetTimerRef.current);
      setStatus('pending');
      submitStagedFeedback(formId)
        .then(() => {
          setStatus('success');
          window.clearTimeout(resetTimerRef.current);
          resetTimerRef.current = window.setTimeout(() => setStatus('idle'), 2000);
        })
        .catch(() => setStatus('error'));
    };

    const isPending = status === 'pending';
    const isSuccess = status === 'success';
    const isDisabled = !isPending && !hasStagedFeedback(formId);

    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          ...(weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : {}),
          marginTop: theme.spacing.xs,
        }}
      >
        <Button
          componentId="shared.model-trace-explorer.custom-view.feedback-submit-button"
          type="primary"
          icon={isSuccess ? <CheckIcon /> : undefined}
          loading={isPending}
          disabled={isDisabled}
          onClick={submit}
        >
          {isSuccess
            ? intl.formatMessage({
                defaultMessage: 'Feedback submitted',
                description: 'Confirmation shown briefly on the feedback submit button after staged feedback is logged',
              })
            : label}
        </Button>
        {status === 'error' && (
          <Typography.Text color="error" size="sm" aria-live="polite">
            {intl.formatMessage({
              defaultMessage: 'Could not submit feedback. Try again.',
              description: 'Inline error shown when submitting staged feedback fails in a custom trace view',
            })}
          </Typography.Text>
        )}
      </div>
    );
  },
);
