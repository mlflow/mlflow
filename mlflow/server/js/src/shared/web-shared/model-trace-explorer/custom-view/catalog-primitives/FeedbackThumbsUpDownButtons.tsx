import { useEffect, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicBooleanSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  Button,
  CheckCircleIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  Typography,
  useDesignSystemTheme,
  visuallyHidden,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { Feedback } from '../../ModelTrace.types';
import { useCreateAssessment } from '../../hooks/useCreateAssessment';
import { getUser } from '../../../global-settings/getUser';
import { asString } from '../catalogPrimitiveUtils';
import { useFeedbackStatus } from '../FeedbackStatusContext';

/** Default assessment name used when the component doesn't specify one. */
export const DEFAULT_FEEDBACK_NAME = 'User feedback';

type SubmitStatus = 'idle' | 'success' | 'error';

/**
 * Schema (API) for the interactive FeedbackThumbsUpDownButtons primitive: a
 * lightweight thumbs up / thumbs down control. A click reflects the choice in
 * the data model (via the bound `value` path) AND logs it as a feedback
 * assessment through the same mutation used by the assessments pane.
 */
const FeedbackThumbsUpDownButtonsApi = {
  name: 'FeedbackThumbsUpDownButtons',
  schema: z
    .object({
      label: DynamicStringSchema.describe(
        'Optional prompt shown next to the thumbs, e.g. "Was this helpful?".',
      ).optional(),
      name: DynamicStringSchema.describe('The assessment name to log. Defaults to "User feedback".').optional(),
      value: DynamicBooleanSchema.describe(
        'Selected state: true = thumbs up, false = thumbs down. Bind to a /feedback/... path to reflect and seed the choice.',
      ).optional(),
      spanId: DynamicStringSchema.describe(
        'Optional span id to scope the feedback to a specific span instead of the whole trace.',
      ).optional(),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const FeedbackThumbsUpDownButtons: ReactComponentImplementation = createComponentImplementation(
  FeedbackThumbsUpDownButtonsApi,
  ({ props }) => {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const { enabled, traceId } = useFeedbackStatus();
    const [status, setStatus] = useState<SubmitStatus>('idle');
    const boundValue = typeof props.value === 'boolean' ? props.value : undefined;
    const [selected, setSelected] = useState<boolean | undefined>(boundValue);

    useEffect(() => setSelected(boundValue), [boundValue]);

    const { createAssessmentMutation, isLoading } = useCreateAssessment({
      traceId,
      onSuccess: () => setStatus('success'),
      onError: () => setStatus('error'),
    });

    const label = typeof props.label === 'string' ? props.label : '';
    const weight = typeof props.weight === 'number' ? props.weight : undefined;

    const submit = (value: boolean) => {
      if (isLoading) {
        return;
      }
      setSelected(value);
      props.setValue(value);

      if (!enabled || !traceId) {
        return;
      }

      setStatus('idle');
      const name = typeof props.name === 'string' && props.name ? props.name : DEFAULT_FEEDBACK_NAME;
      const spanId = typeof props.spanId === 'string' && props.spanId ? props.spanId : undefined;
      const feedbackValue: { feedback: Feedback } = { feedback: { value } };
      createAssessmentMutation({
        assessment: {
          assessment_name: name,
          trace_id: traceId,
          source: { source_type: 'HUMAN', source_id: getUser() ?? '' },
          ...(spanId ? { span_id: spanId } : {}),
          ...feedbackValue,
        },
      });
    };

    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          ...(weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : {}),
        }}
      >
        {label && <Typography.Text>{asString(label)}</Typography.Text>}
        {status === 'success' ? (
          <span
            css={{ display: 'inline-flex', alignItems: 'center', color: theme.colors.textValidationSuccess }}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            <CheckCircleIcon aria-hidden="true" />
            <span css={visuallyHidden}>
              {intl.formatMessage({
                defaultMessage: 'Feedback submitted',
                description: 'Accessible confirmation after thumbs feedback is logged in a custom trace view',
              })}
            </span>
          </span>
        ) : (
          <>
            <Button
              componentId="shared.model-trace-explorer.custom-view.feedback-button-up"
              icon={<ThumbsUpIcon />}
              type={selected === true ? 'primary' : undefined}
              disabled={isLoading}
              onClick={() => submit(true)}
              aria-pressed={selected === true}
              aria-label={intl.formatMessage({
                defaultMessage: 'Thumbs up',
                description: 'Accessible label for the thumbs-up feedback button in a custom trace view',
              })}
            />
            <Button
              componentId="shared.model-trace-explorer.custom-view.feedback-button-down"
              icon={<ThumbsDownIcon />}
              type={selected === false ? 'primary' : undefined}
              disabled={isLoading}
              onClick={() => submit(false)}
              aria-pressed={selected === false}
              aria-label={intl.formatMessage({
                defaultMessage: 'Thumbs down',
                description: 'Accessible label for the thumbs-down feedback button in a custom trace view',
              })}
            />
            {status === 'error' && (
              <Typography.Text color="error" size="sm" aria-live="polite">
                {intl.formatMessage({
                  defaultMessage: 'Could not save. Try again.',
                  description: 'Inline error shown when logging thumbs feedback fails in a custom trace view',
                })}
              </Typography.Text>
            )}
          </>
        )}
      </div>
    );
  },
);
