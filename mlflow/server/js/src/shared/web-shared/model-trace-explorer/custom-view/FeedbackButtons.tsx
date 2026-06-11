import { useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicBooleanSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Button, ThumbsDownIcon, ThumbsUpIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

/**
 * Event name dispatched when a thumb is clicked. The host wires an
 * `actionHandler` on the MessageProcessor that matches on this name and logs an
 * MLflow feedback assessment. The action context carries `{ value, name, spanId? }`.
 */
export const FEEDBACK_SUBMITTED = 'mlflow.custom-view.feedback-submitted';

/** Default assessment name used when the component doesn't specify one. */
export const DEFAULT_FEEDBACK_NAME = 'User feedback';

/**
 * Schema (API) for the interactive FeedbackButtons primitive: a lightweight
 * thumbs up / thumbs down control. A click reflects the choice in the data
 * model (via the bound `value` path) AND dispatches an A2UI client action so
 * the host can persist it as an MLflow feedback assessment.
 */
export const FeedbackButtonsApi = {
  name: 'FeedbackButtons',
  schema: z
    .object({
      label: DynamicStringSchema.describe('Optional prompt shown next to the thumbs, e.g. "Was this helpful?".').optional(),
      name: DynamicStringSchema.describe(
        'The assessment name to log. Defaults to "User feedback".',
      ).optional(),
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

export const FeedbackButtons = createComponentImplementation(FeedbackButtonsApi, ({ props, context }) => {
  const { theme } = useDesignSystemTheme();

  const initial = typeof props.value === 'boolean' ? props.value : undefined;
  const [selected, setSelected] = useState<boolean | undefined>(initial);

  const label = typeof props.label === 'string' ? props.label : '';
  const weight = typeof props.weight === 'number' ? props.weight : undefined;

  const submit = (value: boolean) => {
    setSelected(value);
    // Reflect the choice in the data model (no-op if `value` isn't bound to a path).
    props.setValue(value);
    // Fire the A2UI client action so the host can persist the feedback.
    void context.dispatchAction({
      event: {
        name: FEEDBACK_SUBMITTED,
        context: {
          value,
          name: typeof props.name === 'string' && props.name ? props.name : DEFAULT_FEEDBACK_NAME,
          ...(typeof props.spanId === 'string' && props.spanId ? { spanId: props.spanId } : {}),
        },
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
      {label && <Typography.Text>{label}</Typography.Text>}
      <Button
        componentId="shared.model-trace-explorer.custom-view.feedback-up"
        icon={<ThumbsUpIcon />}
        type={selected === true ? 'primary' : undefined}
        onClick={() => submit(true)}
        aria-label="Thumbs up"
      />
      <Button
        componentId="shared.model-trace-explorer.custom-view.feedback-down"
        icon={<ThumbsDownIcon />}
        type={selected === false ? 'primary' : undefined}
        onClick={() => submit(false)}
        aria-label="Thumbs down"
      />
    </div>
  );
});
