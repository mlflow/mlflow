import { useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { CheckCircleIcon, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { AssessmentCreateForm } from '../assessments-pane/AssessmentCreateForm';
import { ContentBlock } from './ContentViewer';

/**
 * Schema (API) for the FeedbackForm component. It wraps the trace explorer's
 * real AssessmentCreateForm so feedback submitted here is persisted to the
 * backend as a span-scoped assessment. Each instance targets one span.
 *
 * Optionally carries the span's `inputs` / `outputs` content fields so the card
 * can reveal the span's input/output (same renderer as the ContentViewer
 * primitive) behind a toggle, alongside the feedback form.
 */
const ContentFieldSchema = z.object({
  label: DynamicStringSchema.describe('The field key shown above its value.').optional(),
  value: DynamicStringSchema.describe('JSON-encoded field value (object -> JSON, string -> text).'),
});

export const FeedbackFormApi = {
  name: 'FeedbackForm',
  schema: z
    .object({
      traceId: DynamicStringSchema.describe('The id of the trace the span belongs to.'),
      spanId: DynamicStringSchema.describe('The id of the span the feedback is scoped to.'),
      spanName: DynamicStringSchema.describe('The display name of the span, shown in the header.').optional(),
      inputs: z.array(ContentFieldSchema).describe("The span's input fields, shown when the toggle is on.").optional(),
      outputs: z
        .array(ContentFieldSchema)
        .describe("The span's output fields, shown when the toggle is on.")
        .optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const FeedbackForm = createComponentImplementation(FeedbackFormApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const traceId = asString(props.traceId);
  const spanId = asString(props.spanId);
  const spanName = props.spanName ? asString(props.spanName) : spanId;
  const inputs = Array.isArray(props.inputs) ? (props.inputs as { label?: unknown; value?: unknown }[]) : [];
  const outputs = Array.isArray(props.outputs) ? (props.outputs as { label?: unknown; value?: unknown }[]) : [];
  const hasInputs = inputs.length > 0;
  const hasOutputs = outputs.length > 0;

  const [submitted, setSubmitted] = useState(false);
  const [showInputs, setShowInputs] = useState(false);
  const [showOutputs, setShowOutputs] = useState(false);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.sm,
          flexWrap: 'wrap',
        }}
      >
        <Typography.Text bold css={{ fontFamily: 'monospace' }}>
          {spanName}
        </Typography.Text>
        {(hasInputs || hasOutputs) && (
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
            {hasInputs && (
              <Switch
                componentId="shared.model-trace-explorer.custom-view.feedback.show-inputs"
                checked={showInputs}
                onChange={setShowInputs}
                label="Show input"
              />
            )}
            {hasOutputs && (
              <Switch
                componentId="shared.model-trace-explorer.custom-view.feedback.show-outputs"
                checked={showOutputs}
                onChange={setShowOutputs}
                label="Show output"
              />
            )}
          </div>
        )}
      </div>
      {((showInputs && hasInputs) || (showOutputs && hasOutputs)) && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {showInputs && hasInputs && (
            <ContentBlock title="Inputs" icon="list" fields={inputs} emptyMessage={`No inputs on ${spanName}.`} />
          )}
          {showOutputs && hasOutputs && (
            <ContentBlock title="Outputs" icon="checklist" fields={outputs} emptyMessage={`No outputs on ${spanName}.`} />
          )}
        </div>
      )}
      {submitted ? (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
          <Typography.Text color="success">Feedback submitted.</Typography.Text>
        </div>
      ) : (
        <AssessmentCreateForm
          assessmentType="feedback"
          traceId={traceId}
          spanId={spanId}
          setExpanded={(expanded) => {
            // AssessmentCreateForm calls setExpanded(false) after a successful
            // create; treat that as a submission confirmation.
            if (!expanded) {
              setSubmitted(true);
            }
          }}
        />
      )}
    </div>
  );
});
