import { useCallback, useEffect, useRef, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Input, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { FEEDBACK_STAGED } from './feedbackActions';
import { asString } from '../catalogPrimitiveUtils';
import { useFeedbackStatus } from '../FeedbackStatusContext';

const FIELDS = ['value', 'rationale'] as const;
type FieldTarget = (typeof FIELDS)[number];

/**
 * Schema (API) for the FeedbackInputText primitive: a feedback-scoped free-text
 * box. Its text is staged host-side under `name` and only persisted on
 * FeedbackSubmit. `field` controls whether it becomes the assessment value or
 * the rationale attached to a RadioGroup with the same name and form.
 */
const FeedbackInputTextApi = {
  name: 'FeedbackInputText',
  schema: z
    .object({
      label: DynamicStringSchema.describe(
        'Optional prompt shown above the box, e.g. "Optional rationale (why?)".',
      ).optional(),
      name: DynamicStringSchema.describe(
        "The assessment name this text logs (also the staging key). Match a RadioGroup's name to attach as that dimension's rationale; use a unique name for a standalone free-text feedback value.",
      ),
      field: z
        .enum(FIELDS)
        .describe('Whether the typed text becomes the assessment "value" or its "rationale". Defaults to "rationale".')
        .default('rationale')
        .optional(),
      placeholder: DynamicStringSchema.describe('Placeholder text shown when the box is empty.').optional(),
      value: DynamicStringSchema.describe('Bind to a /feedback/... path to reflect and seed the text.').optional(),
      spanId: DynamicStringSchema.describe(
        'Optional span id to scope the feedback to a specific span instead of the whole trace.',
      ).optional(),
      formId: DynamicStringSchema.describe(
        'REQUIRED. Which form this control belongs to. The FeedbackSubmit sharing this same "formId" submits it. As a rationale (field "rationale"), match BOTH the "formId" and "name" of the RadioGroup it annotates. Every feedback form needs a "formId" — even a single-form view. Set a distinct "formId" per form when a view has more than one submit button.',
      ),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const FeedbackInputText: ReactComponentImplementation = createComponentImplementation(
  FeedbackInputTextApi,
  ({ props, context }) => {
    const { theme } = useDesignSystemTheme();
    const { getFeedbackResetVersion, getStagedFeedbackValue } = useFeedbackStatus();
    const boundValue = typeof props.value === 'string' ? props.value : '';
    const setValue = props.setValue;

    const contextRef = useRef(context);
    contextRef.current = context;

    const label = props.label ? asString(props.label) : '';
    const name = props.name ? asString(props.name) : '';
    const placeholder = props.placeholder ? asString(props.placeholder) : undefined;
    const field: FieldTarget = props.field === 'value' ? 'value' : 'rationale';
    const spanId = typeof props.spanId === 'string' && props.spanId ? props.spanId : undefined;
    const formId = typeof props.formId === 'string' && props.formId ? props.formId : undefined;
    const weight = typeof props.weight === 'number' ? props.weight : undefined;
    const initial = getStagedFeedbackValue({ name, spanId, formId }, field) ?? boundValue;
    const [text, setText] = useState<string>(initial);
    const resetVersion = getFeedbackResetVersion({ name, spanId, formId });
    const observedResetVersionRef = useRef(resetVersion);
    const lastSyncedValueRef = useRef<string>();
    const hasSyncedValueRef = useRef(false);

    const dispatchStage = useCallback(
      (next: string) => {
        void contextRef.current.dispatchAction({
          event: {
            name: FEEDBACK_STAGED,
            context: {
              name,
              ...(field === 'value' ? { value: next } : { rationale: next }),
              ...(spanId ? { spanId } : {}),
              ...(formId ? { formId } : {}),
            },
          },
        });
      },
      [field, formId, name, spanId],
    );

    useEffect(() => {
      if (observedResetVersionRef.current !== resetVersion) {
        observedResetVersionRef.current = resetVersion;
        hasSyncedValueRef.current = true;
        lastSyncedValueRef.current = '';
        setText('');
        setValue('');
        return;
      }
      if (hasSyncedValueRef.current && lastSyncedValueRef.current === initial) {
        return;
      }
      const hadSyncedValue = hasSyncedValueRef.current;
      hasSyncedValueRef.current = true;
      lastSyncedValueRef.current = initial;
      setText(initial);
      // A prefilled bound value is active feedback, so stage it immediately.
      // Also propagate a later external clear to remove an already-staged value.
      if (initial.length > 0 || hadSyncedValue) {
        dispatchStage(initial);
      }
    }, [dispatchStage, initial, resetVersion, setValue]);

    const stage = (next: string) => {
      hasSyncedValueRef.current = true;
      lastSyncedValueRef.current = next;
      setValue(next);
      dispatchStage(next);
    };

    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          ...(weight !== undefined ? { flex: `${weight}`, minWidth: 0 } : {}),
        }}
      >
        {label && <Typography.Text color="secondary">{label}</Typography.Text>}
        <Input.TextArea
          componentId="shared.model-trace-explorer.custom-view.feedback-input-text"
          aria-label={label || name}
          placeholder={placeholder}
          value={text}
          autoSize={{ minRows: 2, maxRows: 6 }}
          css={{ backgroundColor: theme.colors.backgroundPrimary }}
          onKeyDown={(event) => event.stopPropagation()}
          onChange={(event) => {
            setText(event.target.value);
            stage(event.target.value);
          }}
        />
      </div>
    );
  },
);
