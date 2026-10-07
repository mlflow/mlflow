import { useCallback, useEffect, useRef, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Radio, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { FEEDBACK_STAGED } from './feedbackActions';
import { asString } from '../catalogPrimitiveUtils';
import { useFeedbackStatus } from '../FeedbackStatusContext';

let radioGroupId = 0;
const nextRadioGroupName = () => `shared-model-trace-explorer-custom-view-radio-group-${radioGroupId++}`;

/**
 * Schema (API) for the interactive RadioGroup feedback primitive: a single
 * choice among a fixed set of string options. Selecting an option reflects the
 * choice in the data model and stages it host-side; it is persisted as a
 * feedback assessment only when a FeedbackSubmit button is clicked.
 */
const RadioGroupApi = {
  name: 'RadioGroup',
  schema: z
    .object({
      label: DynamicStringSchema.describe(
        'Optional prompt shown above the options, e.g. "Who did better on Accuracy?".',
      ).optional(),
      name: DynamicStringSchema.describe(
        'The assessment name this dimension logs (also the staging key). Required, and must be unique per dimension; a FeedbackInputText sharing this name attaches as its rationale.',
      ),
      options: z
        .array(
          z.object({
            label: DynamicStringSchema.describe('The option text shown to the user.'),
            value: z.string().describe('The feedback value logged when this option is selected.'),
          }),
        )
        .describe('The selectable options, in display order.'),
      value: DynamicStringSchema.describe(
        'Selected option value. Bind to a /feedback/... path to reflect and seed the choice.',
      ).optional(),
      spanId: DynamicStringSchema.describe(
        'Optional span id to scope the feedback to a specific span instead of the whole trace.',
      ).optional(),
      formId: DynamicStringSchema.describe(
        'REQUIRED. Which form this control belongs to. The FeedbackSubmit sharing this same "formId" submits it. Every feedback form needs a "formId" — even a single-form view. Set a distinct "formId" per form when a view has more than one submit button. A rationale FeedbackInputText must share BOTH this "formId" and the same "name" to attach.',
      ),
      weight: z.number().describe('Relative flex weight when placed directly inside a Row/Column.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const RadioGroup: ReactComponentImplementation = createComponentImplementation(
  RadioGroupApi,
  ({ props, context }) => {
    const { theme } = useDesignSystemTheme();
    const { getFeedbackResetVersion, getStagedFeedbackValue } = useFeedbackStatus();
    const boundValue = typeof props.value === 'string' ? props.value : '';
    const setValue = props.setValue;

    const contextRef = useRef(context);
    contextRef.current = context;

    const label = props.label ? asString(props.label) : '';
    const name = props.name ? asString(props.name) : '';
    const options = Array.isArray(props.options) ? props.options : [];
    const spanId = typeof props.spanId === 'string' && props.spanId ? props.spanId : undefined;
    const formId = typeof props.formId === 'string' && props.formId ? props.formId : undefined;
    const weight = typeof props.weight === 'number' ? props.weight : undefined;
    const [groupName] = useState(nextRadioGroupName);
    const initial = getStagedFeedbackValue({ name, spanId, formId }) ?? boundValue;
    const [selected, setSelected] = useState(initial);
    const resetVersion = getFeedbackResetVersion({ name, spanId, formId });
    const observedResetVersionRef = useRef(resetVersion);
    const lastSyncedValueRef = useRef<string>();
    const hasSyncedValueRef = useRef(false);

    const dispatchStage = useCallback(
      (value: string) => {
        void contextRef.current.dispatchAction({
          event: {
            name: FEEDBACK_STAGED,
            context: {
              name,
              value,
              ...(spanId ? { spanId } : {}),
              ...(formId ? { formId } : {}),
            },
          },
        });
      },
      [formId, name, spanId],
    );

    useEffect(() => {
      if (observedResetVersionRef.current !== resetVersion) {
        observedResetVersionRef.current = resetVersion;
        hasSyncedValueRef.current = true;
        lastSyncedValueRef.current = '';
        setSelected('');
        setValue('');
        return;
      }
      if (hasSyncedValueRef.current && lastSyncedValueRef.current === initial) {
        return;
      }
      const hadSyncedValue = hasSyncedValueRef.current;
      hasSyncedValueRef.current = true;
      lastSyncedValueRef.current = initial;
      setSelected(initial);
      // A prefilled bound value is active feedback, so stage it immediately.
      // Also propagate a later external clear to remove an already-staged value.
      if (initial || hadSyncedValue) {
        dispatchStage(initial);
      }
    }, [dispatchStage, initial, resetVersion, setValue]);

    const select = (value: string) => {
      hasSyncedValueRef.current = true;
      lastSyncedValueRef.current = value;
      setSelected(value);
      setValue(value);
      dispatchStage(value);
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
        <Radio.Group
          name={groupName}
          componentId="shared.model-trace-explorer.custom-view.feedback-radio-group"
          value={selected}
          onChange={(event) => select(asString(event.target.value))}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {options.map((option, index) => {
              const optionValue = asString(option?.value);
              const isSelected = selected === optionValue;
              return (
                <div
                  key={`${optionValue}-${index}`}
                  onClick={() => select(optionValue)}
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    cursor: 'pointer',
                    padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                    borderRadius: theme.borders.borderRadiusMd,
                    backgroundColor: theme.colors.backgroundPrimary,
                    border: `1px solid ${
                      isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
                    }`,
                  }}
                >
                  <Radio value={optionValue} css={{ width: '100%', marginRight: 0 }}>
                    {asString(option?.label) || optionValue}
                  </Radio>
                </div>
              );
            })}
          </div>
        </Radio.Group>
      </div>
    );
  },
);
