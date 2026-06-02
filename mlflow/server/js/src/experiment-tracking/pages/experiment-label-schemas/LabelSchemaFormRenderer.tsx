import { useState } from 'react';
import { Button, Checkbox, FormUI, Input, PlusIcon, Radio, Tag, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Controller, type Control } from 'react-hook-form';

import type { LabelSchemaType } from '../../components/label-schemas/types';
import {
  MAX_CATEGORICAL_OPTIONS,
  PASS_FAIL_NEGATIVE_DEFAULT,
  PASS_FAIL_POSITIVE_DEFAULT,
  type LabelSchemaFormData,
  type LabelSchemaFormErrors,
  type LabelSchemaInputKind,
} from './labelSchemaFormUtils';

type FormErrors = LabelSchemaFormErrors;

export interface LabelSchemaFormRendererProps {
  control: Control<LabelSchemaFormData>;
  /**
   * On edit, the schema `name`, `type`, and `inputKind` are immutable (the
   * server enforces this); the form disables them. Create allows all three.
   */
  isEdit: boolean;
  errors: FormErrors;
  /** Watch values are passed in by the parent so the renderer is pure. */
  watchedValues: Pick<LabelSchemaFormData, 'inputKind'>;
}

const COMPONENT_PREFIX = 'mlflow.experiment-label-schemas.form';

export const LabelSchemaFormRenderer = ({ control, isEdit, errors, watchedValues }: LabelSchemaFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const { inputKind } = watchedValues;
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label
          htmlFor={`${COMPONENT_PREFIX}.name`}
          required
          infoPopoverContents={
            <FormattedMessage
              defaultMessage="Shown to reviewers as the label prompt and used as the label's key on collected feedback. Up to 256 characters. Immutable after create."
              description="Label schema name hint"
            />
          }
        >
          <FormattedMessage defaultMessage="Name" description="Label schema name input" />
        </FormUI.Label>
        <Controller
          name="name"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.name`}
              id={`${COMPONENT_PREFIX}.name`}
              {...field}
              disabled={isEdit}
              placeholder="Is the answer correct?"
            />
          )}
        />
        {errors.name && <FormUI.Message message={errors.name} type="error" />}
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.instruction`}>
          <FormattedMessage defaultMessage="Instructions (optional)" description="Label schema instruction input" />
        </FormUI.Label>
        <Controller
          name="instruction"
          control={control}
          render={({ field }) => (
            <Input.TextArea
              componentId={`${COMPONENT_PREFIX}.instruction`}
              id={`${COMPONENT_PREFIX}.instruction`}
              {...field}
              rows={3}
              placeholder="Instructions for reviewers on how to complete this label"
            />
          )}
        />
        {errors.instruction && <FormUI.Message message={errors.instruction} type="error" />}
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.input-kind`} required>
          <FormattedMessage defaultMessage="Input type" description="Label schema input variant selector" />
        </FormUI.Label>
        <Controller
          name="inputKind"
          control={control}
          render={({ field }) => (
            <Radio.Group
              componentId={`${COMPONENT_PREFIX}.input-kind`}
              name={`${COMPONENT_PREFIX}.input-kind`}
              layout="horizontal"
              value={field.value}
              onChange={(e) => field.onChange(e.target.value as LabelSchemaInputKind)}
              disabled={isEdit}
            >
              <Radio value="pass_fail">Pass / Fail</Radio>
              <Radio value="categorical">Categorical</Radio>
              <Radio value="numeric">Numeric</Radio>
              <Radio value="text">Text</Radio>
            </Radio.Group>
          )}
        />
        {/* Per-input options live with the input-type selector: the rationale
            toggle always, then multi-select when the type is categorical. */}
        <Controller
          name="enable_comment"
          control={control}
          render={({ field }) => (
            <Checkbox
              componentId={`${COMPONENT_PREFIX}.enable-comment`}
              isChecked={field.value}
              onChange={(checked) => field.onChange(checked)}
              css={{ marginTop: theme.spacing.sm }}
            >
              <FormattedMessage
                defaultMessage="Collect a free-form rationale alongside the input"
                description="Enable rationale checkbox"
              />
            </Checkbox>
          )}
        />
        {inputKind === 'categorical' && (
          <Controller
            name="categoricalMultiSelect"
            control={control}
            render={({ field }) => (
              <Checkbox
                componentId={`${COMPONENT_PREFIX}.categorical.multi-select`}
                isChecked={field.value}
                onChange={(checked) => field.onChange(checked)}
                css={{ marginTop: theme.spacing.sm }}
              >
                <FormattedMessage
                  defaultMessage="Allow multiple selections (multi-select)"
                  description="Categorical multi-select checkbox"
                />
              </Checkbox>
            )}
          />
        )}
      </div>

      {/* Feedback vs. expectation (ground truth): a horizontal radio group
          mirroring the input-type selector above. Immutable on edit. */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.type`} required>
          <FormattedMessage
            defaultMessage="Label type"
            description="Label schema feedback-vs-expectation selector label"
          />
        </FormUI.Label>
        <Controller
          name="type"
          control={control}
          render={({ field }) => (
            <Radio.Group
              componentId={`${COMPONENT_PREFIX}.type`}
              name={`${COMPONENT_PREFIX}.type`}
              layout="horizontal"
              value={field.value}
              onChange={(e) => field.onChange(e.target.value as LabelSchemaType)}
              disabled={isEdit}
            >
              <Radio value="FEEDBACK">Feedback</Radio>
              <Radio value="EXPECTATION">Expectation (ground truth)</Radio>
            </Radio.Group>
          )}
        />
      </div>

      {inputKind === 'pass_fail' && <PassFailFields control={control} errors={errors} />}
      {inputKind === 'categorical' && <CategoricalFields control={control} errors={errors} />}
      {inputKind === 'numeric' && <NumericFields control={control} errors={errors} />}
      {inputKind === 'text' && <TextFields control={control} errors={errors} />}
    </div>
  );
};

const PassFailFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  // Positive / negative sit side-by-side (item 9) to save vertical space.
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.pass-fail.positive`} required>
          <FormattedMessage defaultMessage="Positive label" description="Pass/Fail positive label input" />
        </FormUI.Label>
        <Controller
          name="passFailPositiveLabel"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.pass-fail.positive`}
              id={`${COMPONENT_PREFIX}.pass-fail.positive`}
              {...field}
              placeholder={PASS_FAIL_POSITIVE_DEFAULT}
            />
          )}
        />
        {errors.passFailPositiveLabel && <FormUI.Message message={errors.passFailPositiveLabel} type="error" />}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.pass-fail.negative`} required>
          <FormattedMessage defaultMessage="Negative label" description="Pass/Fail negative label input" />
        </FormUI.Label>
        <Controller
          name="passFailNegativeLabel"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.pass-fail.negative`}
              id={`${COMPONENT_PREFIX}.pass-fail.negative`}
              {...field}
              placeholder={PASS_FAIL_NEGATIVE_DEFAULT}
            />
          )}
        />
        {errors.passFailNegativeLabel && <FormUI.Message message={errors.passFailNegativeLabel} type="error" />}
      </div>
    </div>
  );
};

/**
 * Editable options list: each existing option is an inline-editable row
 * with a remove button, and a trailing input adds a new option (Enter or
 * the Add button). Order is preserved; blanks and duplicates are dropped
 * on submit by `normalizeCategoricalOptions`.
 */
const CategoricalOptionsEditor = ({ value, onChange }: { value: string[]; onChange: (next: string[]) => void }) => {
  const { theme } = useDesignSystemTheme();
  const [draft, setDraft] = useState('');
  const atMax = value.length >= MAX_CATEGORICAL_OPTIONS;

  const addDraft = () => {
    const trimmed = draft.trim();
    if (trimmed === '' || atMax) {
      return;
    }
    setDraft('');
    // Skip duplicates silently; normalizeCategoricalOptions would drop them anyway.
    if (!value.includes(trimmed)) {
      onChange([...value, trimmed]);
    }
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Input
          componentId={`${COMPONENT_PREFIX}.categorical.new-option`}
          id={`${COMPONENT_PREFIX}.categorical.new-option`}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addDraft();
            }
          }}
          placeholder={atMax ? `Max ${MAX_CATEGORICAL_OPTIONS} options` : 'Add an option'}
          disabled={atMax}
          css={{ flex: 1 }}
        />
        <Button
          componentId={`${COMPONENT_PREFIX}.categorical.add-option`}
          icon={<PlusIcon />}
          onClick={addDraft}
          disabled={atMax || draft.trim() === ''}
        >
          <FormattedMessage defaultMessage="Add" description="Categorical add-option button" />
        </Button>
      </div>
      {value.length > 0 && (
        // Render options as removable tag chips (mirroring the Add/Edit tags
        // modal), shown below the add box. Chips flow horizontally and wrap;
        // the list scrolls within its own window once it exceeds it. Options
        // are unique (deduped on add), so the value is a stable key.
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            gap: theme.spacing.xs,
            maxHeight: theme.spacing.md * 9,
            overflowY: 'auto',
          }}
        >
          {value.map((option) => (
            <Tag
              key={option}
              componentId={`${COMPONENT_PREFIX}.categorical.option`}
              closable
              onClose={() => onChange(value.filter((o) => o !== option))}
              css={{
                paddingTop: theme.spacing.xs,
                paddingBottom: theme.spacing.xs,
                paddingLeft: theme.spacing.sm,
              }}
            >
              {option}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
};

const CategoricalFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  // `categoricalMultiSelect` lives with the input-type selector (it modifies
  // the categorical type); this fieldset only owns the options list.
  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.categorical.new-option`} required>
        <FormattedMessage defaultMessage="Options" description="Categorical options list label" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Up to {max} options."
          description="Categorical options hint"
          values={{ max: MAX_CATEGORICAL_OPTIONS }}
        />
      </FormUI.Hint>
      <Controller
        name="categoricalOptions"
        control={control}
        render={({ field }) => <CategoricalOptionsEditor value={field.value} onChange={field.onChange} />}
      />
      {errors.categoricalOptions && <FormUI.Message message={errors.categoricalOptions} type="error" />}
    </div>
  );
};

const NumericFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  // Min / max sit side-by-side (item 9) to save vertical space.
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label
          htmlFor={`${COMPONENT_PREFIX}.numeric.min`}
          infoPopoverContents={
            <FormattedMessage
              defaultMessage="Define the acceptable range for numeric input values. Leave either bound blank for no limit."
              description="Numeric range hint"
            />
          }
        >
          <FormattedMessage defaultMessage="Min value" description="Numeric min value input" />
        </FormUI.Label>
        <Controller
          name="numericMinValue"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.numeric.min`}
              id={`${COMPONENT_PREFIX}.numeric.min`}
              type="number"
              {...field}
              placeholder="No minimum"
            />
          )}
        />
        {errors.numericMinValue && <FormUI.Message message={errors.numericMinValue} type="error" />}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.numeric.max`}>
          <FormattedMessage defaultMessage="Max value" description="Numeric max value input" />
        </FormUI.Label>
        <Controller
          name="numericMaxValue"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.numeric.max`}
              id={`${COMPONENT_PREFIX}.numeric.max`}
              type="number"
              {...field}
              placeholder="No maximum"
            />
          )}
        />
        {errors.numericMaxValue && <FormUI.Message message={errors.numericMaxValue} type="error" />}
      </div>
    </div>
  );
};

const TextFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.text.max-length`}>
          <FormattedMessage defaultMessage="Max length" description="Text max length input" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Optional character limit for the free-form text. Leave blank for no limit."
            description="Text max length hint"
          />
        </FormUI.Hint>
        <Controller
          name="textMaxLength"
          control={control}
          render={({ field }) => (
            <Input
              componentId={`${COMPONENT_PREFIX}.text.max-length`}
              id={`${COMPONENT_PREFIX}.text.max-length`}
              type="number"
              {...field}
              placeholder="No limit"
            />
          )}
        />
        {errors.textMaxLength && <FormUI.Message message={errors.textMaxLength} type="error" />}
      </div>
    </div>
  );
};
