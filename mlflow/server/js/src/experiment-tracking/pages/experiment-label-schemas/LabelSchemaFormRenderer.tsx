import { Checkbox, FormUI, Input, Radio, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Controller, type Control } from 'react-hook-form';

import {
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
              defaultMessage="Shown to reviewers as the label prompt and used as the assessment key. Up to 256 characters. Immutable after create."
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
      </div>

      {/* Compact option checkboxes (Braintrust-style): the assessment type
          reads as a single boolean toggle, sitting beside the rationale
          toggle to save vertical space. */}
      <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.lg }}>
        <Controller
          name="type"
          control={control}
          render={({ field }) => (
            <Checkbox
              componentId={`${COMPONENT_PREFIX}.type-expectation`}
              isChecked={field.value === 'EXPECTATION'}
              onChange={(checked) => field.onChange(checked ? 'EXPECTATION' : 'FEEDBACK')}
              isDisabled={isEdit}
            >
              <FormattedMessage
                defaultMessage="Collect as an expectation (ground truth) instead of feedback"
                description="Label schema type-as-expectation checkbox"
              />
            </Checkbox>
          )}
        />
        <Controller
          name="enable_comment"
          control={control}
          render={({ field }) => (
            <Checkbox
              componentId={`${COMPONENT_PREFIX}.enable-comment`}
              isChecked={field.value}
              onChange={(checked) => field.onChange(checked)}
            >
              <FormattedMessage
                defaultMessage="Collect a free-form rationale alongside the input"
                description="Enable rationale checkbox"
              />
            </Checkbox>
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

const CategoricalFields = ({ control, errors }: { control: Control<LabelSchemaFormData>; errors: FormErrors }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`${COMPONENT_PREFIX}.categorical.options`} required>
          <FormattedMessage defaultMessage="Options (one per line)" description="Categorical options textarea label" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="1-100 options, each 1-64 characters, in your preferred order. Duplicates are removed automatically."
            description="Categorical options hint"
          />
        </FormUI.Hint>
        <Controller
          name="categoricalOptions"
          control={control}
          render={({ field }) => (
            <Input.TextArea
              componentId={`${COMPONENT_PREFIX}.categorical.options`}
              id={`${COMPONENT_PREFIX}.categorical.options`}
              {...field}
              rows={5}
              placeholder={'low\nmedium\nhigh'}
            />
          )}
        />
        {errors.categoricalOptions && <FormUI.Message message={errors.categoricalOptions} type="error" />}
      </div>
      <Controller
        name="categoricalMultiSelect"
        control={control}
        render={({ field }) => (
          <Checkbox
            componentId={`${COMPONENT_PREFIX}.categorical.multi-select`}
            isChecked={field.value}
            onChange={(checked) => field.onChange(checked)}
          >
            <FormattedMessage
              defaultMessage="Allow multiple selections (multi-select)"
              description="Categorical multi-select checkbox"
            />
          </Checkbox>
        )}
      />
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
