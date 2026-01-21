import React from 'react';
import type { Control } from 'react-hook-form';
import { Controller, useWatch } from 'react-hook-form';
import {
  useDesignSystemTheme,
  FormUI,
  Input,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxHintRow,
  DialogComboboxTrigger,
  Typography,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE } from './constants';
import type { LLMScorerFormData } from './LLMScorerFormRenderer';

export interface OutputTypeSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
}

const OUTPUT_TYPE_KIND_OPTIONS = [
  { value: 'default', label: 'Default', hint: 'Let the judge determine the output type automatically' },
  { value: 'bool', label: 'Boolean', hint: 'Yes/no evaluations (pass/fail)' },
  { value: 'int', label: 'Integer', hint: 'Integer ratings (e.g., 1-5 scale)' },
  { value: 'float', label: 'Float', hint: 'Floating point scores (e.g., 0.0-1.0)' },
  { value: 'str', label: 'String', hint: 'Text responses' },
  { value: 'categorical', label: 'Categorical', hint: 'Fixed set of choices (e.g., good/bad/neutral)' },
  { value: 'dict', label: 'Dictionary', hint: 'Key-value pairs (e.g., {"clarity": 4, "accuracy": 5})' },
  { value: 'list', label: 'List', hint: 'List of values (e.g., ["issue1", "issue2"])' },
] as const;

const PRIMITIVE_TYPE_OPTIONS = [
  { value: 'bool', label: 'Boolean' },
  { value: 'int', label: 'Integer' },
  { value: 'float', label: 'Float' },
  { value: 'str', label: 'String' },
] as const;

const OUTPUT_TYPE_KIND_DISPLAY_MAP: Record<string, string> = {
  default: 'Default',
  bool: 'Boolean',
  int: 'Integer',
  float: 'Float',
  str: 'String',
  categorical: 'Categorical',
  dict: 'Dictionary',
  list: 'List',
};

const PRIMITIVE_TYPE_DISPLAY_MAP: Record<string, string> = {
  bool: 'Boolean',
  int: 'Integer',
  float: 'Float',
  str: 'String',
};

const OutputTypeSection: React.FC<OutputTypeSectionProps> = ({ mode, control }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const outputTypeKind = useWatch({ control, name: 'outputTypeKind' }) ?? 'default';

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const isReadOnly = mode === SCORER_FORM_MODE.DISPLAY;
  const showPrimitiveSelector = outputTypeKind === 'dict' || outputTypeKind === 'list';

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {/* Main output type selector with inline primitive type for dict/list */}
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor="mlflow-experiment-scorers-output-type">
          <FormattedMessage defaultMessage="Output type" description="Section header for judge output type selection" />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="The type of value the judge will return."
            description="Hint text for output type selection"
          />
        </FormUI.Hint>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginTop: theme.spacing.xs }}>
          <Controller
            name="outputTypeKind"
            control={control}
            render={({ field }) => (
              <div css={{ flex: showPrimitiveSelector ? undefined : 1 }} onClick={stopPropagationClick}>
                <DialogCombobox
                  componentId={`${COMPONENT_ID_PREFIX}.output-type-select`}
                  id="mlflow-experiment-scorers-output-type"
                  value={[field.value ?? 'default']}
                >
                  <DialogComboboxTrigger
                    withInlineLabel={false}
                    allowClear={false}
                    disabled={isReadOnly}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select output type',
                      description: 'Placeholder for output type selection',
                    })}
                    renderDisplayedValue={(value) => OUTPUT_TYPE_KIND_DISPLAY_MAP[value] || value}
                  />
                  {!isReadOnly && (
                    <DialogComboboxContent maxHeight={350}>
                      <DialogComboboxOptionList>
                        {OUTPUT_TYPE_KIND_OPTIONS.map((option) => (
                          <DialogComboboxOptionListSelectItem
                            key={option.value}
                            value={option.value}
                            onChange={() => field.onChange(option.value)}
                            checked={
                              field.value === option.value || (field.value === undefined && option.value === 'default')
                            }
                          >
                            {option.label}
                            <DialogComboboxHintRow>{option.hint}</DialogComboboxHintRow>
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </DialogComboboxOptionList>
                    </DialogComboboxContent>
                  )}
                </DialogCombobox>
              </div>
            )}
          />

          {outputTypeKind === 'dict' && (
            <>
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="of" description="Connector between dict and value type" />
              </Typography.Text>
              <Controller
                name="dictValueType"
                control={control}
                render={({ field }) => (
                  <div onClick={stopPropagationClick}>
                    <DialogCombobox
                      componentId={`${COMPONENT_ID_PREFIX}.dict-value-type-select`}
                      id="mlflow-experiment-scorers-dict-value-type"
                      value={field.value ? [field.value] : ['int']}
                    >
                      <DialogComboboxTrigger
                        withInlineLabel={false}
                        allowClear={false}
                        disabled={isReadOnly}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'Select value type',
                          description: 'Placeholder for dict value type',
                        })}
                        renderDisplayedValue={(value) => PRIMITIVE_TYPE_DISPLAY_MAP[value] || value}
                      />
                      {!isReadOnly && (
                        <DialogComboboxContent maxHeight={200}>
                          <DialogComboboxOptionList>
                            {PRIMITIVE_TYPE_OPTIONS.map((option) => (
                              <DialogComboboxOptionListSelectItem
                                key={option.value}
                                value={option.value}
                                onChange={() => field.onChange(option.value)}
                                checked={field.value === option.value || (!field.value && option.value === 'int')}
                              >
                                {option.label}
                              </DialogComboboxOptionListSelectItem>
                            ))}
                          </DialogComboboxOptionList>
                        </DialogComboboxContent>
                      )}
                    </DialogCombobox>
                  </div>
                )}
              />
            </>
          )}

          {outputTypeKind === 'list' && (
            <>
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="of" description="Connector between list and element type" />
              </Typography.Text>
              <Controller
                name="listElementType"
                control={control}
                render={({ field }) => (
                  <div onClick={stopPropagationClick}>
                    <DialogCombobox
                      componentId={`${COMPONENT_ID_PREFIX}.list-element-type-select`}
                      id="mlflow-experiment-scorers-list-element-type"
                      value={field.value ? [field.value] : ['str']}
                    >
                      <DialogComboboxTrigger
                        withInlineLabel={false}
                        allowClear={false}
                        disabled={isReadOnly}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'Select element type',
                          description: 'Placeholder for list element type',
                        })}
                        renderDisplayedValue={(value) => PRIMITIVE_TYPE_DISPLAY_MAP[value] || value}
                      />
                      {!isReadOnly && (
                        <DialogComboboxContent maxHeight={200}>
                          <DialogComboboxOptionList>
                            {PRIMITIVE_TYPE_OPTIONS.map((option) => (
                              <DialogComboboxOptionListSelectItem
                                key={option.value}
                                value={option.value}
                                onChange={() => field.onChange(option.value)}
                                checked={field.value === option.value || (!field.value && option.value === 'str')}
                              >
                                {option.label}
                              </DialogComboboxOptionListSelectItem>
                            ))}
                          </DialogComboboxOptionList>
                        </DialogComboboxContent>
                      )}
                    </DialogCombobox>
                  </div>
                )}
              />
            </>
          )}
        </div>
      </div>

      {/* Categorical options input */}
      {outputTypeKind === 'categorical' && (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <FormUI.Label htmlFor="mlflow-experiment-scorers-categorical-options" required>
            <FormattedMessage defaultMessage="Options" description="Label for categorical options input" />
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Enter the allowed values, one per line."
              description="Hint for categorical options"
            />
          </FormUI.Hint>
          <Controller
            name="categoricalOptions"
            control={control}
            rules={{ required: outputTypeKind === 'categorical' }}
            render={({ field, fieldState }) => (
              <>
                <Input.TextArea
                  {...field}
                  componentId={`${COMPONENT_ID_PREFIX}.categorical-options-input`}
                  id="mlflow-experiment-scorers-categorical-options"
                  readOnly={isReadOnly}
                  rows={3}
                  placeholder={'good\nbad\nneutral'}
                  css={{ resize: 'vertical' }}
                  onClick={stopPropagationClick}
                />
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            )}
          />
        </div>
      )}
    </div>
  );
};

export default OutputTypeSection;
