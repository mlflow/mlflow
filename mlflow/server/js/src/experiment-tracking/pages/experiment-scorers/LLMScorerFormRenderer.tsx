import React, { useEffect } from 'react';
import type { Control, UseFormSetValue, UseFormGetValues } from 'react-hook-form';
import { Controller, useFormContext, useWatch } from 'react-hook-form';
import {
  useDesignSystemTheme,
  Typography,
  Input,
  FormUI,
  Button,
  DropdownMenu,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxAddButton,
  DialogComboboxFooter,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxHintRow,
  SparkleDoubleIcon,
  DialogComboboxTrigger,
  PlusIcon,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useTemplateOptions, validateInstructions } from './llmScorerUtils';
import { type SCORER_TYPE, ScorerEvaluationScope } from './constants';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE, DEFAULT_LLM_MODEL } from './constants';
import { LLM_TEMPLATE } from './types';
import { TEMPLATE_INSTRUCTIONS_MAP, EDITABLE_TEMPLATES } from './prompts';
import EvaluateTracesSectionRenderer from './EvaluateTracesSectionRenderer';
import { ScorerFormData } from './utils/scorerTransformUtils';

// Form data type that matches LLMScorer structure
export interface LLMScorerFormData extends Pick<ScorerFormData, 'evaluationScope'> {
  llmTemplate: string;
  name: string;
  sampleRate: number;
  filterString?: string;
  scorerType: typeof SCORER_TYPE.LLM;
  guidelines?: string;
  instructions?: string;
  model?: string;
  disableMonitoring?: boolean;
  isInstructionsJudge?: boolean;
}

interface LLMScorerFormRendererProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
  setValue: UseFormSetValue<LLMScorerFormData>;
  getValues: UseFormGetValues<LLMScorerFormData>;
}

interface LLMTemplateSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
  setValue: UseFormSetValue<LLMScorerFormData>;
  currentTemplate: string;
}

const LLMTemplateSection: React.FC<LLMTemplateSectionProps> = ({ mode, control, setValue, currentTemplate }) => {
  const { theme } = useDesignSystemTheme();
  const { watch } = useFormContext<ScorerFormData>();
  const scope = watch('evaluationScope');
  const intl = useIntl();
  const { templateOptions, displayMap } = useTemplateOptions(scope);

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const handleTemplateChange = (newTemplate: string) => {
    const instructions = TEMPLATE_INSTRUCTIONS_MAP[newTemplate] || '';
    const isInstructionsJudge = EDITABLE_TEMPLATES.has(newTemplate);
    setValue('isInstructionsJudge', isInstructionsJudge);
    setValue('instructions', instructions, { shouldValidate: isInstructionsJudge });
  };

  const isReadOnly = mode !== SCORER_FORM_MODE.CREATE;

  // Don't show template selector for custom LLM judges in non-create mode
  if (isReadOnly && currentTemplate === LLM_TEMPLATE.CUSTOM) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label aria-required={!isReadOnly} htmlFor="mlflow-experiment-scorers-built-in-scorer">
        <FormattedMessage defaultMessage="LLM template" description="Section header for LLM template selection" />
      </FormUI.Label>
      {!isReadOnly && (
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Start with a built-in LLM judge template or create your own."
            description="Hint text for LLM template selection with documentation link"
          />
        </FormUI.Hint>
      )}
      <Controller
        name="llmTemplate"
        control={control}
        render={({ field }) => (
          <div css={{ marginTop: '8px' }} onClick={stopPropagationClick}>
            <DialogCombobox
              componentId={`${COMPONENT_ID_PREFIX}.built-in-scorer-select`}
              id="mlflow-experiment-scorers-built-in-scorer"
              value={field.value ? [field.value] : []}
            >
              <DialogComboboxTrigger
                withInlineLabel={false}
                allowClear={false}
                disabled={isReadOnly}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Select an LLM template',
                  description: 'Placeholder for LLM template selection',
                })}
                renderDisplayedValue={(value) => (
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    {value === LLM_TEMPLATE.CUSTOM ? <PlusIcon /> : <SparkleDoubleIcon />}
                    <span>{displayMap[value] || value}</span>
                  </div>
                )}
              />
              {!isReadOnly && (
                <DialogComboboxContent maxHeight={350}>
                  <DialogComboboxOptionList>
                    {templateOptions
                      .filter((option) => option.value !== LLM_TEMPLATE.CUSTOM)
                      .map((option) => (
                        <DialogComboboxOptionListSelectItem
                          key={option.value}
                          value={option.value}
                          onChange={() => {
                            field.onChange(option.value);
                            handleTemplateChange(option.value);
                          }}
                          checked={field.value === option.value}
                          icon={<SparkleDoubleIcon />}
                        >
                          {option.label}
                          <DialogComboboxHintRow>{option.hint}</DialogComboboxHintRow>
                        </DialogComboboxOptionListSelectItem>
                      ))}
                  </DialogComboboxOptionList>
                  <DialogComboboxFooter>
                    <DialogComboboxAddButton
                      onClick={() => {
                        field.onChange(LLM_TEMPLATE.CUSTOM);
                        handleTemplateChange(LLM_TEMPLATE.CUSTOM);
                      }}
                    >
                      {templateOptions.find((option) => option.value === LLM_TEMPLATE.CUSTOM)?.label}
                    </DialogComboboxAddButton>
                  </DialogComboboxFooter>
                </DialogComboboxContent>
              )}
            </DialogCombobox>
          </div>
        )}
      />
    </div>
  );
};

interface NameSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
}

const NameSection: React.FC<NameSectionProps> = ({ mode, control }) => {
  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  if (mode === SCORER_FORM_MODE.DISPLAY) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor="mlflow-experiment-scorers-name">
        <FormattedMessage defaultMessage="Name" description="Section header for optional judge name" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Must be unique in this experiment. Cannot be changed after creation."
          description="Hint text for Name section"
        />
      </FormUI.Hint>
      <Controller
        name="name"
        control={control}
        render={({ field }) => (
          <Input
            {...field}
            componentId={`${COMPONENT_ID_PREFIX}.name-input`}
            id="mlflow-experiment-scorers-name"
            disabled={mode !== SCORER_FORM_MODE.CREATE}
            placeholder="Custom"
            css={{ cursor: mode === SCORER_FORM_MODE.CREATE ? 'text' : 'auto' }}
            onClick={stopPropagationClick}
          />
        )}
      />
    </div>
  );
};

interface InstructionsSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
  setValue: UseFormSetValue<LLMScorerFormData>;
  getValues: UseFormGetValues<LLMScorerFormData>;
}

const InstructionsSection: React.FC<InstructionsSectionProps> = ({ mode, control, setValue, getValues }) => {
  const intl = useIntl();
  const { watch } = useFormContext<ScorerFormData>();
  const scope = watch('evaluationScope');

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const appendVariable = (variable: string) => {
    const currentValue = getValues('instructions') || '';
    setValue('instructions', currentValue + variable, { shouldValidate: true });
  };

  const isInstructionsJudge = useWatch({ control, name: 'isInstructionsJudge' }) ?? false;
  const isReadOnly = mode === SCORER_FORM_MODE.DISPLAY || !isInstructionsJudge;
  const isSessionLevelScorer = scope === ScorerEvaluationScope.SESSIONS;

  const traceLevelTemplateVariables = (
    <>
      <DropdownMenu.Item
        componentId={`${COMPONENT_ID_PREFIX}.add-variable-inputs`}
        onClick={(e) => {
          e.stopPropagation();
          appendVariable('{{ inputs }}');
        }}
      >
        <FormattedMessage defaultMessage="Inputs" description="Label for inputs variable option" />
        <DropdownMenu.HintRow>
          <FormattedMessage defaultMessage="Input for the trace" description="Description for inputs variable" />
        </DropdownMenu.HintRow>
      </DropdownMenu.Item>
      <DropdownMenu.Item
        componentId={`${COMPONENT_ID_PREFIX}.add-variable-outputs`}
        onClick={(e) => {
          e.stopPropagation();
          appendVariable('{{ outputs }}');
        }}
      >
        <FormattedMessage defaultMessage="Outputs" description="Label for outputs variable option" />
        <DropdownMenu.HintRow>
          <FormattedMessage defaultMessage="Output for the trace" description="Description for outputs variable" />
        </DropdownMenu.HintRow>
      </DropdownMenu.Item>
      <DropdownMenu.Item
        componentId={`${COMPONENT_ID_PREFIX}.add-variable-expectations`}
        onClick={(e) => {
          e.stopPropagation();
          appendVariable('{{ expectations }}');
        }}
      >
        <FormattedMessage defaultMessage="Expectations" description="Label for expectations variable option" />
        <DropdownMenu.HintRow>
          <FormattedMessage
            defaultMessage="Expectations added for a trace"
            description="Description for expectations variable"
          />
        </DropdownMenu.HintRow>
      </DropdownMenu.Item>
      <DropdownMenu.Item
        componentId={`${COMPONENT_ID_PREFIX}.add-variable-trace`}
        onClick={(e) => {
          e.stopPropagation();
          appendVariable('{{ trace }}');
        }}
      >
        <FormattedMessage defaultMessage="Trace" description="Label for trace variable option" />
        <DropdownMenu.HintRow>
          <FormattedMessage
            defaultMessage="Full trace with an agent using the right part of the trace to use to judge"
            description="Description for trace variable"
          />
        </DropdownMenu.HintRow>
      </DropdownMenu.Item>
    </>
  );

  const sessionLevelTemplateVariables = (
    <DropdownMenu.Item
      componentId={`${COMPONENT_ID_PREFIX}.add-variable-conversation`}
      onClick={(e) => {
        e.stopPropagation();
        appendVariable('{{ conversation }}');
      }}
    >
      <FormattedMessage defaultMessage="Conversation" description="Label for conversation variable option" />
      <DropdownMenu.HintRow>
        <FormattedMessage
          defaultMessage="Full conversation between a user and an assistant"
          description="Description for conversation variable"
        />
      </DropdownMenu.HintRow>
    </DropdownMenu.Item>
  );

  const templateVariables = isSessionLevelScorer ? sessionLevelTemplateVariables : traceLevelTemplateVariables;

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FormUI.Label htmlFor="mlflow-experiment-scorers-instructions" aria-required={isInstructionsJudge}>
            <FormattedMessage defaultMessage="Instructions" description="Section header for judge instructions" />
          </FormUI.Label>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId={`${COMPONENT_ID_PREFIX}.add-variable-button`}
                size="small"
                endIcon={<ChevronDownIcon />}
                disabled={mode === SCORER_FORM_MODE.DISPLAY || !isInstructionsJudge}
                onClick={stopPropagationClick}
              >
                <FormattedMessage defaultMessage="Add variable" description="Button text for adding variables" />
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="end">{templateVariables}</DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Define custom instructions for LLM-based evaluation. {learnMore}"
            description="Hint text for Instructions section with documentation link"
            values={{
              learnMore: (
                <Typography.Link
                  componentId={`${COMPONENT_ID_PREFIX}.instructions-learn-more-link`}
                  href="https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/make-judge/"
                  openInNewTab
                >
                  <FormattedMessage defaultMessage="Learn more" description="Learn more link text" />
                </Typography.Link>
              ),
            }}
          />
        </FormUI.Hint>
        <Controller
          name="instructions"
          control={control}
          rules={{
            validate: (value) => (isInstructionsJudge ? validateInstructions(value, scope) : true),
          }}
          render={({ field, fieldState }) => {
            const textArea = (
              <Input.TextArea
                {...field}
                componentId={`${COMPONENT_ID_PREFIX}.instructions-text-area`}
                id="mlflow-experiment-scorers-instructions"
                readOnly={isReadOnly}
                rows={7}
                placeholder={
                  isSessionLevelScorer
                    ? intl.formatMessage({
                        defaultMessage:
                          "Evaluate the conversation between a user and an assistant '{{ conversation }}'.",
                        description: 'Placeholder text for session level instructions textarea',
                      })
                    : intl.formatMessage({
                        defaultMessage:
                          "Evaluate if the response in '{{ outputs }}' correctly answers the question in '{{ inputs }}'. The response should be accurate, complete, and professional.",
                        description: 'Example placeholder text for instructions textarea',
                      })
                }
                css={{ resize: 'vertical', cursor: isReadOnly ? 'auto' : 'text' }}
                onClick={stopPropagationClick}
              />
            );

            const showTooltip = !isInstructionsJudge && mode !== SCORER_FORM_MODE.DISPLAY;

            return (
              <>
                {showTooltip ? (
                  <Tooltip
                    componentId={`${COMPONENT_ID_PREFIX}.instructions-readonly-tooltip`}
                    content={intl.formatMessage({
                      defaultMessage: 'Modifying instructions is not supported for this built-in judge.',
                      description: 'Tooltip explaining why instructions are read-only for non-editable templates',
                    })}
                  >
                    <div>{textArea}</div>
                  </Tooltip>
                ) : (
                  textArea
                )}
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            );
          }}
        />
      </div>
    </div>
  );
};

interface GuidelinesSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
  selectedTemplate: string;
}

const GuidelinesSection: React.FC<GuidelinesSectionProps> = ({ mode, control, selectedTemplate }) => {
  const intl = useIntl();

  if (selectedTemplate !== 'Guidelines') {
    return null;
  }

  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const getLlmJudgeDocUrl = () => {
    return 'https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/';
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor="mlflow-experiment-scorers-guidelines" aria-required>
        <FormattedMessage defaultMessage="Guidelines" description="Section header for scorer guidelines" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Add a set of instructions for the scorer. Enter one guideline per line. {learnMore}"
          description="Hint text for Guidelines section with documentation link"
          values={{
            learnMore: (
              <Typography.Link
                componentId={`${COMPONENT_ID_PREFIX}.guidelines-learn-more-link`}
                href={getLlmJudgeDocUrl()}
                openInNewTab
              >
                <FormattedMessage defaultMessage="Learn more" description="Learn more link text" />
              </Typography.Link>
            ),
          }}
        />
      </FormUI.Hint>
      <Controller
        name="guidelines"
        control={control}
        rules={{
          required: selectedTemplate === 'Guidelines',
        }}
        render={({ field }) => (
          <Input.TextArea
            {...field}
            componentId={`${COMPONENT_ID_PREFIX}.guidelines-text-area`}
            id="mlflow-experiment-scorers-guidelines"
            readOnly={mode === SCORER_FORM_MODE.DISPLAY}
            rows={3}
            placeholder={intl.formatMessage({
              defaultMessage: 'The response must be in English',
              description: 'Placeholder text for guidelines textarea',
            })}
            css={{ resize: 'vertical', cursor: mode === SCORER_FORM_MODE.DISPLAY ? 'auto' : 'text' }}
            onClick={stopPropagationClick}
          />
        )}
      />
    </div>
  );
};

interface ModelSectionProps {
  mode: ScorerFormMode;
  control: Control<LLMScorerFormData>;
}

const ModelSection: React.FC<ModelSectionProps> = ({ mode, control }) => {
  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <FormUI.Label htmlFor="mlflow-experiment-scorers-model">
        <FormattedMessage defaultMessage="Model" description="Section header for model input" />
      </FormUI.Label>
      <FormUI.Hint>
        <FormattedMessage
          defaultMessage="Specify the model for LLM evaluation. Defaults to openai:/gpt-4o-mini if not set. {learnMore}"
          description="Hint text for model input with documentation link"
          values={{
            learnMore: (
              <Typography.Link
                componentId={`${COMPONENT_ID_PREFIX}.model-learn-more-link`}
                href="https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/#supported-models"
                openInNewTab
              >
                <FormattedMessage defaultMessage="Learn more" description="Learn more link text" />
              </Typography.Link>
            ),
          }}
        />
      </FormUI.Hint>
      <Controller
        name="model"
        control={control}
        render={({ field }) => (
          <Input
            {...field}
            componentId={`${COMPONENT_ID_PREFIX}.model-input`}
            id="mlflow-experiment-scorers-model"
            disabled={mode === SCORER_FORM_MODE.DISPLAY}
            placeholder={mode === SCORER_FORM_MODE.DISPLAY ? '' : DEFAULT_LLM_MODEL}
            css={{ cursor: mode === SCORER_FORM_MODE.DISPLAY ? 'auto' : 'text' }}
            onClick={stopPropagationClick}
          />
        )}
      />
    </div>
  );
};

const LLMScorerFormRenderer: React.FC<LLMScorerFormRendererProps> = ({ mode, control, setValue, getValues }) => {
  const { theme } = useDesignSystemTheme();
  const selectedTemplate = useWatch({ control, name: 'llmTemplate' });

  // Update name when template changes
  useEffect(() => {
    if (mode === SCORER_FORM_MODE.CREATE && selectedTemplate) {
      setValue('name', selectedTemplate);
    }
  }, [selectedTemplate, setValue, mode]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        paddingLeft: mode === SCORER_FORM_MODE.DISPLAY ? theme.spacing.lg : 0,
      }}
    >
      <LLMTemplateSection mode={mode} control={control} setValue={setValue} currentTemplate={selectedTemplate} />
      <NameSection mode={mode} control={control} />
      <GuidelinesSection mode={mode} control={control} selectedTemplate={selectedTemplate} />
      <InstructionsSection mode={mode} control={control} setValue={setValue} getValues={getValues} />
      <ModelSection mode={mode} control={control} />
      <EvaluateTracesSectionRenderer control={control} mode={mode} />
    </div>
  );
};

export default LLMScorerFormRenderer;
