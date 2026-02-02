import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  InfoSmallIcon,
  Popover,
  RHFControlledComponents,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback } from 'react';
import { useFormContext, Controller } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { useProviderModelData } from '../hooks/useProviderModelData';

export const ModelConfigForm = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    control,
    watch,
    setValue,
    formState: { errors },
  } = useFormContext();

  const getFieldName = useCallback((name: string) => `modelConfig.${name}`, []);

  /**
   * Gets validation error for a model config field.
   * Errors are stored as errors.modelConfig.fieldName by the parent form validation.
   */
  const getError = (name: string) => {
    return (errors?.['modelConfig'] as any)?.[name];
  };

  const selectedProvider = watch(getFieldName('provider'));
  const handleProviderChange = useCallback(() => {
    setValue(getFieldName('modelName'), '');
  }, [setValue, getFieldName]);

  const { providers, providersLoading, models, modelsLoading } = useProviderModelData(
    selectedProvider,
    handleProviderChange,
  );

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginBottom: theme.spacing.sm }}>
        <Typography.Title level={4} css={{ marginBottom: 0 }}>
          <FormattedMessage
            defaultMessage="Model Configuration"
            description="Section header for model configuration in prompt creation"
          />
        </Typography.Title>
        <Popover.Root componentId="mlflow.prompts.model_config.help">
          <Popover.Trigger aria-label="Model configuration help" css={{ border: 0, background: 'none', padding: 0 }}>
            <InfoSmallIcon />
          </Popover.Trigger>
          <Popover.Content align="start">
            <FormattedMessage
              defaultMessage="Model configuration stores the LLM settings associated with this prompt."
              description="Help text explaining model configuration purpose"
            />
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label htmlFor="mlflow.prompts.model_config.provider">
            <FormattedMessage defaultMessage="Provider" description="Label for model provider input" />
          </FormUI.Label>
          <Controller
            name={getFieldName('provider')}
            control={control}
            render={({ field, fieldState }) => (
              <>
                <DialogCombobox
                  componentId="mlflow.prompts.model_config.provider"
                  label={intl.formatMessage({
                    defaultMessage: 'Provider',
                    description: 'Label for model provider input',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompts.model_config.provider"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., openai, anthropic, gemini',
                      description: 'Placeholder for provider input',
                    })}
                    withInlineLabel={false}
                    onClear={() => {
                      field.onChange('');
                      setValue(getFieldName('modelName'), '');
                    }}
                  />
                  <DialogComboboxContent loading={providersLoading} maxHeight={400} matchTriggerWidth>
                    {!providersLoading && providers && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {providers.map((provider) => (
                            <DialogComboboxOptionListSelectItem
                              value={provider}
                              key={provider}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === provider}
                            >
                              {provider}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            )}
          />
        </div>

        <div>
          <FormUI.Label htmlFor="mlflow.prompts.model_config.modelName">
            <FormattedMessage
              defaultMessage="Model Name"
              description="Label for model name input in model config form"
            />
          </FormUI.Label>
          <Controller
            name={getFieldName('modelName')}
            control={control}
            render={({ field, fieldState }) => (
              <>
                <DialogCombobox
                  componentId="mlflow.prompts.model_config.modelName"
                  label={intl.formatMessage({
                    defaultMessage: 'Model Name',
                    description: 'Label for model name input in model config form',
                  })}
                  modal={false}
                  value={field.value ? [field.value] : undefined}
                >
                  <DialogComboboxTrigger
                    id="mlflow.prompts.model_config.modelName"
                    css={{ width: '100%' }}
                    allowClear
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., gpt-5.2, claude-4.5-opus',
                      description: 'Placeholder for model name input',
                    })}
                    withInlineLabel={false}
                    disabled={!selectedProvider}
                    onClear={() => field.onChange('')}
                  />
                  <DialogComboboxContent loading={modelsLoading} maxHeight={400} matchTriggerWidth>
                    {!modelsLoading && models && (
                      <DialogComboboxOptionList>
                        <DialogComboboxOptionListSearch autoFocus>
                          {models.map((model) => (
                            <DialogComboboxOptionListSelectItem
                              value={model.model}
                              key={model.model}
                              onChange={(value) => field.onChange(value)}
                              checked={field.value === model.model}
                            >
                              {model.model}
                            </DialogComboboxOptionListSelectItem>
                          ))}
                        </DialogComboboxOptionListSearch>
                      </DialogComboboxOptionList>
                    )}
                  </DialogComboboxContent>
                </DialogCombobox>
                {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
              </>
            )}
          />
        </div>

        {/* 2 Column Grid for Parameters */}
        <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.sm }}>
          {/* Temperature */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.temperature">
              <FormattedMessage defaultMessage="Temperature" description="Label for temperature input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.temperature"
              componentId="mlflow.prompts.model_config.temperature"
              name={getFieldName('temperature')}
              validationState={getError('temperature') ? 'error' : undefined}
            />
            {getError('temperature') && <FormUI.Message type="error" message={getError('temperature')?.message} />}
          </div>

          {/* Max Tokens */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.maxTokens">
              <FormattedMessage defaultMessage="Max Tokens" description="Label for max tokens input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.maxTokens"
              componentId="mlflow.prompts.model_config.maxTokens"
              name={getFieldName('maxTokens')}
              validationState={getError('maxTokens') ? 'error' : undefined}
            />
            {getError('maxTokens') && <FormUI.Message type="error" message={getError('maxTokens')?.message} />}
          </div>

          {/* Top P */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.topP">
              <FormattedMessage defaultMessage="Top P" description="Label for top P input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.topP"
              componentId="mlflow.prompts.model_config.topP"
              name={getFieldName('topP')}
              validationState={getError('topP') ? 'error' : undefined}
            />
            {getError('topP') && <FormUI.Message type="error" message={getError('topP')?.message} />}
          </div>

          {/* Top K */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.topK">
              <FormattedMessage defaultMessage="Top K" description="Label for top K input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.topK"
              componentId="mlflow.prompts.model_config.topK"
              name={getFieldName('topK')}
              validationState={getError('topK') ? 'error' : undefined}
            />
            {getError('topK') && <FormUI.Message type="error" message={getError('topK')?.message} />}
          </div>

          {/* Frequency Penalty */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.frequencyPenalty">
              <FormattedMessage defaultMessage="Frequency Penalty" description="Label for frequency penalty input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.frequencyPenalty"
              componentId="mlflow.prompts.model_config.frequencyPenalty"
              name={getFieldName('frequencyPenalty')}
              validationState={getError('frequencyPenalty') ? 'error' : undefined}
            />
            {getError('frequencyPenalty') && (
              <FormUI.Message type="error" message={getError('frequencyPenalty')?.message} />
            )}
          </div>

          {/* Presence Penalty */}
          <div>
            <FormUI.Label htmlFor="mlflow.prompts.model_config.presencePenalty">
              <FormattedMessage defaultMessage="Presence Penalty" description="Label for presence penalty input" />
            </FormUI.Label>
            <RHFControlledComponents.Input
              control={control}
              id="mlflow.prompts.model_config.presencePenalty"
              componentId="mlflow.prompts.model_config.presencePenalty"
              name={getFieldName('presencePenalty')}
              validationState={getError('presencePenalty') ? 'error' : undefined}
            />
            {getError('presencePenalty') && (
              <FormUI.Message type="error" message={getError('presencePenalty')?.message} />
            )}
          </div>
        </div>

        {/* Stop Sequences */}
        <div>
          <FormUI.Label htmlFor="mlflow.prompts.model_config.stopSequences">
            <FormattedMessage
              defaultMessage="Stop Sequences (comma-separated)"
              description="Label for stop sequences input"
            />
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={control}
            id="mlflow.prompts.model_config.stopSequences"
            componentId="mlflow.prompts.model_config.stopSequences"
            name={getFieldName('stopSequences')}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., END, ###, STOP',
              description: 'Placeholder for stop sequences input',
            })}
            validationState={getError('stopSequences') ? 'error' : undefined}
          />
          {getError('stopSequences') && <FormUI.Message type="error" message={getError('stopSequences')?.message} />}
        </div>
      </div>
    </div>
  );
};
