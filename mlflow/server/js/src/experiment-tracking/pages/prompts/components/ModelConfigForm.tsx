import { FormUI, RHFControlledComponents, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useFormContext } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';

interface ModelConfigFormProps {
  namePrefix?: string; // For nested forms, e.g., "modelConfig."
}

export const ModelConfigForm = ({ namePrefix = '' }: ModelConfigFormProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const {
    control,
    formState: { errors },
  } = useFormContext();

  const fieldName = (name: string) => `${namePrefix}${name}`;

  const getError = (name: string) => {
    const parts = fieldName(name).split('.');
    let error: any = errors;
    for (const part of parts) {
      error = error?.[part];
    }
    return error;
  };

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
      }}
    >
      <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Model Configuration"
          description="Section header for model configuration in prompt creation"
        />
      </Typography.Title>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label htmlFor={fieldName('modelName')}>
            <FormattedMessage defaultMessage="Model" description="Label for model name input in model config form" />
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={control}
            id={fieldName('modelName')}
            componentId={`mlflow.prompts.model_config.${fieldName('modelName')}`}
            name={fieldName('modelName')}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., gpt-4, claude-3-opus',
              description: 'Placeholder for model name input',
            })}
          />
        </div>

        {/* Sampling Parameters - 2 Column Grid */}
        <div>
          <Typography.Text
            bold
            css={{
              display: 'block',
              marginBottom: theme.spacing.xs,
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
            }}
          >
            <FormattedMessage
              defaultMessage="Sampling Parameters"
              description="Subsection for sampling parameters in model config"
            />
          </Typography.Text>
          <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.sm }}>
            {/* Temperature */}
            <div>
              <FormUI.Label htmlFor={fieldName('temperature')}>
                <FormattedMessage defaultMessage="Temperature" description="Label for temperature input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('temperature')}
                componentId={`mlflow.prompts.model_config.${fieldName('temperature')}`}
                name={fieldName('temperature')}
                validationState={getError('temperature') ? 'error' : undefined}
              />
              {getError('temperature') && <FormUI.Message type="error" message={getError('temperature')?.message} />}
            </div>

            {/* Max Tokens */}
            <div>
              <FormUI.Label htmlFor={fieldName('maxTokens')}>
                <FormattedMessage defaultMessage="Max Tokens" description="Label for max tokens input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('maxTokens')}
                componentId={`mlflow.prompts.model_config.${fieldName('maxTokens')}`}
                name={fieldName('maxTokens')}
                validationState={getError('maxTokens') ? 'error' : undefined}
              />
              {getError('maxTokens') && <FormUI.Message type="error" message={getError('maxTokens')?.message} />}
            </div>

            {/* Top P */}
            <div>
              <FormUI.Label htmlFor={fieldName('topP')}>
                <FormattedMessage defaultMessage="Top P" description="Label for top P input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('topP')}
                componentId={`mlflow.prompts.model_config.${fieldName('topP')}`}
                name={fieldName('topP')}
                validationState={getError('topP') ? 'error' : undefined}
              />
              {getError('topP') && <FormUI.Message type="error" message={getError('topP')?.message} />}
            </div>

            {/* Top K */}
            <div>
              <FormUI.Label htmlFor={fieldName('topK')}>
                <FormattedMessage defaultMessage="Top K" description="Label for top K input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('topK')}
                componentId={`mlflow.prompts.model_config.${fieldName('topK')}`}
                name={fieldName('topK')}
                validationState={getError('topK') ? 'error' : undefined}
              />
              {getError('topK') && <FormUI.Message type="error" message={getError('topK')?.message} />}
            </div>

            {/* Frequency Penalty */}
            <div>
              <FormUI.Label htmlFor={fieldName('frequencyPenalty')}>
                <FormattedMessage defaultMessage="Frequency Penalty" description="Label for frequency penalty input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('frequencyPenalty')}
                componentId={`mlflow.prompts.model_config.${fieldName('frequencyPenalty')}`}
                name={fieldName('frequencyPenalty')}
                validationState={getError('frequencyPenalty') ? 'error' : undefined}
              />
              {getError('frequencyPenalty') && (
                <FormUI.Message type="error" message={getError('frequencyPenalty')?.message} />
              )}
            </div>

            {/* Presence Penalty */}
            <div>
              <FormUI.Label htmlFor={fieldName('presencePenalty')}>
                <FormattedMessage defaultMessage="Presence Penalty" description="Label for presence penalty input" />
              </FormUI.Label>
              <RHFControlledComponents.Input
                control={control}
                id={fieldName('presencePenalty')}
                componentId={`mlflow.prompts.model_config.${fieldName('presencePenalty')}`}
                name={fieldName('presencePenalty')}
                validationState={getError('presencePenalty') ? 'error' : undefined}
              />
              {getError('presencePenalty') && (
                <FormUI.Message type="error" message={getError('presencePenalty')?.message} />
              )}
            </div>
          </div>
        </div>

        {/* Stop Sequences */}
        <div>
          <FormUI.Label htmlFor={fieldName('stopSequences')}>
            <FormattedMessage
              defaultMessage="Stop Sequences (comma-separated)"
              description="Label for stop sequences input"
            />
          </FormUI.Label>
          <RHFControlledComponents.Input
            control={control}
            id={fieldName('stopSequences')}
            componentId={`mlflow.prompts.model_config.${fieldName('stopSequences')}`}
            name={fieldName('stopSequences')}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., \\n\\n, END, ###',
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
