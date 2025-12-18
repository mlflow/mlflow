import { useState } from 'react';
import { Alert, Button, FormUI, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GatewayInput } from '../common';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, useFormContext } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelectorModal } from '../model-selector';
import { LongFormSection, LongFormSummary } from '../../../common/components/long-form';
import { formatProviderName } from '../../utils/providerUtils';
import { formatTokens, formatCost } from '../../utils/formatters';
import type { Model } from '../../types';

const LONG_FORM_TITLE_WIDTH = 200;

export interface EndpointFormData {
  name: string;
  provider: string;
  modelName: string;
}

export interface EndpointFormRendererProps {
  mode: 'create' | 'edit';
  isSubmitting: boolean;
  error: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  selectedModel: Model | undefined;
  isFormComplete: boolean;
  hasChanges?: boolean;
  onSubmit: (values: EndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
  componentIdPrefix?: string;
}

export const EndpointFormRenderer = ({
  mode,
  isSubmitting,
  error,
  errorMessage,
  resetErrors,
  selectedModel,
  isFormComplete,
  hasChanges = true,
  onSubmit,
  onCancel,
  onNameBlur,
  componentIdPrefix = `mlflow.gateway.${mode}-endpoint`,
}: EndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const form = useFormContext<EndpointFormData>();
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');

  const isButtonDisabled = mode === 'edit' ? !isFormComplete || !hasChanges : !isFormComplete;
  const buttonTooltip = !isFormComplete
    ? intl.formatMessage({
        defaultMessage: 'Please complete all required fields',
        description: 'Tooltip shown when submit button is disabled due to incomplete form',
      })
    : mode === 'edit' && !hasChanges
    ? intl.formatMessage({
        defaultMessage: 'No changes to save',
        description: 'Tooltip shown when save button is disabled due to no changes',
      })
    : undefined;

  return (
    <>
      {error && (
        <div css={{ padding: `0 ${theme.spacing.md}px` }}>
          <Alert
            componentId={`${componentIdPrefix}.error`}
            closable={false}
            message={errorMessage}
            type="error"
            css={{ marginBottom: theme.spacing.md }}
          />
        </div>
      )}

      <div
        css={{
          flex: 1,
          display: 'flex',
          gap: theme.spacing.md,
          padding: `0 ${theme.spacing.md}px`,
          overflow: 'auto',
          '@media (max-width: 1023px)': {
            flexDirection: 'column',
          },
        }}
      >
        {/* Main form column */}
        <div
          css={{
            flexGrow: 1,
            maxWidth: 900,
            minWidth: 0,
            '@media (max-width: 1023px)': {
              maxWidth: '100%',
            },
          }}
        >
          {/* Name Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Name',
              description: 'Section title for endpoint name',
            })}
          >
            <Controller
              control={form.control}
              name="name"
              rules={{ required: 'Name is required' }}
              render={({ field, fieldState }) => (
                <div>
                  <GatewayInput
                    id={`${componentIdPrefix}.name`}
                    componentId={`${componentIdPrefix}.name`}
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      form.clearErrors('name');
                      resetErrors();
                    }}
                    onBlur={() => {
                      field.onBlur();
                      onNameBlur();
                    }}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'my-endpoint',
                      description: 'Placeholder for endpoint name input',
                    })}
                    validationState={fieldState.error ? 'error' : undefined}
                  />
                  {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
                </div>
              )}
            />
          </LongFormSection>

          {/* Model Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Model',
              description: 'Section title for model configuration',
            })}
            hideDivider
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              {/* Provider subsection */}
              <div>
                <Controller
                  control={form.control}
                  name="provider"
                  rules={{ required: 'Provider is required' }}
                  render={({ field, fieldState }) => (
                    <ProviderSelect
                      value={field.value}
                      onChange={(value) => {
                        field.onChange(value);
                        form.setValue('modelName', '');
                      }}
                      error={fieldState.error?.message}
                      componentIdPrefix={`${componentIdPrefix}.provider`}
                    />
                  )}
                />
              </div>

              {/* Model subsection - only show when provider is selected */}
              {provider && (
                <div>
                  <FormUI.Label htmlFor={`${componentIdPrefix}.model`}>
                    <FormattedMessage defaultMessage="Model" description="Model label" />
                  </FormUI.Label>
                  <Button
                    componentId={`${componentIdPrefix}.select-model`}
                    onClick={() => setIsModelSelectorOpen(true)}
                    css={{
                      width: '100%',
                      fontWeight: 'normal',
                      '& > span': {
                        width: '100%',
                        justifyContent: 'flex-start',
                      },
                    }}
                  >
                    {modelName || (
                      <span css={{ color: theme.colors.textSecondary }}>
                        <FormattedMessage
                          defaultMessage="Select a model..."
                          description="Model selection placeholder"
                        />
                      </span>
                    )}
                  </Button>
                </div>
              )}
            </div>
          </LongFormSection>
        </div>

        {/* Summary sidebar */}
        <div
          css={{
            flexShrink: 0,
            width: 360,
            position: 'sticky',
            top: 0,
            alignSelf: 'flex-start',
            '@media (max-width: 1023px)': {
              width: '100%',
              position: 'static',
            },
          }}
        >
          <LongFormSummary
            title={intl.formatMessage({
              defaultMessage: 'Summary',
              description: 'Summary sidebar title',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              {/* Provider */}
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="Provider" description="Summary provider label" />
                </Typography.Text>
                {provider ? (
                  <Typography.Text>{formatProviderName(provider)}</Typography.Text>
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                  </Typography.Text>
                )}
              </div>

              {/* Model */}
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="Model" description="Summary model label" />
                </Typography.Text>
                {modelName ? (
                  <ModelSummary model={selectedModel} modelName={modelName} />
                ) : (
                  <Typography.Text color="secondary">
                    <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
                  </Typography.Text>
                )}
              </div>
            </div>
          </LongFormSummary>
        </div>
      </div>

      {/* Footer buttons */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: theme.spacing.sm,
          padding: theme.spacing.md,
          borderTop: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <Button componentId={`${componentIdPrefix}.cancel`} onClick={onCancel}>
          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
        </Button>
        <Tooltip componentId={`${componentIdPrefix}.submit-tooltip`} content={buttonTooltip}>
          <Button
            componentId={`${componentIdPrefix}.submit`}
            type="primary"
            onClick={form.handleSubmit(onSubmit)}
            loading={isSubmitting}
            disabled={isButtonDisabled}
          >
            {mode === 'create' ? (
              <FormattedMessage defaultMessage="Create" description="Create button" />
            ) : (
              <FormattedMessage defaultMessage="Save changes" description="Save changes button" />
            )}
          </Button>
        </Tooltip>
      </div>

      <ModelSelectorModal
        isOpen={isModelSelectorOpen}
        onClose={() => setIsModelSelectorOpen(false)}
        onSelect={(model) => {
          form.setValue('modelName', model.model);
          setIsModelSelectorOpen(false);
        }}
        provider={provider}
      />
    </>
  );
};

/** Helper component to display model metadata in the summary */
const ModelSummary = ({ model, modelName }: { model: Model | undefined; modelName: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const capabilities: string[] = [];
  if (model?.supports_function_calling) capabilities.push('Tools');
  if (model?.supports_reasoning) capabilities.push('Reasoning');

  const contextWindow = formatTokens(model?.max_input_tokens);
  const inputCost = formatCost(model?.input_cost_per_token);
  const outputCost = formatCost(model?.output_cost_per_token);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 0, maxWidth: '100%' }}>
      {/* Model name */}
      <Typography.Text
        bold
        css={{
          fontSize: theme.typography.fontSizeSm,
          wordBreak: 'break-all',
          overflowWrap: 'anywhere',
        }}
      >
        {modelName}
      </Typography.Text>

      {/* Capabilities */}
      {capabilities.length > 0 && (
        <Typography.Text
          color="secondary"
          css={{ fontSize: theme.typography.fontSizeSm, marginLeft: theme.spacing.sm }}
        >
          {capabilities.join(', ')}
        </Typography.Text>
      )}

      {/* Context & Cost info */}
      {model && (contextWindow || inputCost || outputCost) && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            marginLeft: theme.spacing.sm,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {contextWindow && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Max input: {tokens}', description: 'Max input tokens' },
                { tokens: contextWindow },
              )}
            </span>
          )}
          {(inputCost || outputCost) && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Cost: {input} in / {output} out', description: 'Model cost per token' },
                { input: inputCost ?? '-', output: outputCost ?? '-' },
              )}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
