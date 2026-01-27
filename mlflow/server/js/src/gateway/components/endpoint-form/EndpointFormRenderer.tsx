import { useMemo, useCallback } from 'react';
import { Alert, Button, FormUI, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GatewayInput } from '../common';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, useFormContext } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { ApiKeyConfigurator } from '../model-configuration/components/ApiKeyConfigurator';
import { useApiKeyConfiguration } from '../model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration, SecretMode } from '../model-configuration/types';
import { formatProviderName } from '../../utils/providerUtils';
import { LongFormSection } from '../../../common/components/long-form/LongFormSection';
import { LongFormSummary } from '../../../common/components/long-form/LongFormSummary';
import type { ProviderModel, SecretInfo } from '../../types';
import { formatTokens, formatCost } from '../../utils/formatters';
import type { CreateEndpointFormData } from '../../hooks/useCreateEndpointForm';

const LONG_FORM_TITLE_WIDTH = 200;

export type EndpointFormData = CreateEndpointFormData;

export interface EndpointFormRendererProps {
  /** Whether this is editing an existing endpoint (affects button labels, etc.) */
  mode: 'create' | 'edit';
  /** Whether the form is submitting */
  isSubmitting: boolean;
  /** Error to display */
  error: Error | null;
  /** User-friendly error message */
  errorMessage: string | null;
  /** Callback to reset errors when user makes changes */
  resetErrors: () => void;
  /** The selected model's full metadata (for summary display) */
  selectedModel: ProviderModel | undefined;
  /** Whether all required fields are filled */
  isFormComplete: boolean;
  /** Whether any fields have changed from their initial values (edit mode only) */
  hasChanges?: boolean;
  /** Form submission handler */
  onSubmit: (values: EndpointFormData) => Promise<void>;
  /** Cancel handler */
  onCancel: () => void;
  /** Handler for name field blur (for duplicate checking) */
  onNameBlur: () => void;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
  /** When true, adapts layout for use inside containers like modals */
  embedded?: boolean;
}

/**
 * Unified presentational component for endpoint forms (create and edit).
 * All business logic is handled by the parent hook (useCreateEndpointForm or useEditEndpointForm).
 *
 * This component expects to be wrapped in a FormProvider by the parent.
 * Page-level concerns (breadcrumbs, page wrapper, loading/error states) should be
 * handled by the parent to allow this form to be reused in different contexts
 * (full page, modal, etc.).
 */
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
  embedded = false,
}: EndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const form = useFormContext<EndpointFormData>();

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecret = form.watch('newSecret');

  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({ provider });

  const apiKeyConfig: ApiKeyConfiguration = useMemo(
    () => ({
      mode: secretMode,
      existingSecretId: existingSecretId,
      newSecret: newSecret,
    }),
    [secretMode, existingSecretId, newSecret],
  );

  const handleApiKeyChange = useCallback(
    (config: ApiKeyConfiguration) => {
      if (config.mode !== secretMode) {
        form.setValue('secretMode', config.mode);
      }
      if (config.existingSecretId !== existingSecretId) {
        form.setValue('existingSecretId', config.existingSecretId);
      }
      if (config.newSecret !== newSecret) {
        form.setValue('newSecret', config.newSecret);
      }
    },
    [form, secretMode, existingSecretId, newSecret],
  );

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
        <div css={{ padding: embedded ? 0 : `0 ${theme.spacing.md}px` }}>
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
          padding: embedded ? 0 : `0 ${theme.spacing.md}px`,
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
            css={embedded ? { paddingTop: 0 } : undefined}
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

          {/* Experiment Section (only in create mode) */}
          {mode === 'create' && (
            <LongFormSection
              titleWidth={LONG_FORM_TITLE_WIDTH}
              title={intl.formatMessage({
                defaultMessage: 'Experiment',
                description: 'Section title for experiment configuration',
              })}
            >
              <Controller
                control={form.control}
                name="experimentId"
                render={({ field }) => (
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <GatewayInput
                      id={`${componentIdPrefix}.experiment-id`}
                      componentId={`${componentIdPrefix}.experiment-id`}
                      {...field}
                      placeholder={intl.formatMessage({
                        defaultMessage: 'Leave blank to auto-create',
                        description: 'Placeholder for experiment ID input',
                      })}
                    />
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      <FormattedMessage
                        defaultMessage="Traces from endpoint invocations will be logged to this experiment. If not specified, an experiment will be auto-created."
                        description="Help text for experiment ID input"
                      />
                    </Typography.Text>
                  </div>
                )}
              />
            </LongFormSection>
          )}

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
                      form.setValue('existingSecretId', '');
                      form.setValue('secretMode', 'new');
                      form.setValue('newSecret', {
                        name: '',
                        authMode: '',
                        secretFields: {},
                        configFields: {},
                      });
                    }}
                    error={fieldState.error?.message}
                    componentIdPrefix={`${componentIdPrefix}.provider`}
                  />
                )}
              />
              <Controller
                control={form.control}
                name="modelName"
                rules={{ required: 'Model is required' }}
                render={({ field, fieldState }) => (
                  <ModelSelect
                    provider={provider}
                    value={field.value}
                    onChange={field.onChange}
                    error={fieldState.error?.message}
                    componentIdPrefix={`${componentIdPrefix}.model`}
                  />
                )}
              />

              {/* Connections subsection - nested within Model */}
              {provider && (
                <div css={{ marginTop: theme.spacing.sm }}>
                  <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                    <FormattedMessage
                      defaultMessage="Connections"
                      description="Subsection header for API key configuration"
                    />
                  </Typography.Text>
                  <ApiKeyConfigurator
                    value={apiKeyConfig}
                    onChange={handleApiKeyChange}
                    provider={provider}
                    existingSecrets={existingSecrets}
                    isLoadingSecrets={isLoadingSecrets}
                    authModes={authModes}
                    defaultAuthMode={defaultAuthMode}
                    isLoadingProviderConfig={isLoadingProviderConfig}
                    componentIdPrefix={`${componentIdPrefix}.api-key`}
                  />
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

              {/* API Key */}
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold color="secondary">
                  <FormattedMessage defaultMessage="API Key" description="Summary API key label" />
                </Typography.Text>
                <ApiKeySummary
                  secretMode={secretMode}
                  newSecretName={newSecret?.name}
                  existingSecretId={existingSecretId}
                  existingSecrets={existingSecrets}
                />
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
          padding: embedded ? `${theme.spacing.md}px 0 0 0` : theme.spacing.md,
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
    </>
  );
};

/** Helper component to display model metadata in the summary */
const ModelSummary = ({ model, modelName }: { model: ProviderModel | undefined; modelName: string }) => {
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

/** Helper component to display API key info in the summary */
const ApiKeySummary = ({
  secretMode,
  newSecretName,
  existingSecretId,
  existingSecrets,
}: {
  secretMode: SecretMode;
  newSecretName?: string;
  existingSecretId?: string;
  existingSecrets?: SecretInfo[];
}) => {
  const apiKeyName =
    secretMode === 'new' ? newSecretName : existingSecrets?.find((s) => s.secret_id === existingSecretId)?.secret_name;

  if (!apiKeyName) {
    return (
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
      </Typography.Text>
    );
  }

  return <Typography.Text>{apiKeyName}</Typography.Text>;
};
