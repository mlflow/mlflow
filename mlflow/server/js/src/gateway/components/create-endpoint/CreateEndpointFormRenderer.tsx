import { Link } from '../../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Input,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, FormProvider, UseFormReturn } from 'react-hook-form';
import { ProviderSelect } from './ProviderSelect';
import { ModelSelect } from './ModelSelect';
import { SecretConfigSection } from '../secrets/SecretConfigSection';
import GatewayRoutes from '../../routes';
import { formatProviderName } from '../../utils/providerUtils';
import { LongFormSection, LongFormSummary } from '../../../common/components/long-form';
import type { Model } from '../../types';
import type { CreateEndpointFormData } from '../../hooks/useCreateEndpointForm';
import { formatTokens, formatCost } from '../../utils/formatters';

const LONG_FORM_TITLE_WIDTH = 200;

export interface CreateEndpointFormRendererProps {
  form: UseFormReturn<CreateEndpointFormData>;
  isLoading: boolean;
  error: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  selectedModel?: Model;
  isFormComplete: boolean;
  onSubmit: (values: CreateEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
}

/**
 * Pure presentational component for the create endpoint form.
 * All business logic is handled by the container (useCreateEndpointForm hook).
 */
export const CreateEndpointFormRenderer = ({
  form,
  isLoading,
  error,
  errorMessage,
  resetErrors,
  selectedModel,
  isFormComplete,
  onSubmit,
  onCancel,
  onNameBlur,
}: CreateEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <FormProvider {...form}>
        <div css={{ padding: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.gatewayPageRoute}>
                <FormattedMessage defaultMessage="AI Gateway" description="Breadcrumb link to gateway page" />
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.gatewayPageRoute}>
                <FormattedMessage defaultMessage="Endpoints" description="Breadcrumb link to endpoints list" />
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Create endpoint" description="Page title for create endpoint" />
          </Typography.Title>
          <div
            css={{
              marginTop: theme.spacing.md,
              borderBottom: `1px solid ${theme.colors.border}`,
            }}
          />
        </div>

        {error && (
          <div css={{ padding: `0 ${theme.spacing.md}px` }}>
            <Alert
              componentId="mlflow.gateway.create-endpoint.error"
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
            // Stack vertically on narrow screens
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
                    <Input
                      id="mlflow.gateway.create-endpoint.name"
                      componentId="mlflow.gateway.create-endpoint.name"
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
                        // Reset all dependent fields when provider changes
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
                    />
                  )}
                />
              </div>
            </LongFormSection>

            {/* Connections Section */}
            <LongFormSection
              titleWidth={LONG_FORM_TITLE_WIDTH}
              title={intl.formatMessage({
                defaultMessage: 'Connections',
                description: 'Section title for authentication',
              })}
            >
              <SecretConfigSection
                provider={provider}
                mode={secretMode}
                onModeChange={(mode) => form.setValue('secretMode', mode)}
                selectedSecretId={form.watch('existingSecretId')}
                onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
                newSecretFieldPrefix="newSecret"
                label=""
              />
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
                    <Tag componentId="mlflow.gateway.create-endpoint.summary.provider">
                      {formatProviderName(provider)}
                    </Tag>
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
                  <Typography.Text>
                    {secretMode === 'new' ? (
                      <FormattedMessage defaultMessage="New secret" description="Summary new secret" />
                    ) : (
                      <FormattedMessage defaultMessage="Existing secret" description="Summary existing secret" />
                    )}
                  </Typography.Text>
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
          <Button componentId="mlflow.gateway.create-endpoint.cancel" onClick={onCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.create-endpoint.submit-tooltip"
            content={
              !isFormComplete
                ? intl.formatMessage({
                    defaultMessage: 'Please complete all required fields',
                    description: 'Tooltip shown when create button is disabled',
                  })
                : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.create-endpoint.submit"
              type="primary"
              onClick={form.handleSubmit(onSubmit)}
              loading={isLoading}
              disabled={!isFormComplete}
            >
              <FormattedMessage defaultMessage="Create" description="Create button" />
            </Button>
          </Tooltip>
        </div>
      </FormProvider>
    </ScrollablePageWrapper>
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
      {/* Model name - styled div for proper text wrapping (Tag doesn't support wrapping) */}
      <div
        css={{
          backgroundColor: theme.colors.tagDefault,
          padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
          borderRadius: theme.borders.borderRadiusMd,
          fontSize: theme.typography.fontSizeSm,
          wordBreak: 'break-all',
          overflowWrap: 'anywhere',
        }}
      >
        {modelName}
      </div>

      {/* Capabilities */}
      {capabilities.length > 0 && (
        <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
          {capabilities.map((cap) => (
            <Tag key={cap} color="turquoise" componentId={`mlflow.gateway.create-endpoint.summary.capability.${cap}`}>
              {cap}
            </Tag>
          ))}
        </div>
      )}

      {/* Context & Cost info */}
      {model && (contextWindow !== '-' || inputCost !== '-' || outputCost !== '-') && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            marginTop: theme.spacing.xs,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {contextWindow !== '-' && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                { tokens: contextWindow },
              )}
            </span>
          )}
          {(inputCost !== '-' || outputCost !== '-') && (
            <span>
              {intl.formatMessage(
                { defaultMessage: 'Cost: {input} in / {output} out', description: 'Model cost per token' },
                { input: inputCost, output: outputCost },
              )}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
