import { Link } from '../../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  FormUI,
  Input,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Controller, FormProvider, UseFormReturn } from 'react-hook-form';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { SecretConfigSection, type SecretMode } from '../secrets/SecretConfigSection';
import { EndpointSummary } from '../endpoints/EndpointSummary';
import GatewayRoutes from '../../routes';
import { LongFormLayout, LongFormSection } from '../../../common/components/long-form';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';
import type { Model } from '../../types';

const LONG_FORM_TITLE_WIDTH = 200;

export interface EditEndpointFormRendererProps {
  form: UseFormReturn<EditEndpointFormData>;
  isLoadingEndpoint: boolean;
  isSubmitting: boolean;
  loadError: Error | null;
  mutationError: Error | null;
  errorMessage: string | null;
  resetErrors: () => void;
  endpointId: string;
  endpointName: string | undefined;
  selectedModel?: Model;
  selectedSecretName?: string;
  isFormComplete: boolean;
  onSubmit: (values: EditEndpointFormData) => Promise<void>;
  onCancel: () => void;
  onNameBlur: () => void;
}

/**
 * Pure presentational component for the edit endpoint form.
 * All business logic is handled by the container (useEditEndpointForm hook).
 */
export const EditEndpointFormRenderer = ({
  form,
  isLoadingEndpoint,
  isSubmitting,
  loadError,
  mutationError,
  errorMessage,
  resetErrors,
  endpointId,
  endpointName,
  selectedModel,
  selectedSecretName,
  isFormComplete,
  onSubmit,
  onCancel,
  onNameBlur,
}: EditEndpointFormRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');

  // Loading state
  if (isLoadingEndpoint) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading endpoint..." description="Loading message for endpoint" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  // Error state
  if (loadError) {
    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-endpoint.error"
            type="error"
            message={loadError.message ?? 'Endpoint not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <FormProvider {...form}>
        <div css={{ padding: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.gatewayPageRoute}>
                <FormattedMessage defaultMessage="Gateway" description="Breadcrumb link to gateway page" />
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.getEndpointDetailsRoute(endpointId)}>{endpointName ?? endpointId}</Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Edit endpoint" description="Page title for edit endpoint" />
          </Typography.Title>
        </div>

        {mutationError && (
          <div css={{ padding: `0 ${theme.spacing.md}px` }}>
            <Alert
              componentId="mlflow.gateway.edit-endpoint.mutation-error"
              closable={false}
              message={errorMessage}
              type="error"
              css={{ marginBottom: theme.spacing.md }}
            />
          </div>
        )}

        <LongFormLayout
          sidebar={
            <EndpointSummary
              provider={provider}
              modelName={modelName}
              modelMetadata={selectedModel}
              selectedSecretName={selectedSecretName}
              showConnection
              connectionMode={secretMode === 'new' ? 'new' : 'existing'}
              componentIdPrefix="mlflow.gateway.edit-endpoint.summary"
            />
          }
        >
          {/* General Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'General',
              description: 'Section title for general settings',
            })}
          >
            <Controller
              control={form.control}
              name="name"
              render={({ field, fieldState }) => (
                <div>
                  <FormUI.Label htmlFor="mlflow.gateway.edit-endpoint.name">
                    <FormattedMessage defaultMessage="Name" description="Label for endpoint name input" />
                  </FormUI.Label>
                  <Input
                    id="mlflow.gateway.edit-endpoint.name"
                    componentId="mlflow.gateway.edit-endpoint.name"
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
              description: 'Section title for model selection',
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
                      form.setValue('modelName', '');
                      form.setValue('existingSecretId', '');
                    }}
                    error={fieldState.error?.message}
                    componentIdPrefix="mlflow.gateway.edit-endpoint.provider"
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
                    componentIdPrefix="mlflow.gateway.edit-endpoint.model"
                  />
                )}
              />
            </div>
          </LongFormSection>

          {/* Authentication Section */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Authentication',
              description: 'Section title for authentication',
            })}
            hideDivider
          >
            <SecretConfigSection
              provider={provider}
              mode={secretMode}
              onModeChange={(mode) => form.setValue('secretMode', mode)}
              selectedSecretId={form.watch('existingSecretId')}
              onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
              newSecretFieldPrefix="newSecret"
              componentIdPrefix="mlflow.gateway.edit-endpoint"
            />
          </LongFormSection>
        </LongFormLayout>

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
          <Button componentId="mlflow.gateway.edit-endpoint.cancel" onClick={onCancel}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.edit-endpoint.save-tooltip"
            content={
              !isFormComplete
                ? intl.formatMessage({
                    defaultMessage: 'Please select a provider, model, and configure authentication',
                    description: 'Tooltip shown when save button is disabled',
                  })
                : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.edit-endpoint.save"
              type="primary"
              onClick={form.handleSubmit(onSubmit)}
              loading={isSubmitting}
              disabled={!isFormComplete}
            >
              <FormattedMessage defaultMessage="Save" description="Save button" />
            </Button>
          </Tooltip>
        </div>
      </FormProvider>
    </ScrollablePageWrapper>
  );
};
