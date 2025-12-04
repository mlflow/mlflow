import { useEffect } from 'react';
import { useParams, Link, useNavigate } from '../../common/utils/RoutingUtils';
import {
  Alert,
  Breadcrumb,
  Button,
  Input,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  FormUI,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import GatewayRoutes from '../routes';
import { useModelDefinitionQuery } from '../hooks/useModelDefinitionQuery';
import { useUpdateModelDefinitionMutation } from '../hooks/useUpdateModelDefinitionMutation';
import { useCreateSecretMutation } from '../hooks/useCreateSecretMutation';
import { LongFormSection } from '../../common/components/long-form';
import { formatProviderName } from '../utils/providerUtils';
import { SecretConfigSection, type SecretMode } from '../components/secrets/SecretConfigSection';

const LONG_FORM_TITLE_WIDTH = 200;

interface EditModelDefinitionFormData {
  name: string;
  secretMode: SecretMode;
  existingSecretId: string;
  newSecret: {
    name: string;
    value: string;
    authConfig: Record<string, string>;
  };
}

const EditModelDefinitionPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const { modelDefinitionId } = useParams<{ modelDefinitionId: string }>();

  const { data, isLoading: isLoadingModel, error: loadError } = useModelDefinitionQuery(modelDefinitionId ?? '');
  const modelDefinition = data?.model_definition;

  const {
    mutateAsync: updateModelDefinition,
    error: updateError,
    isLoading: isUpdating,
    reset: resetUpdateError,
  } = useUpdateModelDefinitionMutation();
  const {
    mutateAsync: createSecret,
    error: createSecretError,
    isLoading: isCreatingSecret,
    reset: resetSecretError,
  } = useCreateSecretMutation();

  const form = useForm<EditModelDefinitionFormData>({
    defaultValues: {
      name: '',
      secretMode: 'existing',
      existingSecretId: '',
      newSecret: {
        name: '',
        value: '',
        authConfig: {},
      },
    },
  });

  // Reset form when model definition data loads
  useEffect(() => {
    if (modelDefinition) {
      form.reset({
        name: modelDefinition.name,
        secretMode: 'existing',
        existingSecretId: modelDefinition.secret_id,
        newSecret: {
          name: '',
          value: '',
          authConfig: {},
        },
      });
    }
  }, [modelDefinition, form]);

  const resetErrors = () => {
    resetUpdateError();
    resetSecretError();
  };

  const isLoading = isUpdating || isCreatingSecret;
  const mutationError = (updateError || createSecretError) as Error | null;

  const name = form.watch('name');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretValue = form.watch('newSecret.value');

  // Check if the form has valid secret configuration
  const isSecretConfigured =
    secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && Boolean(newSecretValue);

  // Check if there are changes
  const hasChanges =
    modelDefinition &&
    (name !== modelDefinition.name ||
      (secretMode === 'existing' && existingSecretId !== modelDefinition.secret_id) ||
      secretMode === 'new');

  const handleSubmit = async (values: EditModelDefinitionFormData) => {
    if (!modelDefinition) return;

    try {
      // Determine the secret ID to use
      let secretId = values.existingSecretId;
      if (values.secretMode === 'new') {
        const authConfigJson =
          Object.keys(values.newSecret.authConfig).length > 0 ? JSON.stringify(values.newSecret.authConfig) : undefined;

        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: values.newSecret.value,
          provider: modelDefinition.provider,
          auth_config_json: authConfigJson,
        });

        secretId = (secretResponse as { secret: { secret_id: string } }).secret.secret_id;
      }

      // Update model definition
      await updateModelDefinition({
        model_definition_id: modelDefinition.model_definition_id,
        name: values.name !== modelDefinition.name ? values.name : undefined,
        secret_id: secretId !== modelDefinition.secret_id ? secretId : undefined,
      });

      navigate(GatewayRoutes.getModelDefinitionDetailsRoute(modelDefinition.model_definition_id));
    } catch {
      // Error is captured by the mutation's error state
    }
  };

  const handleCancel = () => {
    navigate(GatewayRoutes.getModelDefinitionDetailsRoute(modelDefinitionId ?? ''));
  };

  if (isLoadingModel) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          <Spinner size="small" />
          <FormattedMessage defaultMessage="Loading model..." description="Loading message for model" />
        </div>
      </div>
    );
  }

  if (loadError || !modelDefinition) {
    return (
      <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="mlflow.gateway.edit-model-definition.error"
            type="error"
            message={(loadError as Error | null)?.message ?? 'Model not found'}
          />
        </div>
      </div>
    );
  }

  return (
    <div css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <FormProvider {...form}>
        <div css={{ padding: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.gatewayPageRoute}>
                <FormattedMessage defaultMessage="Gateway" description="Breadcrumb link to gateway page" />
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.modelDefinitionsPageRoute}>
                <FormattedMessage defaultMessage="Models" description="Breadcrumb link to models page" />
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link to={GatewayRoutes.getModelDefinitionDetailsRoute(modelDefinitionId ?? '')}>
                {modelDefinition.name}
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="Edit model" description="Page title for edit model" />
          </Typography.Title>
        </div>

        {mutationError && (
          <div css={{ padding: `0 ${theme.spacing.md}px` }}>
            <Alert
              componentId="mlflow.gateway.edit-model-definition.mutation-error"
              closable={false}
              message={mutationError.message}
              type="error"
              css={{ marginBottom: theme.spacing.md }}
            />
          </div>
        )}

        <div
          css={{
            flex: 1,
            padding: `0 ${theme.spacing.md}px`,
            overflow: 'auto',
            maxWidth: 900,
          }}
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
                  <FormUI.Label htmlFor="mlflow.gateway.edit-model-definition.name">
                    <FormattedMessage defaultMessage="Name" description="Label for model name input" />
                  </FormUI.Label>
                  <Input
                    id="mlflow.gateway.edit-model-definition.name"
                    componentId="mlflow.gateway.edit-model-definition.name"
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      resetErrors();
                    }}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Enter model name',
                      description: 'Model name placeholder',
                    })}
                    disabled={isLoading}
                    validationState={fieldState.error ? 'error' : undefined}
                  />
                  {fieldState.error && <FormUI.Message type="error" message={fieldState.error.message} />}
                </div>
              )}
            />
          </LongFormSection>

          {/* Model Configuration Section (read-only) */}
          <LongFormSection
            titleWidth={LONG_FORM_TITLE_WIDTH}
            title={intl.formatMessage({
              defaultMessage: 'Model',
              description: 'Section title for model configuration',
            })}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <div>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Provider" description="Provider label" />
                </FormUI.Label>
                <Typography.Text>{formatProviderName(modelDefinition.provider)}</Typography.Text>
              </div>
              <div>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Provider model" description="Provider model label" />
                </FormUI.Label>
                <Typography.Text css={{ fontFamily: 'monospace' }}>{modelDefinition.model_name}</Typography.Text>
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, display: 'block', marginTop: theme.spacing.xs }}
                >
                  <FormattedMessage
                    defaultMessage="The provider and model cannot be changed. Create a new model if you need a different configuration."
                    description="Info text about model being read-only"
                  />
                </Typography.Text>
              </div>
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
              provider={modelDefinition.provider}
              mode={secretMode}
              onModeChange={(mode) => form.setValue('secretMode', mode)}
              selectedSecretId={existingSecretId}
              onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
              newSecretFieldPrefix="newSecret"
              componentIdPrefix="mlflow.gateway.edit-model-definition"
            />
          </LongFormSection>
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
          <Button
            componentId="mlflow.gateway.edit-model-definition.cancel"
            onClick={handleCancel}
            disabled={isLoading}
          >
            <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
          </Button>
          <Tooltip
            componentId="mlflow.gateway.edit-model-definition.save-tooltip"
            content={
              !isSecretConfigured
                ? intl.formatMessage({
                    defaultMessage: 'Please configure authentication',
                    description: 'Tooltip shown when save button is disabled',
                  })
                : undefined
            }
          >
            <Button
              componentId="mlflow.gateway.edit-model-definition.save"
              type="primary"
              onClick={form.handleSubmit(handleSubmit)}
              loading={isLoading}
              disabled={!hasChanges || !name.trim() || !isSecretConfigured}
            >
              <FormattedMessage defaultMessage="Save" description="Save button" />
            </Button>
          </Tooltip>
        </div>
      </FormProvider>
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, EditModelDefinitionPage);
