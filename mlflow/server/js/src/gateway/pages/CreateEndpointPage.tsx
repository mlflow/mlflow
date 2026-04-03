import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { FormProvider } from 'react-hook-form';
import { Breadcrumb, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useLocation, useNavigate } from '../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { useCreateEndpointForm } from '../hooks/useCreateEndpointForm';
import { getReadableErrorMessage } from '../utils/errorUtils';
import { EndpointFormRenderer } from '../components/endpoint-form';
import GatewayRoutes from '../routes';
import { GatewayLabel } from '../../common/components/GatewayNewTag';

const CreateEndpointPage = () => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const prefill = parsePrefillState(location.state);

  const {
    form,
    isLoading,
    error,
    resetErrors,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  } = useCreateEndpointForm({
    defaultProvider: prefill?.provider,
    defaultModel: prefill?.model,
    defaultName: prefill?.endpointName,
    defaultSecretName: prefill?.secretName,
    onSuccess: (endpoint) => navigate(GatewayRoutes.getEndpointDetailsRoute(endpoint.endpoint_id)),
    onCancel: () => navigate(GatewayRoutes.gatewayPageRoute),
  });

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <FormProvider {...form}>
        <div css={{ padding: theme.spacing.md }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link
                componentId="mlflow.gateway.create_endpoint.breadcrumb_gateway_link"
                to={GatewayRoutes.gatewayPageRoute}
              >
                <GatewayLabel />
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <Link
                componentId="mlflow.gateway.create_endpoint.breadcrumb_endpoints_link"
                to={GatewayRoutes.gatewayPageRoute}
              >
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
        <EndpointFormRenderer
          mode="create"
          isSubmitting={isLoading}
          error={error}
          errorMessage={getReadableErrorMessage(error)}
          resetErrors={resetErrors}
          selectedModel={selectedModel}
          isFormComplete={isFormComplete}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
          onNameBlur={handleNameBlur}
        />
      </FormProvider>
    </ScrollablePageWrapper>
  );
};

interface PrefillState {
  provider?: string;
  model?: string;
  endpointName?: string;
  secretName?: string;
}

const parsePrefillState = (raw: unknown): PrefillState | null => {
  if (raw === null || raw === undefined || typeof raw !== 'object') return null;
  const obj = raw as Record<string, unknown>;
  return {
    provider: typeof obj['provider'] === 'string' ? obj['provider'] : undefined,
    model: typeof obj['model'] === 'string' ? obj['model'] : undefined,
    endpointName: typeof obj['endpointName'] === 'string' ? obj['endpointName'] : undefined,
    secretName: typeof obj['secretName'] === 'string' ? obj['secretName'] : undefined,
  };
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, CreateEndpointPage);
