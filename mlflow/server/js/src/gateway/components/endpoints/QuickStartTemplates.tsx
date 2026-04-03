import { Button, CloudModelIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';
import { formatProviderName } from '../../utils/providerUtils';

interface QuickStartTemplate {
  provider: string;
  model: string;
  endpointName: string;
  secretName: string;
  componentId: string;
}

/**
 * Quick-start templates for the gateway empty state.
 *
 * UPDATE WHEN STALE: These models are hardcoded and should be reviewed
 * periodically (e.g. each release) to ensure they still represent current,
 * popular models for each provider. Check the model catalog JSON
 * (mlflow/utils/model_prices_and_context_window.json) for available models.
 *
 * Last reviewed: 2026-04-02
 */
const QUICK_START_TEMPLATES: QuickStartTemplate[] = [
  {
    provider: 'openai',
    model: 'gpt-5.4',
    endpointName: 'openai-gpt-5.4-endpoint',
    secretName: 'openai-api-key',
    componentId: 'mlflow.gateway.quick_start.openai',
  },
  {
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    endpointName: 'anthropic-claude-sonnet-endpoint',
    secretName: 'anthropic-api-key',
    componentId: 'mlflow.gateway.quick_start.anthropic',
  },
  {
    provider: 'gemini',
    model: 'gemini-2.5-pro',
    endpointName: 'gemini-2.5-pro-endpoint',
    secretName: 'gemini-api-key',
    componentId: 'mlflow.gateway.quick_start.gemini',
  },
  {
    provider: 'databricks',
    model: 'databricks-gpt-5',
    endpointName: 'databricks-gpt-5-endpoint',
    secretName: 'databricks-api-key',
    componentId: 'mlflow.gateway.quick_start.databricks',
  },
];

export const QuickStartTemplates = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: theme.spacing.lg,
        gap: theme.spacing.lg,
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: theme.spacing.sm }}>
        <CloudModelIcon css={{ fontSize: 36, color: theme.colors.textSecondary }} />
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="Get started with AI Gateway"
            description="Gateway > Endpoints > Quick start title"
          />
        </Typography.Title>
        <Typography.Text color="secondary" css={{ textAlign: 'center', maxWidth: 520 }}>
          <FormattedMessage
            defaultMessage="A Gateway endpoint routes your agent calls to any AI model, with built-in usage tracking through MLflow Tracing, budget controls, and more."
            description="Gateway > Endpoints > Quick start description explaining what an endpoint is and how to get started"
          />
        </Typography.Text>
        <Typography.Text color="secondary" css={{ textAlign: 'center', maxWidth: 520 }}>
          <FormattedMessage
            defaultMessage="Quick-start with a popular model below, or create an endpoint from 60+ providers and all their supported models."
            description="Gateway > Endpoints > Quick start call to action"
          />
        </Typography.Text>
      </div>

      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: theme.spacing.md,
          maxWidth: 520,
          width: '100%',
        }}
      >
        {QUICK_START_TEMPLATES.map((template) => (
          <Link
            key={template.provider}
            componentId={template.componentId}
            to={GatewayRoutes.createEndpointPageRoute}
            state={{
              provider: template.provider,
              model: template.model,
              endpointName: template.endpointName,
              secretName: template.secretName,
            }}
            css={{
              textDecoration: 'none',
              color: 'inherit',
              display: 'flex',
            }}
          >
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                padding: theme.spacing.md,
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusMd,
                width: '100%',
                cursor: 'pointer',
                transition: 'border-color 0.15s, box-shadow 0.15s',
                '&:hover': {
                  borderColor: theme.colors.actionPrimaryBackgroundDefault,
                  boxShadow: theme.shadows.sm,
                },
              }}
            >
              <Typography.Text bold>{formatProviderName(template.provider)}</Typography.Text>
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                {template.model}
              </Typography.Text>
            </div>
          </Link>
        ))}
      </div>

      <Link
        componentId="mlflow.gateway.quick_start.browse_all"
        to={GatewayRoutes.createEndpointPageRoute}
        css={{ textDecoration: 'none' }}
      >
        <Button componentId="mlflow.gateway.quick_start.browse_all.button" type="tertiary">
          <FormattedMessage
            defaultMessage="Or browse all providers and models →"
            description="Gateway > Quick start > Link to create endpoint with full model selection"
          />
        </Button>
      </Link>
    </div>
  );
};
