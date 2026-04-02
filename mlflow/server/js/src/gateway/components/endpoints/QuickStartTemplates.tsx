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
  capabilities: string;
}

const QUICK_START_TEMPLATES: QuickStartTemplate[] = [
  {
    provider: 'openai',
    model: 'gpt-5.4',
    endpointName: 'openai-gpt-5.4-endpoint',
    secretName: 'openai-api-key',
    capabilities: 'Tools · Vision · JSON mode',
  },
  {
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    endpointName: 'anthropic-claude-sonnet-endpoint',
    secretName: 'anthropic-api-key',
    capabilities: 'Tools · Vision · Extended thinking',
  },
  {
    provider: 'gemini',
    model: 'gemini-2.5-pro',
    endpointName: 'gemini-2.5-pro-endpoint',
    secretName: 'gemini-api-key',
    capabilities: 'Tools · Vision · 1M context',
  },
  {
    provider: 'databricks',
    model: 'databricks-gpt-5',
    endpointName: 'databricks-gpt-5-endpoint',
    secretName: 'databricks-api-key',
    capabilities: 'Tools · Enterprise-grade',
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
            defaultMessage="Create an endpoint to route API calls to any AI model. Add your API key, start calling it from your agents, and configure usage tracking and budgets as you go."
            description="Gateway > Endpoints > Quick start description explaining what an endpoint is and how to get started"
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
            componentId={`mlflow.gateway.quick_start.${template.provider}`}
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
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                {template.capabilities}
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
        <Button componentId="mlflow.gateway.quick_start.browse_all_button" type="tertiary">
          <FormattedMessage
            defaultMessage="Or browse all providers and models →"
            description="Gateway > Quick start > Link to create endpoint with full model selection"
          />
        </Button>
      </Link>
    </div>
  );
};
