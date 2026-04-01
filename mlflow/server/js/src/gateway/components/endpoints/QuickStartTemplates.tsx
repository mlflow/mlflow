import { Button, CloudModelIcon, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';
import { formatProviderName } from '../../utils/providerUtils';

interface QuickStartTemplate {
  provider: string;
  model: string;
  endpointName: string;
  secretName: string;
}

const QUICK_START_TEMPLATES: QuickStartTemplate[] = [
  { provider: 'openai', model: 'gpt-5.4', endpointName: 'openai-gpt-5.4-endpoint', secretName: 'openai-api-key' },
  {
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    endpointName: 'anthropic-claude-sonnet-endpoint',
    secretName: 'anthropic-api-key',
  },
  {
    provider: 'gemini',
    model: 'gemini-2.5-pro',
    endpointName: 'gemini-2.5-pro-endpoint',
    secretName: 'gemini-api-key',
  },
  {
    provider: 'databricks',
    model: 'databricks-gpt-5',
    endpointName: 'databricks-gpt-5-endpoint',
    secretName: 'databricks-api-key',
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
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Quick-start with a popular model, or choose from 60+ providers and all their supported models."
            description="Gateway > Endpoints > Quick start description"
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
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, marginTop: theme.spacing.xs }}>
                <PlusIcon css={{ fontSize: 12, color: theme.colors.actionPrimaryBackgroundDefault }} />
                <Typography.Text
                  css={{ color: theme.colors.actionPrimaryBackgroundDefault, fontSize: theme.typography.fontSizeSm }}
                >
                  <FormattedMessage
                    defaultMessage="Create endpoint"
                    description="Gateway > Quick start card > Create endpoint link text"
                  />
                </Typography.Text>
              </div>
            </div>
          </Link>
        ))}
      </div>

      <Link
        componentId="mlflow.gateway.quick_start.browse_all"
        to={GatewayRoutes.createEndpointPageRoute}
        css={{
          color: theme.colors.actionPrimaryBackgroundDefault,
          fontSize: theme.typography.fontSizeSm,
          textDecoration: 'none',
          '&:hover': { textDecoration: 'underline' },
        }}
      >
        <FormattedMessage
          defaultMessage="Or browse all providers and models →"
          description="Gateway > Quick start > Link to create endpoint with full model selection"
        />
      </Link>
    </div>
  );
};
