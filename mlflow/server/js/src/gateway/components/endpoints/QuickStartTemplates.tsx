import { useMemo } from 'react';
import { Button, Card, CloudModelIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';
import { formatProviderName } from '../../utils/providerUtils';
import { getModelCapabilities } from '../../utils/formatters';
import { useModelsQuery } from '../../hooks/useModelsQuery';

interface QuickStartTemplate {
  provider: string;
  model: string;
  endpointName: string;
  secretName: string;
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
  },
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

const useTemplateCapabilities = (templates: QuickStartTemplate[]) => {
  const providers = [...new Set(templates.map((t) => t.provider))];

  // Fetch models for each provider. The number of hooks is fixed (4 providers)
  // so the rules of hooks are satisfied.
  const openaiModels = useModelsQuery({ provider: providers.includes('openai') ? 'openai' : undefined });
  const anthropicModels = useModelsQuery({ provider: providers.includes('anthropic') ? 'anthropic' : undefined });
  const geminiModels = useModelsQuery({ provider: providers.includes('gemini') ? 'gemini' : undefined });
  const databricksModels = useModelsQuery({ provider: providers.includes('databricks') ? 'databricks' : undefined });

  const allModels = useMemo(() => {
    const combined = [
      ...(openaiModels.data ?? []),
      ...(anthropicModels.data ?? []),
      ...(geminiModels.data ?? []),
      ...(databricksModels.data ?? []),
    ];
    const map = new Map<string, (typeof combined)[number]>();
    for (const m of combined) {
      map.set(`${m.provider}/${m.model}`, m);
    }
    return map;
  }, [openaiModels.data, anthropicModels.data, geminiModels.data, databricksModels.data]);

  return useMemo(
    () =>
      new Map(
        templates.map((t) => {
          const model = allModels.get(`${t.provider}/${t.model}`);
          return [t.provider, getModelCapabilities(model)];
        }),
      ),
    [templates, allModels],
  );
};

export const QuickStartTemplates = () => {
  const { theme } = useDesignSystemTheme();
  const capabilities = useTemplateCapabilities(QUICK_START_TEMPLATES);

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
        {QUICK_START_TEMPLATES.map((template) => {
          const caps = capabilities.get(template.provider);
          return (
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
              <Card
                componentId={`mlflow.gateway.quick_start.${template.provider}.card`}
                css={{
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                }}
              >
                <Typography.Text bold>{formatProviderName(template.provider)}</Typography.Text>
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  {template.model}
                </Typography.Text>
                {caps && (
                  <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                    {caps}
                  </Typography.Text>
                )}
              </Card>
            </Link>
          );
        })}
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
