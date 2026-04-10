import {
  Button,
  ChevronRightIcon,
  CloudModelIcon,
  LightningIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import GatewayRoutes from '../../routes';
import { formatProviderName } from '../../utils/providerUtils';

import OpenAiLogo from '../../../common/static/logos/openai.svg';
import OpenAiLogoDark from '../../../common/static/logos/openai-dark.svg';
import AnthropicLogo from '../../../common/static/logos/anthropic.svg';
import AnthropicLogoDark from '../../../common/static/logos/anthropic-dark.png';
import GeminiLogo from '../../../common/static/logos/gemini.png';
import DatabricksLogo from '../../../common/static/logos/databricks.svg';

interface ModelOption {
  model: string;
  endpointName: string;
  componentId: string;
}

interface ProviderTemplate {
  provider: string;
  secretName: string;
  logo: string;
  logoDark?: string;
  models: ModelOption[];
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
const PROVIDER_TEMPLATES: ProviderTemplate[] = [
  {
    provider: 'openai',
    secretName: 'openai-api-key',
    logo: OpenAiLogo,
    logoDark: OpenAiLogoDark,
    models: [
      { model: 'gpt-5', endpointName: 'openai-gpt-5-endpoint', componentId: 'mlflow.gateway.quick_start.openai.gpt-5' },
      {
        model: 'gpt-5-mini',
        endpointName: 'openai-gpt-5-mini-endpoint',
        componentId: 'mlflow.gateway.quick_start.openai.gpt-5-mini',
      },
      {
        model: 'gpt-5.4',
        endpointName: 'openai-gpt-5.4-endpoint',
        componentId: 'mlflow.gateway.quick_start.openai.gpt-5.4',
      },
      {
        model: 'o4-mini',
        endpointName: 'openai-o4-mini-endpoint',
        componentId: 'mlflow.gateway.quick_start.openai.o4-mini',
      },
    ],
  },
  {
    provider: 'anthropic',
    secretName: 'anthropic-api-key',
    logo: AnthropicLogo,
    logoDark: AnthropicLogoDark,
    models: [
      {
        model: 'claude-opus-4-6',
        endpointName: 'anthropic-claude-opus-endpoint',
        componentId: 'mlflow.gateway.quick_start.anthropic.claude-opus-4-6',
      },
      {
        model: 'claude-sonnet-4-6',
        endpointName: 'anthropic-claude-sonnet-endpoint',
        componentId: 'mlflow.gateway.quick_start.anthropic.claude-sonnet-4-6',
      },
      {
        model: 'claude-sonnet-4-20250514',
        endpointName: 'anthropic-claude-sonnet-4-endpoint',
        componentId: 'mlflow.gateway.quick_start.anthropic.claude-sonnet-4-20250514',
      },
      {
        model: 'claude-haiku-3-5-20241022',
        endpointName: 'anthropic-claude-haiku-endpoint',
        componentId: 'mlflow.gateway.quick_start.anthropic.claude-haiku-3-5-20241022',
      },
    ],
  },
  {
    provider: 'gemini',
    secretName: 'gemini-api-key',
    logo: GeminiLogo,
    models: [
      {
        model: 'gemini-3.0-pro',
        endpointName: 'gemini-3.0-pro-endpoint',
        componentId: 'mlflow.gateway.quick_start.gemini.gemini-3.0-pro',
      },
      {
        model: 'gemini-3.0-flash',
        endpointName: 'gemini-3.0-flash-endpoint',
        componentId: 'mlflow.gateway.quick_start.gemini.gemini-3.0-flash',
      },
      {
        model: 'gemini-2.5-pro',
        endpointName: 'gemini-2.5-pro-endpoint',
        componentId: 'mlflow.gateway.quick_start.gemini.gemini-2.5-pro',
      },
      {
        model: 'gemini-2.5-flash',
        endpointName: 'gemini-2.5-flash-endpoint',
        componentId: 'mlflow.gateway.quick_start.gemini.gemini-2.5-flash',
      },
    ],
  },
  {
    provider: 'databricks',
    secretName: 'databricks-api-key',
    logo: DatabricksLogo,
    models: [
      {
        model: 'databricks-gpt-4.1',
        endpointName: 'databricks-gpt-4.1-endpoint',
        componentId: 'mlflow.gateway.quick_start.databricks.databricks-gpt-4.1',
      },
      {
        model: 'databricks-claude-sonnet-4-6',
        endpointName: 'databricks-claude-sonnet-endpoint',
        componentId: 'mlflow.gateway.quick_start.databricks.databricks-claude-sonnet-4-6',
      },
      {
        model: 'databricks-gemini-2.5-flash',
        endpointName: 'databricks-gemini-flash-endpoint',
        componentId: 'mlflow.gateway.quick_start.databricks.databricks-gemini-2.5-flash',
      },
      {
        model: 'databricks-llama-4-maverick',
        endpointName: 'databricks-llama-maverick-endpoint',
        componentId: 'mlflow.gateway.quick_start.databricks.databricks-llama-4-maverick',
      },
    ],
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
        {PROVIDER_TEMPLATES.map((template) => (
          <div
            key={template.provider}
            css={{
              display: 'flex',
              flexDirection: 'column',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              overflow: 'hidden',
            }}
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: theme.spacing.md,
                borderBottom: `1px solid ${theme.colors.border}`,
              }}
            >
              <img
                src={theme.isDarkMode && template.logoDark ? template.logoDark : template.logo}
                alt={formatProviderName(template.provider)}
                css={{ width: 20, height: 20 }}
              />
              <Typography.Text bold>{formatProviderName(template.provider)}</Typography.Text>
            </div>
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              {template.models.map((modelOption) => (
                <Link
                  key={modelOption.model}
                  componentId={modelOption.componentId}
                  to={GatewayRoutes.createEndpointPageRoute}
                  state={{
                    provider: template.provider,
                    model: modelOption.model,
                    endpointName: modelOption.endpointName,
                    secretName: template.secretName,
                  }}
                  css={{
                    textDecoration: 'none',
                    color: 'inherit',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                    '&:hover': {
                      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                    },
                  }}
                >
                  <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                    {modelOption.model}
                  </Typography.Text>
                  <ChevronRightIcon css={{ color: theme.colors.textSecondary, fontSize: 14 }} />
                </Link>
              ))}
            </div>
          </div>
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

export const QuickStartTemplatesCompact = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <LightningIcon css={{ color: theme.colors.textSecondary, fontSize: 16 }} />
          <Typography.Text bold css={{ fontSize: theme.typography.fontSizeMd }}>
            <FormattedMessage
              defaultMessage="Quick start"
              description="Gateway > Endpoints > Compact quick start section label"
            />
          </Typography.Text>
        </div>
        <Link
          componentId="mlflow.gateway.quick_start_compact.browse_all"
          to={GatewayRoutes.createEndpointPageRoute}
          css={{ textDecoration: 'none', fontSize: theme.typography.fontSizeSm }}
        >
          <FormattedMessage
            defaultMessage="Browse all providers →"
            description="Gateway > Endpoints > Compact quick start browse all providers link"
          />
        </Link>
      </div>
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          overflowX: 'auto',
          '&::-webkit-scrollbar': { display: 'none' },
          scrollbarWidth: 'none',
        }}
      >
        {PROVIDER_TEMPLATES.map((template) => (
          <div
            key={template.provider}
            css={{
              flex: 1,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              overflow: 'hidden',
            }}
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderBottom: `1px solid ${theme.colors.border}`,
              }}
            >
              <img
                src={theme.isDarkMode && template.logoDark ? template.logoDark : template.logo}
                alt={formatProviderName(template.provider)}
                css={{ width: 16, height: 16, flexShrink: 0 }}
              />
              <Typography.Text bold css={{ fontSize: theme.typography.fontSizeSm }}>
                {formatProviderName(template.provider)}
              </Typography.Text>
            </div>
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              {template.models.map((modelOption) => (
                <Link
                  key={modelOption.model}
                  componentId={modelOption.componentId}
                  to={GatewayRoutes.createEndpointPageRoute}
                  state={{
                    provider: template.provider,
                    model: modelOption.model,
                    endpointName: modelOption.endpointName,
                    secretName: template.secretName,
                  }}
                  css={{
                    textDecoration: 'none',
                    color: 'inherit',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: theme.spacing.xs,
                    padding: `3px ${theme.spacing.sm}px`,
                    cursor: 'pointer',
                    transition: 'background-color 0.15s',
                    '&:hover': {
                      backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                    },
                  }}
                >
                  <Typography.Text
                    color="secondary"
                    css={{
                      fontSize: theme.typography.fontSizeSm,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {modelOption.model}
                  </Typography.Text>
                  <ChevronRightIcon css={{ color: theme.colors.textSecondary, fontSize: 12, flexShrink: 0 }} />
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
