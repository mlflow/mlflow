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
import type { CodingAgentType } from '../../types';
import { getDefaultLLMProvider } from '../../defaultModels';

import OpenAiLogo from '../../../common/static/logos/openai.svg';
import OpenAiLogoDark from '../../../common/static/logos/openai-dark.svg';
import AnthropicLogo from '../../../common/static/logos/anthropic.svg';
import AnthropicLogoDark from '../../../common/static/logos/anthropic-dark.png';
import GeminiLogo from '../../../common/static/logos/gemini.png';
import DatabricksLogo from '../../../common/static/logos/databricks.svg';

interface CodingAgentDoc {
  name: string;
  provider: string;
  logo: string;
  logoDark?: string;
  codingAgent: CodingAgentType;
  componentId: string;
}

interface ModelOption {
  model: string;
  endpointName: string;
}

interface ProviderTemplate {
  provider: string;
  secretName: string;
  logo: string;
  logoDark?: string;
  componentId: string;
  models: ModelOption[];
}

interface ProviderCardProps {
  template: ProviderTemplate;
  componentId: string;
  compact?: boolean;
}

const getDefaultModelOptions = (provider: string): ModelOption[] =>
  getDefaultLLMProvider(provider)!.models.map(({ model }) => ({
    model,
    endpointName: `${provider}-${model}-endpoint`,
  }));

const CodingAgentsCard = ({ compact }: { compact?: boolean }) => {
  const { theme } = useDesignSystemTheme();

  const logoSize = compact ? 16 : 20;
  const headerPadding = compact
    ? `${theme.spacing.xs}px ${theme.spacing.sm}px`
    : `${theme.spacing.sm}px ${theme.spacing.md}px`;
  const headerGap = compact ? theme.spacing.xs : theme.spacing.sm;
  const headerFontSize = compact ? theme.typography.fontSizeSm : undefined;
  const rowPadding = compact ? `3px ${theme.spacing.sm}px` : `${theme.spacing.xs}px ${theme.spacing.md}px`;
  const chevronSize = compact ? 12 : 14;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        ...(compact ? { flex: 1, minWidth: 0 } : {}),
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: headerGap,
          padding: headerPadding,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text bold css={headerFontSize ? { fontSize: headerFontSize } : undefined}>
          <FormattedMessage
            defaultMessage="Coding Agents"
            description="Gateway > Quick start > Coding Agents card header"
          />
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {CODING_AGENTS.map(({ componentId, ...agent }) => (
          <Link
            key={agent.name}
            componentId={componentId}
            to={GatewayRoutes.createEndpointPageRoute}
            state={{ codingAgent: agent.codingAgent }}
            css={{
              textDecoration: 'none',
              color: 'inherit',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: compact ? theme.spacing.xs : undefined,
              padding: rowPadding,
              cursor: 'pointer',
              transition: 'background-color 0.15s',
              '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
              },
            }}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: compact ? 4 : 8 }}>
              <img
                src={theme.isDarkMode && agent.logoDark ? agent.logoDark : agent.logo}
                alt={agent.name}
                css={{
                  width: compact ? 12 : logoSize,
                  height: compact ? 12 : logoSize,
                  objectFit: 'contain',
                  flexShrink: 0,
                }}
              />
              <Typography.Text
                color="secondary"
                css={{
                  fontSize: theme.typography.fontSizeSm,
                  ...(compact ? { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } : {}),
                }}
              >
                {agent.name}
              </Typography.Text>
            </div>
            <ChevronRightIcon css={{ color: theme.colors.textSecondary, fontSize: chevronSize, flexShrink: 0 }} />
          </Link>
        ))}
      </div>
    </div>
  );
};

const ProviderCard = ({ template, componentId, compact }: ProviderCardProps) => {
  const { theme } = useDesignSystemTheme();

  const logoSize = compact ? 16 : 20;
  const headerPadding = compact
    ? `${theme.spacing.xs}px ${theme.spacing.sm}px`
    : `${theme.spacing.sm}px ${theme.spacing.md}px`;
  const headerGap = compact ? theme.spacing.xs : theme.spacing.sm;
  const headerFontSize = compact ? theme.typography.fontSizeSm : undefined;
  const rowPadding = compact ? `3px ${theme.spacing.sm}px` : `${theme.spacing.xs}px ${theme.spacing.md}px`;
  const chevronSize = compact ? 12 : 14;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        ...(compact ? { flex: 1, minWidth: 0 } : {}),
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: headerGap,
          padding: headerPadding,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <img
          src={theme.isDarkMode && template.logoDark ? template.logoDark : template.logo}
          alt={formatProviderName(template.provider)}
          css={{ width: logoSize, height: logoSize, objectFit: 'contain', flexShrink: 0 }}
        />
        <Typography.Text bold css={headerFontSize ? { fontSize: headerFontSize } : undefined}>
          {formatProviderName(template.provider)}
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {template.models.map((modelOption) => (
          <Link
            key={modelOption.model}
            componentId={componentId}
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
              gap: compact ? theme.spacing.xs : undefined,
              padding: rowPadding,
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
                ...(compact ? { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } : {}),
              }}
            >
              {modelOption.model}
            </Typography.Text>
            <ChevronRightIcon css={{ color: theme.colors.textSecondary, fontSize: chevronSize, flexShrink: 0 }} />
          </Link>
        ))}
      </div>
    </div>
  );
};

/** Quick-start templates for the gateway empty state. */
const PROVIDER_TEMPLATES: ProviderTemplate[] = [
  {
    provider: 'openai',
    secretName: 'openai-api-key',
    logo: OpenAiLogo,
    logoDark: OpenAiLogoDark,
    componentId: 'mlflow.gateway.quick_start.openai',
    models: getDefaultModelOptions('openai'),
  },
  {
    provider: 'anthropic',
    secretName: 'anthropic-api-key',
    logo: AnthropicLogo,
    logoDark: AnthropicLogoDark,
    componentId: 'mlflow.gateway.quick_start.anthropic',
    models: getDefaultModelOptions('anthropic'),
  },
  {
    provider: 'gemini',
    secretName: 'gemini-api-key',
    logo: GeminiLogo,
    componentId: 'mlflow.gateway.quick_start.gemini',
    models: getDefaultModelOptions('gemini'),
  },
  {
    provider: 'databricks',
    secretName: 'databricks-api-key',
    logo: DatabricksLogo,
    componentId: 'mlflow.gateway.quick_start.databricks',
    models: [
      { model: 'databricks-gpt-4.1', endpointName: 'databricks-gpt-4.1-endpoint' },
      { model: 'databricks-claude-sonnet-4-6', endpointName: 'databricks-claude-sonnet-endpoint' },
      { model: 'databricks-gemini-2.5-flash', endpointName: 'databricks-gemini-flash-endpoint' },
      { model: 'databricks-llama-4-maverick', endpointName: 'databricks-llama-maverick-endpoint' },
    ],
  },
];

const COMPACT_PROVIDER_CONFIGS: { template: ProviderTemplate; componentId: string }[] = [
  { template: PROVIDER_TEMPLATES[0], componentId: 'mlflow.gateway.quick_start.compact.openai' },
  { template: PROVIDER_TEMPLATES[1], componentId: 'mlflow.gateway.quick_start.compact.anthropic' },
  { template: PROVIDER_TEMPLATES[2], componentId: 'mlflow.gateway.quick_start.compact.gemini' },
  { template: PROVIDER_TEMPLATES[3], componentId: 'mlflow.gateway.quick_start.compact.databricks' },
];

const CODING_AGENTS: CodingAgentDoc[] = [
  {
    name: 'Claude Code',
    provider: 'anthropic',
    logo: AnthropicLogo,
    logoDark: AnthropicLogoDark,
    codingAgent: 'claude-code',
    componentId: 'mlflow.gateway.quick_start.coding_agent.claude-code',
  },
  {
    name: 'OpenAI Codex',
    provider: 'openai',
    logo: OpenAiLogo,
    logoDark: OpenAiLogoDark,
    codingAgent: 'codex',
    componentId: 'mlflow.gateway.quick_start.coding_agent.codex',
  },
  {
    name: 'Gemini CLI',
    provider: 'gemini',
    logo: GeminiLogo,
    codingAgent: 'gemini-cli',
    componentId: 'mlflow.gateway.quick_start.coding_agent.gemini-cli',
  },
];

export const QuickStartTemplates = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
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
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: theme.spacing.sm,
          width: '100%',
        }}
      >
        <CodingAgentsCard />
        {PROVIDER_TEMPLATES.map((template) => (
          <ProviderCard key={template.provider} template={template} componentId={template.componentId} />
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
          componentId="mlflow.gateway.quick_start.compact.browse_all"
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
        <CodingAgentsCard compact />
        {COMPACT_PROVIDER_CONFIGS.map(({ componentId, template }) => (
          <ProviderCard key={template.provider} template={template} componentId={componentId} compact />
        ))}
      </div>
    </div>
  );
};
