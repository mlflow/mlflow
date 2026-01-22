/**
 * Step 1: Provider selection for MLflow Assistant setup.
 */

import { useState } from 'react';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { toRGBA } from '@mlflow/mlflow/src/common/utils/toRGBA';
import AnthropicLogo from '@mlflow/mlflow/src/common/static/logos/anthropic.svg';
import GeminiLogo from '@mlflow/mlflow/src/common/static/logos/gemini.png';
import OpenAiLogo from '@mlflow/mlflow/src/common/static/logos/openai.svg';

const COMPONENT_ID = 'mlflow.assistant.setup.provider';

interface Provider {
  id: string;
  name: string;
  description: string;
  logo: string;
  available: boolean;
}

const PROVIDERS: Provider[] = [
  {
    id: 'claude_code',
    name: 'Claude Code',
    description: "AI assistant powered by Anthropic's Claude. Requires Claude Code CLI installed locally.",
    logo: AnthropicLogo,
    available: true,
  },
];

const COMING_SOON_LOGOS = [OpenAiLogo, GeminiLogo];

interface SetupStepProviderProps {
  selectedProvider: string;
  onContinue: (provider: string) => void;
}

export const SetupStepProvider = ({ selectedProvider, onContinue }: SetupStepProviderProps) => {
  const { theme } = useDesignSystemTheme();
  const [selected, setSelected] = useState<string>(selectedProvider);

  const handleContinue = () => {
    if (selected) {
      onContinue(selected);
    }
  };

  const renderProviderCard = (provider: Provider) => (
    <div
      key={provider.id}
      onClick={() => provider.available && setSelected(provider.id)}
      css={{
        cursor: provider.available ? 'pointer' : 'not-allowed',
        opacity: provider.available ? 1 : 0.5,
        padding: theme.spacing.md,
        border: `2px solid ${selected === provider.id ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor:
          selected === provider.id ? toRGBA(theme.colors.actionPrimaryBackgroundDefault, 0.1) : 'transparent',
        width: '100%',
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
        <div
          css={{
            width: 20,
            height: 20,
            borderRadius: '50%',
            border: `2px solid ${
              selected === provider.id ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border
            }`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {selected === provider.id && (
            <div
              css={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
              }}
            />
          )}
        </div>
        <img
          src={provider.logo}
          alt={provider.name}
          css={{
            width: 32,
            height: 32,
            objectFit: 'contain',
            flexShrink: 0,
          }}
        />
        <div css={{ flex: 1 }}>
          <Typography.Text
            bold
            css={{
              color: provider.available ? theme.colors.textPrimary : theme.colors.textSecondary,
            }}
          >
            {provider.name}
          </Typography.Text>
          <Typography.Text
            color="secondary"
            css={{ fontSize: theme.typography.fontSizeSm, display: 'block', marginTop: 2 }}
          >
            {provider.description}
          </Typography.Text>
        </div>
      </div>
    </div>
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <div>
        <Typography.Title level={5} css={{ marginBottom: theme.spacing.xs }}>
          Select AI Provider
        </Typography.Title>
        <Typography.Text color="secondary">Choose the AI provider for your assistant:</Typography.Text>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {PROVIDERS.map(renderProviderCard)}
      </div>

      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          More coming soon:
        </Typography.Text>
        {COMING_SOON_LOGOS.map((logo, index) => (
          <img
            key={index}
            src={logo}
            alt="Coming soon"
            css={{
              width: 28,
              height: 28,
              objectFit: 'contain',
              opacity: 0.5,
            }}
          />
        ))}
      </div>

      <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
        <Button
          componentId={`${COMPONENT_ID}.continue`}
          type="primary"
          onClick={handleContinue}
          disabled={!selected || !PROVIDERS.find((p) => p.id === selected)?.available}
        >
          Continue
        </Button>
      </div>
    </div>
  );
};
