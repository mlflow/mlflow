import { useState, useMemo } from 'react';
import {
  CopyIcon,
  SegmentedControlGroup,
  SegmentedControlButton,
  Tabs,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';

type UnifiedVariant = 'mlflow-invocations' | 'chat-completions';

const getBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return 'http://localhost:5000';
};

const getCodeExamples = (
  base: string,
  endpointName: string,
  variant: UnifiedVariant,
): { curl: string; python: string } => {
  if (variant === 'chat-completions') {
    return {
      curl: `curl -X POST ${base}/gateway/mlflow/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`,
      python: `from openai import OpenAI

client = OpenAI(
    base_url="${base}/gateway/mlflow/v1",
    api_key="",  # API key not needed, configured server-side
)

messages = [{"role": "user", "content": "How are you?"}]

response = client.chat.completions.create(
    model="${endpointName}",  # Endpoint name as model
    messages=messages,
)
print(response.choices[0].message)`,
    };
  }
  return {
    curl: `curl -X POST ${base}/gateway/${endpointName}/mlflow/invocations \\
  -H "Content-Type: application/json" \\
  -d '{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ]
}'`,
    python: `import requests

response = requests.post(
    "${base}/gateway/${endpointName}/mlflow/invocations",
    json={
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)
print(response.json())`,
  };
};

interface StarterCodeCardProps {
  endpointName: string;
}

export const StarterCodeCard = ({ endpointName }: StarterCodeCardProps) => {
  const { theme } = useDesignSystemTheme();
  const [variant, setVariant] = useState<UnifiedVariant>('chat-completions');
  const [language, setLanguage] = useState<'curl' | 'python'>('curl');

  const base = useMemo(() => getBaseUrl(), []);
  const examples = useMemo(() => getCodeExamples(base, endpointName, variant), [base, endpointName, variant]);
  const code = language === 'curl' ? examples.curl : examples.python;

  return (
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: theme.spacing.sm }}>
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="View starter code" description="Title for starter code card on endpoint overview" />
        </Typography.Title>
        <SegmentedControlGroup
          name="starter-code-variant"
          componentId="mlflow.gateway.edit-endpoint.starter-code.variant"
          value={variant}
          onChange={({ target: { value } }) => setVariant(value as UnifiedVariant)}
        >
          <SegmentedControlButton value="mlflow-invocations">
            <FormattedMessage defaultMessage="MLflow Invocations" description="Starter code variant: MLflow Invocations" />
          </SegmentedControlButton>
          <SegmentedControlButton value="chat-completions">
            <FormattedMessage defaultMessage="OpenAI Chat Completions" description="Starter code variant: OpenAI Chat Completions" />
          </SegmentedControlButton>
        </SegmentedControlGroup>
      </div>

      <div
        css={{
          marginTop: theme.spacing.md,
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.backgroundPrimary,
          overflow: 'hidden',
        }}
      >
        <Tabs.Root
          componentId="mlflow.gateway.edit-endpoint.starter-code.language"
          valueHasNoPii
          value={language}
          onValueChange={(value) => setLanguage(value as 'curl' | 'python')}
        >
          <div
            css={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: `0 ${theme.spacing.md}px`,
            }}
          >
            <Tabs.List>
              <Tabs.Trigger value="curl">cURL</Tabs.Trigger>
              <Tabs.Trigger value="python">Python</Tabs.Trigger>
            </Tabs.List>
            <CopyButton
              componentId="mlflow.gateway.edit-endpoint.starter-code.copy"
              copyText={code}
              icon={<CopyIcon />}
              showLabel={false}
            />
          </div>

          <Tabs.Content value="curl">
            <CodeSnippet
              language="text"
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{
                fontSize: theme.typography.fontSizeSm,
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              }}
            >
              {examples.curl}
            </CodeSnippet>
          </Tabs.Content>

          <Tabs.Content value="python">
            <CodeSnippet
              language="python"
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{
                fontSize: theme.typography.fontSizeSm,
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              }}
            >
              {examples.python}
            </CodeSnippet>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
};
