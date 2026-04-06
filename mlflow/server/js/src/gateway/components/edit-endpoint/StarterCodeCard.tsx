import { useState, useMemo, useCallback } from 'react';
import {
  Button,
  CopyIcon,
  Modal,
  SegmentedControlGroup,
  SegmentedControlButton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { TryItPanel } from '../endpoints/TryItPanel';

const UNIFIED_COMMENT =
  '# Unified OpenAI compatible API for model invocations. Set the endpoint name as the model parameter.';

const getBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return 'http://localhost:5000';
};

type ApiVariant = 'chat-completions' | 'openai-responses' | 'anthropic-messages' | 'gemini-generate';

const getCodeExamples = (base: string, endpointName: string, variant: ApiVariant): { curl: string; python: string } => {
  switch (variant) {
    case 'chat-completions':
      return {
        curl: `${UNIFIED_COMMENT}

curl -X POST ${base}/gateway/mlflow/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`,
        python: `${UNIFIED_COMMENT}

from openai import OpenAI

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
    case 'openai-responses':
      return {
        curl: `# Passthrough to OpenAI's Responses API, supporting multi-turn conversations
# with vision and audio. New OpenAI features are available immediately.

curl -X POST ${base}/gateway/openai/v1/responses \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "input": "How are you?"
}'`,
        python: `# Passthrough to OpenAI's Responses API, supporting multi-turn conversations
# with vision and audio. New OpenAI features are available immediately.

from openai import OpenAI

client = OpenAI(
    base_url="${base}/gateway/openai/v1",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.responses.create(
    model="${endpointName}",
    input="How are you?",
)
print(response.output_text)`,
      };
    case 'anthropic-messages':
      return {
        curl: `# Passthrough to Anthropic's Messages API with Claude-specific features.
# New Anthropic features are available immediately.

curl -X POST ${base}/gateway/anthropic/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`,
        python: `# Passthrough to Anthropic's Messages API with Claude-specific features.
# New Anthropic features are available immediately.

import anthropic

client = anthropic.Anthropic(
    base_url="${base}/gateway/anthropic",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.messages.create(
    model="${endpointName}",
    max_tokens=1024,
    messages=[{"role": "user", "content": "How are you?"}],
)
print(response.content[0].text)`,
      };
    case 'gemini-generate':
      return {
        curl: `# Passthrough to Google's Gemini API. New Google features are available immediately.
# Note: endpoint name is part of the URL path.

curl -X POST ${base}/gateway/gemini/v1beta/models/${endpointName}:generateContent \\
  -H "Content-Type: application/json" \\
  -d '{
  "contents": [{
    "parts": [{"text": "How are you?"}]
  }]
}'`,
        python: `# Passthrough to Google's Gemini API. New Google features are available immediately.
# Note: endpoint name is part of the URL path.

from google import genai

# Configure with custom endpoint
client = genai.Client(
    api_key='dummy',
    http_options={
        'base_url': "${base}/gateway/gemini",
    }
)
response = client.models.generate_content(
    model="${endpointName}",
    contents={'text': 'How are you?'},
)
client.close()
print(response.candidates[0].content.parts[0].text)`,
      };
  }
};

type PassthroughInfo = {
  variant: ApiVariant;
  label: string;
};

const getPassthroughForProvider = (provider: string | undefined): PassthroughInfo | null => {
  switch (provider) {
    case 'openai':
    case 'azure':
      return { variant: 'openai-responses', label: 'OpenAI Responses' };
    case 'anthropic':
      return { variant: 'anthropic-messages', label: 'Anthropic Messages' };
    case 'gemini':
      return { variant: 'gemini-generate', label: 'Gemini Generate Content' };
    default:
      return null;
  }
};

const getRequestUrl = (base: string, endpointName: string, variant: ApiVariant): string => {
  switch (variant) {
    case 'chat-completions':
      return `${base}/gateway/mlflow/v1/chat/completions`;
    case 'openai-responses':
      return `${base}/gateway/openai/v1/responses`;
    case 'anthropic-messages':
      return `${base}/gateway/anthropic/v1/messages`;
    case 'gemini-generate':
      return `${base}/gateway/gemini/v1beta/models/${endpointName}:generateContent`;
  }
};

const getDefaultBody = (endpointName: string, variant: ApiVariant): string => {
  switch (variant) {
    case 'chat-completions':
      return JSON.stringify({ model: endpointName, messages: [{ role: 'user', content: 'How are you?' }] }, null, 2);
    case 'openai-responses':
      return JSON.stringify({ model: endpointName, input: 'How are you?' }, null, 2);
    case 'anthropic-messages':
      return JSON.stringify(
        { model: endpointName, max_tokens: 1024, messages: [{ role: 'user', content: 'How are you?' }] },
        null,
        2,
      );
    case 'gemini-generate':
      return JSON.stringify({ contents: [{ parts: [{ text: 'How are you?' }] }] }, null, 2);
  }
};

interface StarterCodeCardProps {
  endpointName: string;
  provider?: string;
}

export const StarterCodeCard = ({ endpointName, provider }: StarterCodeCardProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeApi, setActiveApi] = useState<ApiVariant>('chat-completions');
  const [language, setLanguage] = useState<'curl' | 'python'>('curl');
  const [isTryItOpen, setIsTryItOpen] = useState(false);
  const [tryItResetKey, setTryItResetKey] = useState(0);

  const handleOpenTryIt = useCallback(() => {
    setTryItResetKey((k) => k + 1);
    setIsTryItOpen(true);
  }, []);

  const passthrough = useMemo(() => getPassthroughForProvider(provider), [provider]);

  const apiOptions = useMemo(() => {
    const options: { value: ApiVariant; label: string }[] = [
      { value: 'chat-completions', label: 'MLflow Chat Completions' },
    ];
    if (passthrough) {
      options.push({ value: passthrough.variant, label: passthrough.label });
    }
    return options;
  }, [passthrough]);

  const base = useMemo(() => getBaseUrl(), []);
  const examples = useMemo(() => getCodeExamples(base, endpointName, activeApi), [base, endpointName, activeApi]);
  const code = language === 'curl' ? examples.curl : examples.python;
  const tryItRequestUrl = useMemo(() => getRequestUrl(base, endpointName, activeApi), [base, endpointName, activeApi]);
  const tryItDefaultBody = useMemo(() => getDefaultBody(endpointName, activeApi), [endpointName, activeApi]);
  const tryItOptions =
    activeApi === 'anthropic-messages'
      ? { headers: { 'anthropic-dangerous-direct-browser-access': 'true' } }
      : undefined;

  return (
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="View starter code"
            description="Title for starter code card on endpoint overview"
          />
        </Typography.Title>
        <SegmentedControlGroup
          name="starter-code-api"
          componentId="mlflow.gateway.edit-endpoint.starter-code.api"
          value={activeApi}
          onChange={({ target: { value } }) => setActiveApi(value as ApiVariant)}
        >
          {apiOptions.map((opt) => (
            <SegmentedControlButton key={opt.value} value={opt.value}>
              {opt.label}
            </SegmentedControlButton>
          ))}
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
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-end',
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px 0 ${theme.spacing.md}px`,
            borderBottom: `1px solid ${theme.colors.borderDecorative}`,
          }}
        >
          <div css={{ display: 'flex', gap: theme.spacing.md }}>
            {(['curl', 'python'] as const).map((lang) => (
              <button
                key={lang}
                type="button"
                onClick={() => setLanguage(lang)}
                css={{
                  background: 'none',
                  border: 'none',
                  borderBottom: `2px solid ${language === lang ? theme.colors.actionPrimaryBackgroundDefault : 'transparent'}`,
                  padding: `${theme.spacing.sm}px 0`,
                  cursor: 'pointer',
                  fontSize: theme.typography.fontSizeBase,
                  fontWeight: language === lang ? theme.typography.typographyBoldFontWeight : 'normal',
                  color: language === lang ? theme.colors.textPrimary : theme.colors.textSecondary,
                  '&:hover': {
                    color: theme.colors.textPrimary,
                  },
                }}
              >
                {lang === 'curl' ? 'cURL' : 'Python'}
              </button>
            ))}
          </div>
          <CopyButton
            componentId="mlflow.gateway.edit-endpoint.starter-code.copy"
            copyText={code}
            icon={<CopyIcon />}
            showLabel={false}
          />
        </div>

        <CodeSnippet
          language={language === 'python' ? 'python' : 'text'}
          theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
          style={{
            fontSize: theme.typography.fontSizeSm,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            margin: 0,
          }}
        >
          {code}
        </CodeSnippet>
      </div>

      <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
        <Button
          componentId="mlflow.gateway.edit-endpoint.starter-code.try-in-browser"
          type="primary"
          onClick={handleOpenTryIt}
        >
          <FormattedMessage defaultMessage="Try in Browser" description="Button to open try-it dialog for endpoint" />
        </Button>
      </div>

      <Modal
        componentId="mlflow.gateway.edit-endpoint.try-it-modal"
        visible={isTryItOpen}
        onCancel={() => setIsTryItOpen(false)}
        title={<FormattedMessage defaultMessage="Query endpoint" description="Title for try-it modal dialog" />}
        footer={null}
        size="wide"
      >
        <div css={{ minHeight: 400 }}>
          <TryItPanel
            key={`try-it-${activeApi}-${tryItResetKey}`}
            description={
              <FormattedMessage
                defaultMessage="Edit the request body below and click Send request to call the endpoint."
                description="Try it description in starter code modal"
              />
            }
            requestTooltipContent={
              <FormattedMessage
                defaultMessage="JSON body sent to the endpoint. Edit the fields and click Send request."
                description="Request body tooltip in starter code try-it modal"
              />
            }
            componentId="mlflow.gateway.edit-endpoint.try-it-modal.request-tooltip"
            tryItRequestUrl={tryItRequestUrl}
            tryItDefaultBody={tryItDefaultBody}
            tryItOptions={tryItOptions}
          />
        </div>
      </Modal>
    </div>
  );
};
