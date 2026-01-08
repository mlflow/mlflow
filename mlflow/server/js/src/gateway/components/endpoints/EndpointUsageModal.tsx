import { Modal, Typography, useDesignSystemTheme, CopyIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState } from 'react';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';

interface EndpointUsageModalProps {
  open: boolean;
  onClose: () => void;
  endpointName: string;
  baseUrl?: string;
}

const getBaseUrl = (baseUrl?: string): string => {
  if (baseUrl) return baseUrl;
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return 'http://localhost:5000';
};

export const EndpointUsageModal = ({ open, onClose, endpointName, baseUrl }: EndpointUsageModalProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useState<'unified' | 'passthrough'>('unified');
  const base = getBaseUrl(baseUrl);
  const mlflowInvocationsCurlExample = `curl -X POST ${base}/gateway/${endpointName}/mlflow/invocations \\
  -H "Content-Type: application/json" \\
  -d '{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ]
}'`;

  const mlflowInvocationsPythonExample = `import requests

response = requests.post(
    "${base}/gateway/${endpointName}/mlflow/invocations",
    json={
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)
print(response.json())`;

  const openaiChatCurlExample = `curl -X POST ${base}/gateway/mlflow/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`;

  const openaiChatPythonExample = `from openai import OpenAI

client = OpenAI(
    base_url="${base}/gateway/mlflow/v1",
    api_key="",  # API key not needed, configured server-side
)

messages = [{"role": "user", "content": "How are you?"}]

response = client.chat.completions.create(
    model="${endpointName}",  # Endpoint name as model
    messages=messages,
)
print(response.choices[0].message)`;

  const openaiPassthroughCurlExample = `curl -X POST ${base}/gateway/openai/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`;

  const openaiPassthroughPythonExample = `from openai import OpenAI

client = OpenAI(
    base_url="${base}/gateway/openai/v1",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.chat.completions.create(
    model="${endpointName}",
    messages=[{"role": "user", "content": "How are you?"}],
)
print(response.choices[0].message)`;

  const anthropicPassthroughCurlExample = `curl -X POST ${base}/gateway/anthropic/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`;

  const anthropicPassthroughPythonExample = `import anthropic

client = anthropic.Anthropic(
    base_url="${base}/gateway/anthropic",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.messages.create(
    model="${endpointName}",
    max_tokens=1024,
    messages=[{"role": "user", "content": "How are you?"}],
)
print(response.content[0].text)`;

  const geminiPassthroughCurlExample = `curl -X POST ${base}/gateway/gemini/v1beta/models/${endpointName}:generateContent \\
  -H "Content-Type: application/json" \\
  -d '{
  "contents": [{
    "parts": [{"text": "How are you?"}]
  }]
}'`;

  const geminiPassthroughPythonExample = `from google import genai

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
print(response.candidates[0].content.parts[0].text)`;

  const renderCodeExample = (label: string, code: string, language: 'text' | 'python' = 'text') => (
    <div css={{ marginBottom: theme.spacing.md }}>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}
      >
        <Typography.Text bold>{label}</Typography.Text>
        <CopyButton
          componentId={`mlflow.gateway.usage-modal.copy-${label.toLowerCase().replace(/\s+/g, '-')}`}
          copyText={code}
          icon={<CopyIcon />}
          showLabel={false}
        />
      </div>
      <div css={{ position: 'relative' }}>
        <CodeSnippet
          language={language}
          theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
          style={{
            fontSize: theme.typography.fontSizeSm,
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          }}
        >
          {code}
        </CodeSnippet>
      </div>
    </div>
  );

  return (
    <Modal
      componentId="mlflow.gateway.endpoint-usage-modal"
      visible={open}
      onCancel={onClose}
      title={<FormattedMessage defaultMessage="Query endpoint" description="Endpoint usage modal title" />}
      footer={null}
      size="wide"
    >
      <div
        css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, maxHeight: '70vh', overflowY: 'auto' }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Use these code examples to call your endpoint. Choose between unified APIs for seamless model switching or passthrough APIs for provider-specific features."
            description="Endpoint usage modal description"
          />
        </Typography.Text>

        <div css={{ display: 'flex', gap: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.border}` }}>
          <div
            css={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              cursor: 'pointer',
              borderBottom:
                activeTab === 'unified'
                  ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`
                  : '2px solid transparent',
              color: activeTab === 'unified' ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textSecondary,
              fontWeight: activeTab === 'unified' ? 'bold' : 'normal',
            }}
            onClick={() => setActiveTab('unified')}
          >
            <FormattedMessage defaultMessage="Unified APIs" description="Unified APIs tab title" />
          </div>
          <div
            css={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              cursor: 'pointer',
              borderBottom:
                activeTab === 'passthrough'
                  ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`
                  : '2px solid transparent',
              color:
                activeTab === 'passthrough' ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textSecondary,
              fontWeight: activeTab === 'passthrough' ? 'bold' : 'normal',
            }}
            onClick={() => setActiveTab('passthrough')}
          >
            <FormattedMessage defaultMessage="Passthrough APIs" description="Passthrough APIs tab title" />
          </div>
        </div>

        {activeTab === 'unified' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            <div>
              <Typography.Title level={4} css={{ marginTop: 0 }}>
                <FormattedMessage
                  defaultMessage="MLflow Invocations API"
                  description="MLflow invocations API section title"
                />
              </Typography.Title>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Native MLflow API for model invocations. Supports seamless model switching and advanced routing."
                  description="MLflow invocations API description"
                />
              </Typography.Text>
              {renderCodeExample('cURL', mlflowInvocationsCurlExample, 'text')}
              {renderCodeExample('Python', mlflowInvocationsPythonExample, 'python')}
            </div>

            <div>
              <Typography.Title level={4}>
                <FormattedMessage
                  defaultMessage="OpenAI-Compatible Chat Completions API"
                  description="OpenAI compatible API section title"
                />
              </Typography.Title>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Unified OpenAI compatible API for model invocations. Set the endpoint name as the model parameter."
                  description="OpenAI compatible API description"
                />
              </Typography.Text>
              {renderCodeExample('cURL', openaiChatCurlExample, 'text')}
              {renderCodeExample('Python', openaiChatPythonExample, 'python')}
            </div>
          </div>
        )}

        {activeTab === 'passthrough' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            <div>
              <Typography.Title level={4} css={{ marginTop: 0 }}>
                <FormattedMessage defaultMessage="OpenAI Passthrough" description="OpenAI passthrough section title" />
              </Typography.Title>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Direct access to OpenAI-specific features and parameters."
                  description="OpenAI passthrough description"
                />
              </Typography.Text>
              {renderCodeExample('cURL', openaiPassthroughCurlExample, 'text')}
              {renderCodeExample('Python', openaiPassthroughPythonExample, 'python')}
            </div>

            <div>
              <Typography.Title level={4}>
                <FormattedMessage
                  defaultMessage="Anthropic Passthrough"
                  description="Anthropic passthrough section title"
                />
              </Typography.Title>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Direct access to Anthropic's Messages API with Claude-specific features."
                  description="Anthropic passthrough description"
                />
              </Typography.Text>
              {renderCodeExample('cURL', anthropicPassthroughCurlExample, 'text')}
              {renderCodeExample('Python', anthropicPassthroughPythonExample, 'python')}
            </div>

            <div>
              <Typography.Title level={4}>
                <FormattedMessage
                  defaultMessage="Google Gemini Passthrough"
                  description="Gemini passthrough section title"
                />
              </Typography.Title>
              <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                <FormattedMessage
                  defaultMessage="Direct access to Google's Gemini API. Note: endpoint name is part of the URL path."
                  description="Gemini passthrough description"
                />
              </Typography.Text>
              {renderCodeExample('cURL', geminiPassthroughCurlExample)}
              {renderCodeExample('Python', geminiPassthroughPythonExample)}
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
};
