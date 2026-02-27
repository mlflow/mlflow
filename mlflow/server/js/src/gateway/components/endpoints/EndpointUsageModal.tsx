import {
  Button,
  Modal,
  Tooltip,
  Tabs,
  Typography,
  useDesignSystemTheme,
  CopyIcon,
  Input,
  SegmentedControlGroup,
  SegmentedControlButton,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState, useCallback, useMemo, useEffect } from 'react';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { TryItPanel } from './TryItPanel';

type Provider = 'openai' | 'anthropic' | 'gemini';
type Language = 'curl' | 'python';

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

type TryItUnifiedVariant = 'mlflow-invocations' | 'chat-completions';

const getTryItRequestUrl = (
  base: string,
  endpointName: string,
  apiType: 'unified' | 'passthrough',
  unifiedVariant: TryItUnifiedVariant,
  provider: Provider,
): string => {
  if (apiType === 'unified') {
    if (unifiedVariant === 'chat-completions') {
      return `${base}/gateway/mlflow/v1/chat/completions`;
    }
    return `${base}/gateway/${endpointName}/mlflow/invocations`;
  }
  switch (provider) {
    case 'openai':
      return `${base}/gateway/openai/v1/responses`;
    case 'anthropic':
      return `${base}/gateway/anthropic/v1/messages`;
    case 'gemini':
      return `${base}/gateway/gemini/v1beta/models/${endpointName}:generateContent`;
    default:
      return `${base}/gateway/${endpointName}/mlflow/invocations`;
  }
};

type CodeExampleVariant = TryItUnifiedVariant | Provider;

const getCodeExamples = (
  base: string,
  endpointName: string,
  variant: CodeExampleVariant,
): { curl: string; python: string; defaultBody: string } => {
  switch (variant) {
    case 'mlflow-invocations':
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
        defaultBody: JSON.stringify(
          {
            messages: [{ role: 'user', content: 'Hello, how are you?' }],
          },
          null,
          2,
        ),
      };
    case 'chat-completions':
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
        defaultBody: JSON.stringify(
          {
            model: endpointName,
            messages: [{ role: 'user', content: 'How are you?' }],
          },
          null,
          2,
        ),
      };
    case 'openai':
      return {
        curl: `curl -X POST ${base}/gateway/openai/v1/responses \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "input": "How are you?"
}'`,
        python: `from openai import OpenAI

client = OpenAI(
    base_url="${base}/gateway/openai/v1",
    api_key="dummy",  # API key not needed, configured server-side
)

response = client.responses.create(
    model="${endpointName}",
    input="How are you?",
)
print(response.output_text)`,
        defaultBody: JSON.stringify({ model: endpointName, input: 'How are you?' }, null, 2),
      };
    case 'anthropic':
      return {
        curl: `curl -X POST ${base}/gateway/anthropic/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "${endpointName}",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "How are you?"}
  ]
}'`,
        python: `import anthropic

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
        defaultBody: JSON.stringify(
          {
            model: endpointName,
            max_tokens: 1024,
            messages: [{ role: 'user', content: 'How are you?' }],
          },
          null,
          2,
        ),
      };
    case 'gemini':
      return {
        curl: `curl -X POST ${base}/gateway/gemini/v1beta/models/${endpointName}:generateContent \\
  -H "Content-Type: application/json" \\
  -d '{
  "contents": [{
    "parts": [{"text": "How are you?"}]
  }]
}'`,
        python: `from google import genai

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
        defaultBody: JSON.stringify(
          {
            contents: [{ parts: [{ text: 'How are you?' }] }],
          },
          null,
          2,
        ),
      };
    default:
      return getCodeExamples(base, endpointName, 'mlflow-invocations');
  }
};

const getDefaultRequestBody = (endpointName: string, variant: CodeExampleVariant): string =>
  getCodeExamples('', endpointName, variant).defaultBody;

type ViewMode = 'try-it' | 'curl' | 'python';

export const EndpointUsageModal = ({ open, onClose, endpointName, baseUrl }: EndpointUsageModalProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useState<'unified' | 'passthrough'>('unified');
  const [viewMode, setViewMode] = useState<ViewMode>('try-it');
  const [selectedProvider, setSelectedProvider] = useState<Provider>('openai');
  const [tryItUnifiedVariant, setTryItUnifiedVariant] = useState<'mlflow-invocations' | 'chat-completions'>(
    'mlflow-invocations',
  );
  const [tryItResetKey, setTryItResetKey] = useState(0);
  const base = getBaseUrl(baseUrl);

  // Reset modal state when opened so users get a fresh Try-it experience each time
  useEffect(() => {
    if (open) {
      setActiveTab('unified');
      setViewMode('try-it');
      setTryItUnifiedVariant('mlflow-invocations');
      setSelectedProvider('openai');
      setTryItResetKey((k) => k + 1);
    }
  }, [open]);

  const tryItRequestUrl = useMemo(
    () => getTryItRequestUrl(base, endpointName, activeTab, tryItUnifiedVariant, selectedProvider),
    [base, endpointName, activeTab, tryItUnifiedVariant, selectedProvider],
  );

  const tryItDefaultBody = useMemo(
    () => getDefaultRequestBody(endpointName, activeTab === 'unified' ? tryItUnifiedVariant : selectedProvider),
    [activeTab, tryItUnifiedVariant, selectedProvider, endpointName],
  );

  const renderCodeExample = (label: string, code: string, language: 'text' | 'python' = 'text') => (
    <div css={{ marginBottom: theme.spacing.md }}>
      <div css={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', marginBottom: theme.spacing.xs }}>
        <CopyButton
          componentId="codegen_mlflow_app_src_oss_gateway_components_endpoints_EndpointUsageModal.tsx_147"
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

        <Tabs.Root
          componentId="mlflow.gateway.usage-modal.tabs"
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as 'unified' | 'passthrough')}
          css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
        >
          <Tabs.List
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              borderBottom: `1px solid ${theme.colors.border}`,
              padding: 0,
            }}
          >
            <Tabs.Trigger value="unified">
              <FormattedMessage defaultMessage="Unified APIs" description="Unified APIs tab title" />
            </Tabs.Trigger>
            <Tabs.Trigger value="passthrough">
              <FormattedMessage defaultMessage="Passthrough APIs" description="Passthrough APIs tab title" />
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="unified" css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              <div>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="Unified API" description="Unified API variant label" />
                </Typography.Text>
                <SegmentedControlGroup
                  name="try-it-unified-variant"
                  componentId="mlflow.gateway.usage-modal.try-it.unified-variant"
                  value={tryItUnifiedVariant}
                  onChange={({ target: { value } }) =>
                    setTryItUnifiedVariant(value as 'mlflow-invocations' | 'chat-completions')
                  }
                  css={{ marginBottom: theme.spacing.sm }}
                >
                  <SegmentedControlButton value="mlflow-invocations">
                    <FormattedMessage
                      defaultMessage="MLflow Invocations"
                      description="Unified API variant: MLflow Invocations"
                    />
                  </SegmentedControlButton>
                  <SegmentedControlButton value="chat-completions">
                    <FormattedMessage
                      defaultMessage="OpenAI Chat Completions"
                      description="Unified API variant: OpenAI Chat Completions"
                    />
                  </SegmentedControlButton>
                </SegmentedControlGroup>
              </div>

              {tryItUnifiedVariant === 'mlflow-invocations' && (
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Native MLflow API for model invocations. Supports seamless model switching and advanced routing."
                    description="MLflow invocations API description"
                  />
                </Typography.Text>
              )}
              {tryItUnifiedVariant === 'chat-completions' && (
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Unified OpenAI compatible API for model invocations. Set the endpoint name as the model parameter."
                    description="OpenAI compatible API description"
                  />
                </Typography.Text>
              )}

              <SegmentedControlGroup
                name="unified-view-mode"
                componentId="mlflow.gateway.usage-modal.unified-view-mode"
                value={viewMode}
                onChange={({ target: { value } }) => setViewMode(value as ViewMode)}
                css={{ marginBottom: theme.spacing.sm }}
              >
                <SegmentedControlButton value="try-it">
                  <FormattedMessage defaultMessage="Try it" description="Try it - interactive request/response" />
                </SegmentedControlButton>
                <SegmentedControlButton value="curl">cURL</SegmentedControlButton>
                <SegmentedControlButton value="python">Python</SegmentedControlButton>
              </SegmentedControlGroup>

              {viewMode === 'try-it' && (
                <TryItPanel
                  key={`try-it-unified-${tryItResetKey}`}
                  description={
                    <FormattedMessage
                      defaultMessage="Edit the request body below and click Send request to call the endpoint."
                      description="Try it description for unified"
                    />
                  }
                  requestTooltipContent={
                    tryItUnifiedVariant === 'chat-completions' ? (
                      <FormattedMessage
                        defaultMessage='JSON body for the OpenAI-compatible Chat Completions API. Include "model" (endpoint name) and "messages".'
                        description="Request body tooltip for unified Chat Completions API"
                      />
                    ) : (
                      <FormattedMessage
                        defaultMessage='JSON body for the MLflow Invocations API. Use "messages" for chat or "input" for embeddings.'
                        description="Request body tooltip for unified MLflow Invocations API"
                      />
                    )
                  }
                  requestTooltipComponentId="mlflow.gateway.usage-modal.try-it.request-tooltip"
                  tryItRequestUrl={tryItRequestUrl}
                  tryItDefaultBody={tryItDefaultBody}
                />
              )}

              {(viewMode === 'curl' || viewMode === 'python') && (
                <div>
                  {tryItUnifiedVariant === 'mlflow-invocations' && (
                    <>
                      {viewMode === 'curl' &&
                        renderCodeExample(
                          'cURL',
                          getCodeExamples(base, endpointName, 'mlflow-invocations').curl,
                          'text',
                        )}
                      {viewMode === 'python' &&
                        renderCodeExample(
                          'Python',
                          getCodeExamples(base, endpointName, 'mlflow-invocations').python,
                          'python',
                        )}
                    </>
                  )}
                  {tryItUnifiedVariant === 'chat-completions' && (
                    <>
                      {viewMode === 'curl' &&
                        renderCodeExample('cURL', getCodeExamples(base, endpointName, 'chat-completions').curl, 'text')}
                      {viewMode === 'python' &&
                        renderCodeExample(
                          'Python',
                          getCodeExamples(base, endpointName, 'chat-completions').python,
                          'python',
                        )}
                    </>
                  )}
                </div>
              )}
            </div>
          </Tabs.Content>

          <Tabs.Content value="passthrough" css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              <div>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="Provider" description="Provider selector label" />
                </Typography.Text>
                <SegmentedControlGroup
                  name="try-it-provider"
                  componentId="mlflow.gateway.usage-modal.try-it.provider"
                  value={selectedProvider}
                  onChange={({ target: { value } }) => setSelectedProvider(value as Provider)}
                  css={{ marginBottom: theme.spacing.sm }}
                >
                  <SegmentedControlButton value="openai">OpenAI</SegmentedControlButton>
                  <SegmentedControlButton value="anthropic">Anthropic</SegmentedControlButton>
                  <SegmentedControlButton value="gemini">Google Gemini</SegmentedControlButton>
                </SegmentedControlGroup>
              </div>

              {selectedProvider === 'openai' && (
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Direct access to OpenAI's Responses API for multi-turn conversations with vision and audio capabilities."
                    description="OpenAI passthrough description"
                  />
                </Typography.Text>
              )}
              {selectedProvider === 'anthropic' && (
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Direct access to Anthropic's Messages API with Claude-specific features."
                    description="Anthropic passthrough description"
                  />
                </Typography.Text>
              )}
              {selectedProvider === 'gemini' && (
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Direct access to Google's Gemini API. Note: endpoint name is part of the URL path."
                    description="Gemini passthrough description"
                  />
                </Typography.Text>
              )}

              <SegmentedControlGroup
                name="passthrough-view-mode"
                componentId="mlflow.gateway.usage-modal.passthrough-view-mode"
                value={viewMode}
                onChange={({ target: { value } }) => setViewMode(value as ViewMode)}
                css={{ marginBottom: theme.spacing.sm }}
              >
                <SegmentedControlButton value="try-it">
                  <FormattedMessage defaultMessage="Try it" description="Try it - interactive request/response" />
                </SegmentedControlButton>
                <SegmentedControlButton value="curl">cURL</SegmentedControlButton>
                <SegmentedControlButton value="python">Python</SegmentedControlButton>
              </SegmentedControlGroup>

              {viewMode === 'try-it' && (
                <TryItPanel
                  key={`try-it-passthrough-${tryItResetKey}`}
                  description={
                    <FormattedMessage
                      defaultMessage="Edit the request body below and click Send request to call the provider API."
                      description="Try it description for passthrough"
                    />
                  }
                  requestTooltipContent={
                    <FormattedMessage
                      defaultMessage="JSON body for the {providerName} API. This payload is sent directly to the provider in its native format."
                      description="Request body tooltip for passthrough provider"
                      values={{
                        providerName: {
                          openai: 'OpenAI',
                          anthropic: 'Anthropic',
                          gemini: 'Google Gemini',
                        }[selectedProvider],
                      }}
                    />
                  }
                  requestTooltipComponentId="mlflow.gateway.usage-modal.try-it.request-tooltip-passthrough"
                  tryItRequestUrl={tryItRequestUrl}
                  tryItDefaultBody={tryItDefaultBody}
                />
              )}

              {(viewMode === 'curl' || viewMode === 'python') && (
                <div>
                  {selectedProvider === 'openai' &&
                    viewMode === 'curl' &&
                    renderCodeExample('cURL', getCodeExamples(base, endpointName, 'openai').curl, 'text')}
                  {selectedProvider === 'openai' &&
                    viewMode === 'python' &&
                    renderCodeExample('Python', getCodeExamples(base, endpointName, 'openai').python, 'python')}
                  {selectedProvider === 'anthropic' &&
                    viewMode === 'curl' &&
                    renderCodeExample('cURL', getCodeExamples(base, endpointName, 'anthropic').curl, 'text')}
                  {selectedProvider === 'anthropic' &&
                    viewMode === 'python' &&
                    renderCodeExample('Python', getCodeExamples(base, endpointName, 'anthropic').python, 'python')}
                  {selectedProvider === 'gemini' &&
                    viewMode === 'curl' &&
                    renderCodeExample('cURL', getCodeExamples(base, endpointName, 'gemini').curl, 'text')}
                  {selectedProvider === 'gemini' &&
                    viewMode === 'python' &&
                    renderCodeExample('Python', getCodeExamples(base, endpointName, 'gemini').python, 'python')}
                </div>
              )}
            </div>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </Modal>
  );
};
