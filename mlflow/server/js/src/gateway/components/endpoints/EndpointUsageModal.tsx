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
import { getDefaultHeaders } from '../../../common/utils/FetchUtils';

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

// Default request body for MLflow Invocations (no endpoint name in body). Used for initial state and by getCodeExamples.
const MLFLOW_INVOCATIONS_DEFAULT_BODY = JSON.stringify(
  {
    messages: [{ role: 'user', content: 'Hello, how are you?' }],
  },
  null,
  2,
);

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
        defaultBody: MLFLOW_INVOCATIONS_DEFAULT_BODY,
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

export const EndpointUsageModal = ({ open, onClose, endpointName, baseUrl }: EndpointUsageModalProps) => {
  const { theme } = useDesignSystemTheme();
  const [activeTab, setActiveTab] = useState<'try-it' | 'unified' | 'passthrough'>('try-it');
  const [selectedProvider, setSelectedProvider] = useState<Provider>('openai');
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('curl');
  const [unifiedLanguage, setUnifiedLanguage] = useState<Language>('curl');
  const [tryItApiType, setTryItApiType] = useState<'unified' | 'passthrough'>('unified');
  const [tryItUnifiedVariant, setTryItUnifiedVariant] = useState<'mlflow-invocations' | 'chat-completions'>(
    'mlflow-invocations',
  );
  const [tryItProvider, setTryItProvider] = useState<Provider>('openai');
  const [requestBody, setRequestBody] = useState(MLFLOW_INVOCATIONS_DEFAULT_BODY);
  const [responseBody, setResponseBody] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const base = getBaseUrl(baseUrl);

  // Reset modal state when opened so users get a fresh Try-it experience each time
  useEffect(() => {
    if (open) {
      setActiveTab('try-it');
      setTryItApiType('unified');
      setTryItUnifiedVariant('mlflow-invocations');
      setTryItProvider('openai');
      setRequestBody(MLFLOW_INVOCATIONS_DEFAULT_BODY);
      setResponseBody('');
      setSendError(null);
    }
  }, [open]);

  const tryItRequestUrl = useMemo(
    () => getTryItRequestUrl(base, endpointName, tryItApiType, tryItUnifiedVariant, tryItProvider),
    [base, endpointName, tryItApiType, tryItUnifiedVariant, tryItProvider],
  );

  const tryItDefaultBody = useMemo(
    () => getDefaultRequestBody(endpointName, tryItApiType === 'unified' ? tryItUnifiedVariant : tryItProvider),
    [tryItApiType, tryItUnifiedVariant, tryItProvider, endpointName],
  );

  const handleSendRequest = useCallback(async () => {
    setSendError(null);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(requestBody);
    } catch {
      setSendError('Invalid JSON in request body');
      setResponseBody('');
      return;
    }
    setIsSending(true);
    setResponseBody('');
    try {
      const response = await fetch(tryItRequestUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getDefaultHeaders(document.cookie),
        },
        body: JSON.stringify(parsed),
      });
      const text = await response.text();
      if (!response.ok) {
        setSendError(`Request failed (${response.status})`);
        setResponseBody(text || '');
        return;
      }
      try {
        const formatted = JSON.stringify(JSON.parse(text), null, 2);
        setResponseBody(formatted);
      } catch {
        setResponseBody(text);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSendError(message);
      setResponseBody('');
    } finally {
      setIsSending(false);
    }
  }, [requestBody, tryItRequestUrl]);

  const handleResetExample = useCallback(() => {
    setRequestBody(tryItDefaultBody);
    setResponseBody('');
    setSendError(null);
  }, [tryItDefaultBody]);
  const renderCodeExample = (label: string, code: string, language: 'text' | 'python' = 'text') => (
    <div css={{ marginBottom: theme.spacing.md }}>
      <div
        css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}
      >
        <Typography.Text bold>{label}</Typography.Text>
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
          onValueChange={(value) => setActiveTab(value as 'try-it' | 'unified' | 'passthrough')}
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
            <Tabs.Trigger value="try-it">
              <FormattedMessage defaultMessage="Try it" description="Try it tab - interactive request/response" />
            </Tabs.Trigger>
            <Tabs.Trigger value="unified">
              <FormattedMessage defaultMessage="Unified APIs" description="Unified APIs tab title" />
            </Tabs.Trigger>
            <Tabs.Trigger value="passthrough">
              <FormattedMessage defaultMessage="Passthrough APIs" description="Passthrough APIs tab title" />
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="try-it" css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Edit the request body below and click Send request to call the endpoint. Choose Unified for the MLflow Invocations API, or Passthrough for provider-specific APIs (OpenAI, Anthropic, Gemini)."
                  description="Try it tab description"
                />
              </Typography.Text>
              <div>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="API" description="API type selector label" />
                </Typography.Text>
                <SegmentedControlGroup
                  name="try-it-api-type"
                  componentId="mlflow.gateway.usage-modal.try-it.api-type"
                  value={tryItApiType}
                  onChange={({ target: { value } }) => {
                    setTryItApiType(value as 'unified' | 'passthrough');
                    setRequestBody(
                      getDefaultRequestBody(endpointName, value === 'unified' ? tryItUnifiedVariant : tryItProvider),
                    );
                    setResponseBody('');
                    setSendError(null);
                  }}
                  css={{ marginBottom: theme.spacing.sm }}
                >
                  <SegmentedControlButton value="unified">
                    <FormattedMessage defaultMessage="Unified" description="Unified API option" />
                  </SegmentedControlButton>
                  <SegmentedControlButton value="passthrough">
                    <FormattedMessage defaultMessage="Passthrough" description="Passthrough API option" />
                  </SegmentedControlButton>
                </SegmentedControlGroup>
              </div>
              {tryItApiType === 'unified' && (
                <div>
                  <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="Unified API" description="Unified API variant label" />
                  </Typography.Text>
                  <SegmentedControlGroup
                    name="try-it-unified-variant"
                    componentId="mlflow.gateway.usage-modal.try-it.unified-variant"
                    value={tryItUnifiedVariant}
                    onChange={({ target: { value } }) => {
                      setTryItUnifiedVariant(value as 'mlflow-invocations' | 'chat-completions');
                      setRequestBody(
                        getDefaultRequestBody(endpointName, value as 'mlflow-invocations' | 'chat-completions'),
                      );
                      setResponseBody('');
                      setSendError(null);
                    }}
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
              )}
              {tryItApiType === 'passthrough' && (
                <div>
                  <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="Provider" description="Provider selector label" />
                  </Typography.Text>
                  <SegmentedControlGroup
                    name="try-it-provider"
                    componentId="mlflow.gateway.usage-modal.try-it.provider"
                    value={tryItProvider}
                    onChange={({ target: { value } }) => {
                      const provider = value as Provider;
                      setTryItProvider(provider);
                      setRequestBody(getDefaultRequestBody(endpointName, provider));
                      setResponseBody('');
                      setSendError(null);
                    }}
                    css={{ marginBottom: theme.spacing.sm }}
                  >
                    <SegmentedControlButton value="openai">OpenAI</SegmentedControlButton>
                    <SegmentedControlButton value="anthropic">Anthropic</SegmentedControlButton>
                    <SegmentedControlButton value="gemini">Google Gemini</SegmentedControlButton>
                  </SegmentedControlGroup>
                </div>
              )}
              <div
                css={{
                  display: 'flex',
                  gap: theme.spacing.md,
                  minHeight: 0,
                  flex: 1,
                }}
              >
                <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      marginBottom: theme.spacing.xs,
                    }}
                  >
                    <Typography.Text bold>
                      <FormattedMessage defaultMessage="Request" description="Request body label" />
                    </Typography.Text>
                    <Tooltip
                      componentId="mlflow.gateway.usage-modal.try-it.request-tooltip"
                      content={
                        tryItApiType === 'unified' ? (
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
                        ) : (
                          <FormattedMessage
                            defaultMessage="JSON body for the {providerName} API. This payload is sent directly to the provider in its native format."
                            description="Request body tooltip for passthrough provider"
                            values={{
                              providerName: { openai: 'OpenAI', anthropic: 'Anthropic', gemini: 'Google Gemini' }[
                                tryItProvider
                              ],
                            }}
                          />
                        )
                      }
                    >
                      <span css={{ cursor: 'help', color: theme.colors.textSecondary }} aria-label="Request help">
                        ?
                      </span>
                    </Tooltip>
                  </div>
                  <Input.TextArea
                    componentId="mlflow.gateway.usage-modal.try-it.request"
                    value={requestBody}
                    onChange={(e) => setRequestBody(e.target.value)}
                    disabled={isSending}
                    rows={14}
                    css={{
                      fontFamily: 'monospace',
                      fontSize: theme.typography.fontSizeSm,
                      minHeight: 220,
                    }}
                  />
                </div>
                <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.xs,
                      marginBottom: theme.spacing.xs,
                    }}
                  >
                    <Typography.Text bold>
                      <FormattedMessage defaultMessage="Response" description="Response body label" />
                    </Typography.Text>
                    <Tooltip
                      componentId="mlflow.gateway.usage-modal.try-it.response-tooltip"
                      content={
                        <FormattedMessage
                          defaultMessage="Response from the endpoint after clicking Send request."
                          description="Response body tooltip"
                        />
                      }
                    >
                      <span css={{ cursor: 'help', color: theme.colors.textSecondary }} aria-label="Response help">
                        ?
                      </span>
                    </Tooltip>
                  </div>
                  <Input.TextArea
                    componentId="mlflow.gateway.usage-modal.try-it.response"
                    value={responseBody}
                    readOnly
                    rows={14}
                    placeholder={
                      sendError ? undefined : isSending ? undefined : 'Click "Send request" to see the response here.'
                    }
                    css={{
                      fontFamily: 'monospace',
                      fontSize: theme.typography.fontSizeSm,
                      backgroundColor: theme.colors.backgroundSecondary,
                      minHeight: 220,
                    }}
                  />
                  {sendError && (
                    <Typography.Text color="error" css={{ marginTop: theme.spacing.xs }}>
                      {sendError}
                    </Typography.Text>
                  )}
                </div>
              </div>
              <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                <Button
                  componentId="mlflow.gateway.usage-modal.try-it.send"
                  type="primary"
                  onClick={handleSendRequest}
                  disabled={isSending}
                  loading={isSending}
                >
                  {isSending ? (
                    <FormattedMessage defaultMessage="Sending..." description="Send request button loading state" />
                  ) : (
                    <FormattedMessage defaultMessage="Send request" description="Send request button" />
                  )}
                </Button>
                <Button
                  componentId="mlflow.gateway.usage-modal.try-it.reset"
                  onClick={handleResetExample}
                  disabled={isSending}
                >
                  <FormattedMessage defaultMessage="Reset example" description="Reset example button" />
                </Button>
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content value="unified" css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              <SegmentedControlGroup
                name="unified-language-selector"
                componentId="mlflow.gateway.usage-modal.unified-language-selector"
                value={unifiedLanguage}
                onChange={({ target: { value } }) => setUnifiedLanguage(value as Language)}
              >
                <SegmentedControlButton value="curl">cURL</SegmentedControlButton>
                <SegmentedControlButton value="python">Python</SegmentedControlButton>
              </SegmentedControlGroup>

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
                {unifiedLanguage === 'curl' &&
                  renderCodeExample('cURL', getCodeExamples(base, endpointName, 'mlflow-invocations').curl, 'text')}
                {unifiedLanguage === 'python' &&
                  renderCodeExample(
                    'Python',
                    getCodeExamples(base, endpointName, 'mlflow-invocations').python,
                    'python',
                  )}
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
                {unifiedLanguage === 'curl' &&
                  renderCodeExample('cURL', getCodeExamples(base, endpointName, 'chat-completions').curl, 'text')}
                {unifiedLanguage === 'python' &&
                  renderCodeExample('Python', getCodeExamples(base, endpointName, 'chat-completions').python, 'python')}
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content value="passthrough" css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              <div>
                <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                  <FormattedMessage defaultMessage="Provider" description="Provider selection label" />
                </Typography.Text>
                <SegmentedControlGroup
                  name="provider-selector"
                  componentId="mlflow.gateway.usage-modal.provider-selector"
                  value={selectedProvider}
                  onChange={({ target: { value } }) => setSelectedProvider(value as Provider)}
                >
                  <SegmentedControlButton value="openai">OpenAI</SegmentedControlButton>
                  <SegmentedControlButton value="anthropic">Anthropic</SegmentedControlButton>
                  <SegmentedControlButton value="gemini">Google Gemini</SegmentedControlButton>
                </SegmentedControlGroup>
              </div>

              <div>
                {selectedProvider === 'openai' && (
                  <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                    <FormattedMessage
                      defaultMessage="Direct access to OpenAI's Responses API for multi-turn conversations with vision and audio capabilities."
                      description="OpenAI passthrough description"
                    />
                  </Typography.Text>
                )}
                {selectedProvider === 'anthropic' && (
                  <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                    <FormattedMessage
                      defaultMessage="Direct access to Anthropic's Messages API with Claude-specific features."
                      description="Anthropic passthrough description"
                    />
                  </Typography.Text>
                )}
                {selectedProvider === 'gemini' && (
                  <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
                    <FormattedMessage
                      defaultMessage="Direct access to Google's Gemini API. Note: endpoint name is part of the URL path."
                      description="Gemini passthrough description"
                    />
                  </Typography.Text>
                )}

                <SegmentedControlGroup
                  name="language-selector"
                  componentId="mlflow.gateway.usage-modal.language-selector"
                  value={selectedLanguage}
                  onChange={({ target: { value } }) => setSelectedLanguage(value as Language)}
                  css={{ marginBottom: theme.spacing.md }}
                >
                  <SegmentedControlButton value="curl">cURL</SegmentedControlButton>
                  <SegmentedControlButton value="python">Python</SegmentedControlButton>
                </SegmentedControlGroup>

                {selectedProvider === 'openai' &&
                  selectedLanguage === 'curl' &&
                  renderCodeExample('cURL', getCodeExamples(base, endpointName, 'openai').curl, 'text')}
                {selectedProvider === 'openai' &&
                  selectedLanguage === 'python' &&
                  renderCodeExample('Python', getCodeExamples(base, endpointName, 'openai').python, 'python')}
                {selectedProvider === 'anthropic' &&
                  selectedLanguage === 'curl' &&
                  renderCodeExample('cURL', getCodeExamples(base, endpointName, 'anthropic').curl, 'text')}
                {selectedProvider === 'anthropic' &&
                  selectedLanguage === 'python' &&
                  renderCodeExample('Python', getCodeExamples(base, endpointName, 'anthropic').python, 'python')}
                {selectedProvider === 'gemini' &&
                  selectedLanguage === 'curl' &&
                  renderCodeExample('cURL', getCodeExamples(base, endpointName, 'gemini').curl, 'text')}
                {selectedProvider === 'gemini' &&
                  selectedLanguage === 'python' &&
                  renderCodeExample('Python', getCodeExamples(base, endpointName, 'gemini').python, 'python')}
              </div>
            </div>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </Modal>
  );
};
