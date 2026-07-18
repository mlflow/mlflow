import { useCallback, useEffect, useState } from 'react';
import {
  Button,
  CheckCircleIcon,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { listProviderModels, updateConfig } from '../AssistantService';
import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import type { AuthState } from '../types';

const DEFAULT_BASE_URL = 'http://localhost:11434';

interface OllamaAuthProps {
  cachedAuthStatus?: AuthState;
  onAuthStatusChange: (status: AuthState) => void;
  onBack: () => void;
  onContinue: () => void;
}

export const OllamaAuth = ({ cachedAuthStatus, onAuthStatusChange, onBack, onContinue }: OllamaAuthProps) => {
  const { theme } = useDesignSystemTheme();
  const { config } = useAssistantConfigQuery();
  const [authState, setAuthState] = useState<AuthState>(cachedAuthStatus ?? 'not_authenticated');
  const [error, setError] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isFetchingModels, setIsFetchingModels] = useState(false);

  useEffect(() => {
    const configuredBaseUrl = config?.providers?.['ollama']?.base_url;
    setBaseUrl(configuredBaseUrl ?? DEFAULT_BASE_URL);
    setSelectedModel(config?.providers?.['ollama']?.model ?? '');
  }, [config]);

  const connect = useCallback(async () => {
    setAuthState('checking');
    setError(null);
    setIsFetchingModels(true);
    try {
      const fetchedModels = await listProviderModels('ollama', baseUrl);
      setModels(fetchedModels);
      setSelectedModel((current) => {
        if (current && fetchedModels.includes(current)) return current;
        return fetchedModels[0] ?? '';
      });
      await updateConfig({ providers: { ollama: { base_url: baseUrl } } });
      setAuthState('authenticated');
      onAuthStatusChange('authenticated');
    } catch (err) {
      setModels([]);
      setError(err instanceof Error ? err.message : 'Failed to connect to Ollama');
      setAuthState('not_authenticated');
      onAuthStatusChange('not_authenticated');
    } finally {
      setIsFetchingModels(false);
    }
  }, [baseUrl, onAuthStatusChange]);

  const handleContinue = useCallback(async () => {
    if (selectedModel) {
      await updateConfig({ providers: { ollama: { model: selectedModel, selected: true } } });
    }
    onContinue();
  }, [onContinue, selectedModel]);

  let content: React.ReactNode;

  if (authState === 'checking') {
    content = (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg * 2,
          gap: theme.spacing.md,
        }}
      >
        <Spinner size="default" />
        <Typography.Text color="secondary">Connecting to Ollama...</Typography.Text>
      </div>
    );
  } else if (authState === 'authenticated') {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
          <Typography.Text>Connected to Ollama at {baseUrl}</Typography.Text>
        </div>
        <div css={{ marginTop: theme.spacing.sm }}>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm }}>Select a model:</Typography.Text>
          {isFetchingModels ? (
            <Spinner size="small" />
          ) : models.length > 0 ? (
            <SimpleSelect
              id="mlflow.assistant.setup.provider.model"
              componentId="mlflow.assistant.setup.provider.model"
              value={selectedModel}
              onChange={({ target }) => setSelectedModel(target.value)}
              css={{ width: '100%' }}
            >
              {models.map((model) => (
                <SimpleSelectOption key={model} value={model}>
                  {model}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          ) : (
            <Typography.Text color="secondary">
              No models found. Pull a model first (e.g. <code>ollama pull llama3.2</code>), then click Connect again.
              Browse available models at{' '}
              <Typography.Link
                componentId="mlflow.assistant.setup.ollama.link"
                href="https://ollama.com/library"
                target="_blank"
              >
                ollama.com/library
              </Typography.Link>
              .
            </Typography.Text>
          )}
        </div>
      </div>
    );
  } else {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">Enter the URL of your running Ollama server:</Typography.Text>
        <Input
          componentId="mlflow.assistant.setup.provider.url"
          value={baseUrl}
          onChange={(event) => setBaseUrl(event.target.value)}
          placeholder={DEFAULT_BASE_URL}
        />
        {error && (
          <Typography.Text color="error" css={{ fontSize: theme.typography.fontSizeSm }}>
            {error}
          </Typography.Text>
        )}
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          Make sure Ollama is installed and running. Download it at{' '}
          <Typography.Link componentId="mlflow.assistant.setup.ollama.link" href="https://ollama.com" target="_blank">
            ollama.com
          </Typography.Link>
        </Typography.Text>
      </div>
    );
  }

  const continueDisabled = authState !== 'authenticated' || isFetchingModels || models.length === 0 || !selectedModel;

  const actionButton =
    authState === 'authenticated' ? (
      <Button
        componentId="mlflow.assistant.setup.connection.continue"
        type="primary"
        onClick={handleContinue}
        disabled={continueDisabled}
      >
        Continue
      </Button>
    ) : (
      <Button
        componentId="mlflow.assistant.setup.connection.connect"
        type="primary"
        onClick={connect}
        disabled={authState === 'checking' || !baseUrl}
      >
        Connect
      </Button>
    );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1 }}>{content}</div>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: theme.spacing.lg,
          paddingTop: theme.spacing.md,
          borderTop: `1px solid ${theme.colors.border}`,
        }}
      >
        <Button componentId="mlflow.assistant.setup.connection.back" onClick={onBack}>
          Back
        </Button>
        {actionButton}
      </div>
    </div>
  );
};
