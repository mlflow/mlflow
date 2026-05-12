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

import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { checkProviderHealth, listProviderModels, updateConfig } from '../AssistantService';
import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import type { AuthState } from '../types';

interface SetupStepAuthProps {
  provider: string;
  cachedAuthStatus?: AuthState;
  onAuthStatusChange: (status: AuthState) => void;
  onBack: () => void;
  onContinue: () => void;
}

interface ProviderSetupConfig {
  defaultBaseUrl?: string;
  requiresBaseUrl?: boolean;
  supportsModelSelection?: boolean;
  connectLabel: string;
}

const PROVIDER_CONFIG = {
  claude_code: {
    connectLabel: 'Check Again',
  },
  ollama: {
    defaultBaseUrl: 'http://localhost:11434',
    requiresBaseUrl: true,
    supportsModelSelection: true,
    connectLabel: 'Connect',
  },
} satisfies Record<string, ProviderSetupConfig>;

const DEFAULT_PROVIDER_CONFIG: ProviderSetupConfig = {
  connectLabel: 'Check Again',
};

export const SetupStepAuth = ({
  provider,
  cachedAuthStatus,
  onAuthStatusChange,
  onBack,
  onContinue,
}: SetupStepAuthProps) => {
  const { theme } = useDesignSystemTheme();
  const { config } = useAssistantConfigQuery();
  const providerConfig = (PROVIDER_CONFIG as Record<string, ProviderSetupConfig | undefined>)[provider] ?? DEFAULT_PROVIDER_CONFIG;

  const [authState, setAuthState] = useState<AuthState>(cachedAuthStatus ?? 'checking');
  const [error, setError] = useState<string | null>(null);
  const [providerBaseUrl, setProviderBaseUrl] = useState(providerConfig.defaultBaseUrl ?? '');
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isFetchingModels, setIsFetchingModels] = useState(false);

  useEffect(() => {
    const configuredBaseUrl = config?.providers?.[provider]?.base_url;
    setProviderBaseUrl(configuredBaseUrl ?? providerConfig.defaultBaseUrl ?? '');
    setSelectedModel(config?.providers?.[provider]?.model ?? '');
  }, [config, provider, providerConfig.defaultBaseUrl]);

  const hasProviderModelSelection = providerConfig.supportsModelSelection === true;
  const requiresProviderBaseUrl = providerConfig.requiresBaseUrl === true;

  const persistProviderBaseUrl = useCallback(async () => {
    if (!requiresProviderBaseUrl || !providerBaseUrl) {
      return;
    }

    await updateConfig({ providers: { [provider]: { base_url: providerBaseUrl } } });
  }, [provider, providerBaseUrl, requiresProviderBaseUrl]);

  const runProviderHealthCheck = useCallback(async () => {
    setAuthState('checking');
    setError(null);

    try {
      if (hasProviderModelSelection) {
        setIsFetchingModels(true);
        try {
          const models = await listProviderModels(provider, providerBaseUrl);
          setProviderModels(models);
          setSelectedModel((currentModel) => {
            if (currentModel && models.includes(currentModel)) {
              return currentModel;
            }
            return models[0] ?? '';
          });
          await persistProviderBaseUrl();
          setAuthState('authenticated');
          onAuthStatusChange('authenticated');
        } catch (err) {
          setProviderModels([]);
          setError(err instanceof Error ? err.message : 'Failed to check provider status');
          setAuthState('not_authenticated');
          onAuthStatusChange('not_authenticated');
        } finally {
          setIsFetchingModels(false);
        }
        return;
      }

      const result = await checkProviderHealth(provider);
      if (result.ok) {
        setAuthState('authenticated');
        onAuthStatusChange('authenticated');
        return;
      }

      const nextState = result.status === 412 ? 'cli_not_installed' : 'not_authenticated';
      setAuthState(nextState);
      setError(result.error);
      onAuthStatusChange(nextState);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check status');
      setAuthState('not_authenticated');
      onAuthStatusChange('not_authenticated');
    }
  }, [hasProviderModelSelection, onAuthStatusChange, persistProviderBaseUrl, provider, providerBaseUrl]);

  useEffect(() => {
    if (cachedAuthStatus) {
      return;
    }

    if (hasProviderModelSelection) {
      setAuthState('not_authenticated');
      onAuthStatusChange('not_authenticated');
      return;
    }

    runProviderHealthCheck();
  }, [cachedAuthStatus, hasProviderModelSelection, onAuthStatusChange, runProviderHealthCheck]);

  const handleContinue = useCallback(async () => {
    if (selectedModel) {
      await updateConfig({ providers: { [provider]: { model: selectedModel, selected: true } } });
    }
    onContinue();
  }, [onContinue, provider, selectedModel]);

  const renderCodeBlock = (code: string) => (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        fontFamily: 'monospace',
        fontSize: theme.typography.fontSizeSm,
        marginTop: theme.spacing.md,
        marginBottom: theme.spacing.md,
      }}
    >
      <code>{code}</code>
      <CopyButton copyText={code} showLabel={false} size="small" componentId="mlflow.assistant.setup.connection.copy" />
    </div>
  );

  const renderCheckingState = (message: string, detail?: string) => (
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
      <Typography.Text color="secondary">{message}</Typography.Text>
      {detail && (
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          {detail}
        </Typography.Text>
      )}
    </div>
  );

  const renderAuthenticatedChecks = (items: string[]) => (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {items.map((item) => (
        <div key={item} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
          <Typography.Text>{item}</Typography.Text>
        </div>
      ))}
    </div>
  );

  const renderClaudeCodeContent = () => {
    if (authState === 'checking') {
      return renderCheckingState('Checking connection...', 'This may take several seconds...');
    }

    if (authState === 'cli_not_installed') {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CheckCircleIcon css={{ color: theme.colors.grey300, fontSize: 18 }} />
            <span
              css={{ fontSize: theme.typography.fontSizeMd, fontWeight: theme.typography.typographyBoldFontWeight }}
            >
              Install Claude Code CLI
            </span>
          </div>
          <Typography.Text color="secondary">
            The Claude Code CLI is required but not installed on your system.
          </Typography.Text>
          <Typography.Text>Install it by running this command in your terminal:</Typography.Text>
          {renderCodeBlock('npm install -g @anthropic-ai/claude-code')}
          <Typography.Text color="secondary">After installation, click "Check Again" to verify.</Typography.Text>
          <Typography.Link
            componentId="mlflow.assistant.setup.connection.learn_more"
            href="https://docs.anthropic.com/en/docs/claude-code"
            target="_blank"
          >
            Learn more about Claude Code
          </Typography.Link>
        </div>
      );
    }

    if (authState === 'not_authenticated') {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
            <Typography.Text>Claude Code CLI Installed</Typography.Text>
          </div>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CheckCircleIcon css={{ color: theme.colors.grey300, fontSize: 18 }} />
            <span
              css={{ fontSize: theme.typography.fontSizeMd, fontWeight: theme.typography.typographyBoldFontWeight }}
            >
              Authenticate
            </span>
          </div>
          <Typography.Text color="secondary">
            Please log in to Claude Code by running this command in your terminal:
          </Typography.Text>
          {renderCodeBlock('claude login')}
          <Typography.Text color="secondary">This will open your browser to complete authentication.</Typography.Text>
          <Typography.Text color="secondary">After logging in, click "Check Again" to verify.</Typography.Text>
          {error && (
            <Typography.Text color="error" css={{ fontSize: theme.typography.fontSizeSm }}>
              {error}
            </Typography.Text>
          )}
        </div>
      );
    }

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {renderAuthenticatedChecks(['Claude Code CLI Found', 'Connection Verified'])}
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.sm }}>
          You&apos;re connected and ready to continue!
        </Typography.Text>
      </div>
    );
  };

  const renderProviderModelSelector = () => {
    if (!hasProviderModelSelection) {
      return null;
    }

    return (
      <div css={{ marginTop: theme.spacing.sm }}>
        <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm }}>Select a model:</Typography.Text>
        {isFetchingModels ? (
          <Spinner size="small" />
        ) : providerModels.length > 0 ? (
          <SimpleSelect
            id="mlflow.assistant.setup.provider.model"
            componentId="mlflow.assistant.setup.provider.model"
            value={selectedModel}
            onChange={({ target }) => setSelectedModel(target.value)}
            css={{ width: '100%' }}
          >
            {providerModels.map((model) => (
              <SimpleSelectOption key={model} value={model}>
                {model}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        ) : (
          <Typography.Text color="secondary">
            No models found. Pull a model first with: <code>ollama pull llama3.2</code>
          </Typography.Text>
        )}
      </div>
    );
  };

  const renderProviderConnectionForm = () => {
    if (!requiresProviderBaseUrl) {
      return null;
    }

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">Enter the URL of your running Ollama server:</Typography.Text>
        <Input
          componentId="mlflow.assistant.setup.provider.url"
          value={providerBaseUrl}
          onChange={(event) => setProviderBaseUrl(event.target.value)}
          placeholder={providerConfig.defaultBaseUrl}
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
  };

  const renderOllamaContent = () => {
    if (authState === 'checking') {
      return renderCheckingState('Connecting to Ollama...');
    }

    if (authState === 'authenticated') {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {renderAuthenticatedChecks([`Connected to Ollama at ${providerBaseUrl}`])}
          {renderProviderModelSelector()}
        </div>
      );
    }

    return renderProviderConnectionForm();
  };

  const content = hasProviderModelSelection ? renderOllamaContent() : renderClaudeCodeContent();

  const continueDisabled = hasProviderModelSelection
    ? authState !== 'authenticated' || isFetchingModels || providerModels.length === 0 || !selectedModel
    : authState !== 'authenticated';

  const actionButton =
    hasProviderModelSelection && authState === 'authenticated' ? (
      <Button
        componentId="mlflow.assistant.setup.connection.continue"
        type="primary"
        onClick={handleContinue}
        disabled={continueDisabled}
      >
        Continue
      </Button>
    ) : !hasProviderModelSelection && authState === 'authenticated' ? (
      <Button componentId="mlflow.assistant.setup.connection.continue" type="primary" onClick={onContinue}>
        Continue
      </Button>
    ) : hasProviderModelSelection ? (
      <Button
        componentId="mlflow.assistant.setup.connection.connect"
        type="primary"
        onClick={runProviderHealthCheck}
        disabled={authState === 'checking' || !providerBaseUrl}
      >
        {providerConfig.connectLabel}
      </Button>
    ) : (
      <Button
        componentId="mlflow.assistant.setup.connection.check_again"
        type="primary"
        onClick={runProviderHealthCheck}
        disabled={authState === 'checking'}
      >
        {providerConfig.connectLabel}
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
