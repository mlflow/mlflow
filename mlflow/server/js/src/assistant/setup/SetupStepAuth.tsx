import { useCallback, useEffect, useState } from 'react';
import {
  Button,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  CheckCircleIcon,
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

export const SetupStepAuth = ({
  provider,
  cachedAuthStatus,
  onAuthStatusChange,
  onBack,
  onContinue,
}: SetupStepAuthProps) => {
  const { theme } = useDesignSystemTheme();
  const { config } = useAssistantConfigQuery();

  const [authState, setAuthState] = useState<AuthState>(cachedAuthStatus ?? 'checking');
  const [error, setError] = useState<string | null>(null);

  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [ollamaModel, setOllamaModel] = useState<string>('');
  const [isFetchingModels, setIsFetchingModels] = useState(false);

  useEffect(() => {
    if (provider === 'ollama' && config?.providers?.['ollama']?.base_url) {
      setOllamaUrl(config.providers['ollama'].base_url as string);
    }
  }, [provider, config]);

  const runHealthCheck = useCallback(async () => {
    setAuthState('checking');
    setError(null);

    try {
      if (provider === 'ollama') {
        setIsFetchingModels(true);
        try {
          const models = await listProviderModels('ollama', ollamaUrl);
          setOllamaModels(models);
          if (models.length > 0) {
            setOllamaModel(models[0]);
          }
          // Only persist base_url after successful connection
          await updateConfig({ providers: { ollama: { base_url: ollamaUrl } } });
          setAuthState('authenticated');
          onAuthStatusChange('authenticated');
        } catch (err) {
          setOllamaModels([]);
          setError(err instanceof Error ? err.message : 'Failed to check Ollama status');
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
      } else {
        if (result.status === 412) {
          setAuthState('cli_not_installed');
          onAuthStatusChange('cli_not_installed');
        } else {
          setAuthState('not_authenticated');
          onAuthStatusChange('not_authenticated');
        }
        setError(result.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check status');
      setAuthState('not_authenticated');
      onAuthStatusChange('not_authenticated');
    }
  }, [provider, ollamaUrl, onAuthStatusChange]);

  const handleOllamaContinue = useCallback(async () => {
    if (ollamaModel) {
      await updateConfig({ providers: { ollama: { model: ollamaModel, selected: true } } });
    }
    onContinue();
  }, [ollamaModel, onContinue]);

  useEffect(() => {
    if (!cachedAuthStatus) {
      if (provider === 'ollama') {
        setAuthState('not_authenticated');
        onAuthStatusChange('not_authenticated');
      } else {
        runHealthCheck();
      }
    }
  }, [cachedAuthStatus, provider, runHealthCheck, onAuthStatusChange]);

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

  const renderClaudeContent = () => {
    if (authState === 'checking') {
      return (
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
          <Typography.Text color="secondary">Checking connection...</Typography.Text>
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            This may take several seconds...
          </Typography.Text>
        </div>
      );
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
        </div>
      );
    }

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
          <Typography.Text>Claude Code CLI Found</Typography.Text>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
          <Typography.Text>Connection Verified</Typography.Text>
        </div>
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.sm }}>
          You're connected and ready to continue!
        </Typography.Text>
      </div>
    );
  };

  const renderOllamaContent = () => {
    if (authState === 'checking') {
      return (
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
    }

    if (authState === 'authenticated') {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
            <Typography.Text>Connected to Ollama at {ollamaUrl}</Typography.Text>
          </div>
          <div css={{ marginTop: theme.spacing.sm }}>
            <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm }}>
              Select a model:
            </Typography.Text>
            {isFetchingModels ? (
              <Spinner size="small" />
            ) : ollamaModels.length > 0 ? (
              <SimpleSelect
                id="mlflow.assistant.setup.ollama.model"
                componentId="mlflow.assistant.setup.ollama.model"
                value={ollamaModel}
                onChange={({ target }) => setOllamaModel(target.value)}
                css={{ width: '100%' }}
              >
                {ollamaModels.map((m) => (
                  <SimpleSelectOption key={m} value={m}>
                    {m}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            ) : (
              <Typography.Text color="secondary">
                No models found. Pull a model first with: <code>ollama pull llama3.2</code>
              </Typography.Text>
            )}
          </div>
        </div>
      );
    }

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">Enter the URL of your running Ollama server:</Typography.Text>
        <Input
          componentId="mlflow.assistant.setup.ollama.url"
          value={ollamaUrl}
          onChange={(e) => setOllamaUrl(e.target.value)}
          placeholder="http://localhost:11434"
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

  const renderContent = () => {
    if (provider === 'ollama') return renderOllamaContent();
    return renderClaudeContent();
  };

  const isContinueDisabled = () => {
    if (provider === 'ollama') {
      return authState !== 'authenticated' || isFetchingModels || ollamaModels.length === 0 || !ollamaModel;
    }
    return authState !== 'authenticated';
  };

  const handleContinue = provider === 'ollama' ? handleOllamaContinue : onContinue;

  const renderFooterActions = () => {
    if (provider === 'ollama') {
      if (authState === 'authenticated') {
        return (
          <Button
            componentId="mlflow.assistant.setup.connection.continue"
            type="primary"
            onClick={handleContinue}
            disabled={isContinueDisabled()}
          >
            Continue
          </Button>
        );
      }
      return (
        <Button
          componentId="mlflow.assistant.setup.connection.connect"
          type="primary"
          onClick={runHealthCheck}
          disabled={authState === 'checking' || !ollamaUrl}
        >
          Connect
        </Button>
      );
    }

    if (authState === 'authenticated') {
      return (
        <Button componentId="mlflow.assistant.setup.connection.continue" type="primary" onClick={handleContinue}>
          Continue
        </Button>
      );
    }

    return (
      <Button
        componentId="mlflow.assistant.setup.connection.check_again"
        type="primary"
        onClick={runHealthCheck}
        disabled={authState === 'checking'}
      >
        Check Again
      </Button>
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1 }}>{renderContent()}</div>

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
        {renderFooterActions()}
      </div>
    </div>
  );
};
