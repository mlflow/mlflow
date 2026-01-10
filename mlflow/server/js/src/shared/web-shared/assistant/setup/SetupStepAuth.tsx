/**
 * Step 2: Connection check for MLflow Assistant setup.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  Button,
  Typography,
  useDesignSystemTheme,
  Spinner,
  CheckCircleIcon,
  WarningIcon,
  CopyIcon,
  Tooltip,
} from '@databricks/design-system';

import { checkProviderHealth } from '../AssistantService';

const COMPONENT_ID = 'mlflow.assistant.setup.connection';

type AuthState = 'checking' | 'cli_not_installed' | 'not_authenticated' | 'authenticated';

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

  const [authState, setAuthState] = useState<AuthState>(cachedAuthStatus ?? 'checking');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const runHealthCheck = useCallback(async () => {
    setAuthState('checking');
    setError(null);

    try {
      const result = await checkProviderHealth(provider);

      if (result.ok) {
        setAuthState('authenticated');
        onAuthStatusChange('authenticated');
      } else {
        // Use status code to determine state: 412 = CLI not installed, 401 = not authenticated
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
  }, [provider, onAuthStatusChange]);

  // Check health on mount only if no cached status
  useEffect(() => {
    if (!cachedAuthStatus) {
      runHealthCheck();
    }
  }, [cachedAuthStatus, runHealthCheck]);

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

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
      <Tooltip componentId={`${COMPONENT_ID}.copy.tooltip`} content={copied ? 'Copied!' : 'Copy'}>
        <Button
          componentId={`${COMPONENT_ID}.copy`}
          size="small"
          icon={<CopyIcon />}
          onClick={() => handleCopy(code)}
        />
      </Tooltip>
    </div>
  );

  const renderContent = () => {
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
            <WarningIcon css={{ color: theme.colors.textValidationWarning, fontSize: 20 }} />
            <Typography.Title level={5}>Claude Code CLI Not Found</Typography.Title>
          </div>

          <Typography.Text color="secondary">
            The Claude Code CLI is required but not installed on your system.
          </Typography.Text>

          <Typography.Text>Install it by running this command in your terminal:</Typography.Text>

          {renderCodeBlock('npm install -g @anthropic-ai/claude-code')}

          <Typography.Text color="secondary">After installation, click "Check Again" to verify.</Typography.Text>

          <Typography.Link
            componentId={`${COMPONENT_ID}.learn_more`}
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
            <WarningIcon css={{ color: theme.colors.textValidationWarning, fontSize: 20 }} />
            <Typography.Title level={5}>Authentication Required</Typography.Title>
          </div>

          <Typography.Text color="secondary">
            Please log in to Claude Code by running this command in your terminal:
          </Typography.Text>

          {renderCodeBlock('claude login')}

          <Typography.Text color="secondary">This will open your browser to complete authentication.</Typography.Text>

          <Typography.Text color="secondary">After logging in, click "Check Again" to verify.</Typography.Text>

          {error && <Typography.Text css={{ color: theme.colors.textValidationDanger }}>{error}</Typography.Text>}
        </div>
      );
    }

    // authenticated
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
        <Button componentId={`${COMPONENT_ID}.back`} onClick={onBack}>
          Back
        </Button>

        {authState === 'authenticated' ? (
          <Button componentId={`${COMPONENT_ID}.continue`} type="primary" onClick={onContinue}>
            Continue
          </Button>
        ) : (
          <Button
            componentId={`${COMPONENT_ID}.check_again`}
            type="primary"
            onClick={runHealthCheck}
            disabled={authState === 'checking'}
          >
            Check Again
          </Button>
        )}
      </div>
    </div>
  );
};
