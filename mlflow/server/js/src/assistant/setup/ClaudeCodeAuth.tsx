import { useCallback, useEffect, useState } from 'react';
import { Button, CheckCircleIcon, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';

import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { checkProviderHealth } from '../AssistantService';
import type { AuthState } from '../types';

interface ClaudeCodeAuthProps {
  cachedAuthStatus?: AuthState;
  onAuthStatusChange: (status: AuthState) => void;
  onBack: () => void;
  onContinue: () => void;
}

export const ClaudeCodeAuth = ({ cachedAuthStatus, onAuthStatusChange, onBack, onContinue }: ClaudeCodeAuthProps) => {
  const { theme } = useDesignSystemTheme();
  const [authState, setAuthState] = useState<AuthState>(cachedAuthStatus ?? 'checking');
  const [error, setError] = useState<string | null>(null);

  const runHealthCheck = useCallback(async () => {
    setAuthState('checking');
    setError(null);
    try {
      const result = await checkProviderHealth('claude_code');
      if (result.ok) {
        setAuthState('authenticated');
        onAuthStatusChange('authenticated');
        return;
      }
      const nextState = result.status === 412 ? 'cli_not_installed' : 'not_authenticated';
      setAuthState(nextState);
      setError(result.error ?? null);
      onAuthStatusChange(nextState);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check status');
      setAuthState('not_authenticated');
      onAuthStatusChange('not_authenticated');
    }
  }, [onAuthStatusChange]);

  useEffect(() => {
    if (!cachedAuthStatus) {
      runHealthCheck();
    }
  }, [cachedAuthStatus, runHealthCheck]);

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
        <Typography.Text color="secondary">Checking connection...</Typography.Text>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          This may take several seconds...
        </Typography.Text>
      </div>
    );
  } else if (authState === 'cli_not_installed') {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.grey300, fontSize: 18 }} />
          <span
            css={{
              fontSize: theme.typography.fontSizeMd,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            Install Claude Code CLI
          </span>
        </div>
        <Typography.Text color="secondary">
          The Claude Code CLI is required but not installed on your system.
        </Typography.Text>
        <Typography.Text>Install it by running this command in your terminal:</Typography.Text>
        {renderCodeBlock('npm install -g @anthropic-ai/claude-code')}
        <Typography.Text color="secondary">
          After installation, click &quot;Check Again&quot; to verify.
        </Typography.Text>
        <Typography.Link
          componentId="mlflow.assistant.setup.connection.learn_more"
          href="https://docs.anthropic.com/en/docs/claude-code"
          target="_blank"
        >
          Learn more about Claude Code
        </Typography.Link>
      </div>
    );
  } else if (authState === 'not_authenticated') {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
          <Typography.Text>Claude Code CLI Installed</Typography.Text>
        </div>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <CheckCircleIcon css={{ color: theme.colors.grey300, fontSize: 18 }} />
          <span
            css={{
              fontSize: theme.typography.fontSizeMd,
              fontWeight: theme.typography.typographyBoldFontWeight,
            }}
          >
            Authenticate
          </span>
        </div>
        <Typography.Text color="secondary">
          Please log in to Claude Code by running this command in your terminal:
        </Typography.Text>
        {renderCodeBlock('claude login')}
        <Typography.Text color="secondary">This will open your browser to complete authentication.</Typography.Text>
        <Typography.Text color="secondary">After logging in, click &quot;Check Again&quot; to verify.</Typography.Text>
        {error && (
          <Typography.Text color="error" css={{ fontSize: theme.typography.fontSizeSm }}>
            {error}
          </Typography.Text>
        )}
      </div>
    );
  } else {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {['Claude Code CLI Found', 'Connection Verified'].map((item) => (
            <div key={item} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess, fontSize: 20 }} />
              <Typography.Text>{item}</Typography.Text>
            </div>
          ))}
        </div>
        <Typography.Text color="secondary" css={{ marginTop: theme.spacing.sm }}>
          You&apos;re connected and ready to continue!
        </Typography.Text>
      </div>
    );
  }

  const actionButton =
    authState === 'authenticated' ? (
      <Button componentId="mlflow.assistant.setup.connection.continue" type="primary" onClick={onContinue}>
        Continue
      </Button>
    ) : (
      <Button
        componentId="mlflow.assistant.setup.connection.check_again"
        type="primary"
        onClick={runHealthCheck}
        disabled={authState === 'checking'}
      >
        Check Again
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
