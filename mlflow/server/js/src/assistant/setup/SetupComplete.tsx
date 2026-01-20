/**
 * Setup complete screen for MLflow Assistant.
 */

import { Button, Typography, useDesignSystemTheme, CheckCircleIcon, WrenchSparkleIcon } from '@databricks/design-system';

interface SetupCompleteProps {
  onStartChatting: () => void;
}

export const SetupComplete = ({ onStartChatting }: SetupCompleteProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        textAlign: 'center',
        gap: theme.spacing.lg,
      }}
    >
      <CheckCircleIcon css={{ fontSize: 48, color: theme.colors.textValidationSuccess }} />

      <Typography.Title level={4}>Setup Complete!</Typography.Title>

      <Typography.Text color="secondary">You're all set to use the MLflow Assistant.</Typography.Text>

      <Button componentId="mlflow.assistant.setup.complete.start_chatting" type="primary" onClick={onStartChatting}>
        Start Chatting
      </Button>

      <div
        css={{
          width: '100%',
          borderTop: `1px solid ${theme.colors.border}`,
          paddingTop: theme.spacing.lg,
          marginTop: theme.spacing.md,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.sm,
            marginBottom: theme.spacing.md,
          }}
        >
          <WrenchSparkleIcon color="ai" css={{ fontSize: 20 }} />
          <Typography.Text color="secondary">
            Ask questions about your experiments, traces, evaluations, and more.
          </Typography.Text>
        </div>
      </div>
    </div>
  );
};
