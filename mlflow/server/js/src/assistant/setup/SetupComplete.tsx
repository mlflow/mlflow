/**
 * Setup complete screen for MLflow Assistant.
 */

import {
  Button,
  Typography,
  useDesignSystemTheme,
  CheckCircleIcon,
  WrenchSparkleIcon,
} from '@databricks/design-system';

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
        paddingBottom: theme.spacing.lg * 3,
      }}
    >
      <CheckCircleIcon css={{ fontSize: 48, color: theme.colors.textValidationSuccess }} />

      <Typography.Title level={4}>Setup Complete!</Typography.Title>

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginLeft: theme.spacing.sm }}>
        <Typography.Text color="secondary">You're all set to use the MLflow Assistant.</Typography.Text>
        <Typography.Text color="secondary">
          Ask questions about your experiments, traces, evaluations, and more.
        </Typography.Text>
      </div>

      <Button componentId="mlflow.assistant.setup.complete.start_chatting" type="primary" onClick={onStartChatting}>
        Start Chatting
      </Button>
    </div>
  );
};
