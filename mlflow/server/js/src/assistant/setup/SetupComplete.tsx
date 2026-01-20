/**
 * Setup complete screen for MLflow Assistant.
 */

import {
  Button,
  Card,
  Typography,
  useDesignSystemTheme,
  SparkleIcon,
  WrenchSparkleIcon,
  CheckCircleIcon,
} from '@databricks/design-system';

const SUGGESTIONS = [
  'What does this trace show?',
  'Debug the error in this trace.',
  'What is the performance bottleneck?',
];

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

        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            width: '100%',
            maxWidth: 400,
            margin: '0 auto',
          }}
        >
          {SUGGESTIONS.map((suggestion) => (
            <Card
              key={suggestion}
              componentId="mlflow.assistant.setup.complete.suggestion"
              onClick={onStartChatting}
              css={{
                cursor: 'pointer',
                padding: theme.spacing.sm,
                textAlign: 'left',
                transition: 'all 0.2s ease',
                '&:hover': {
                  borderColor: theme.colors.actionPrimaryBackgroundDefault,
                },
              }}
            >
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <SparkleIcon color="ai" css={{ fontSize: 16, flexShrink: 0 }} />
                <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>"{suggestion}"</Typography.Text>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};
