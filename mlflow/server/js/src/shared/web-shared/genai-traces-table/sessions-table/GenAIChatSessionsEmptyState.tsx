import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { CodeSnippet, SnippetCopyAction } from '@databricks/web-shared/snippet';

const EXAMPLE_CODE = `import mlflow

@mlflow.trace
def chat_completion(message: str, user_id: str, session_id: str):
    # Set these metadata keys to associate the trace to a user and session
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )

    # Replace with your chat logic
    return "Your response here"

# Depending on your setup, user and session IDs can be passed to your
# server handler via the network request from your client applications,
# or derived from some other request context. 
user_id = "user-123"
session_id = "session-123"

chat_completion("Hello, how are you?", user_id, session_id)`;

export const GenAIChatSessionsEmptyState = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ flex: 1, flexDirection: 'column', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="Group traces from the same chat session together"
          description="Empty state title for the chat sessions table"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
        <FormattedMessage
          defaultMessage="MLflow allows you associate traces with users and chat sessions. This is useful for analyzing multi-turn conversations, enabling you to inspect what happened at each step. Copy the code snippet below to generate a sample trace, or visit the documentation for a more in-depth example."
          description="Empty state description for the chat sessions table"
        />
        <Typography.Link
          componentId="mlflow.chat_sessions.empty_state.learn_more_link"
          css={{ marginLeft: theme.spacing.xs }}
          href="https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/"
          openInNewTab
        >
          <FormattedMessage
            defaultMessage="Learn more"
            description="Link to the documentation page for user and session tracking"
          />
        </Typography.Link>
      </Typography.Paragraph>
      <div css={{ position: 'relative' }}>
        <SnippetCopyAction
          componentId="mlflow.chat_sessions.empty_state.example_code_copy"
          css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          copyText={EXAMPLE_CODE}
        />
        <CodeSnippet language="python" showLineNumbers>
          {EXAMPLE_CODE}
        </CodeSnippet>
      </div>
    </div>
  );
};
