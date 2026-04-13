import { useState } from 'react';
import { CopyIcon, Typography, useDesignSystemTheme, type ThemeType } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import sessionsVideo from '@mlflow/mlflow/src/common/static/video/chat-sessions-demo.mp4';

type Language = 'python' | 'typescript';

const PYTHON_CODE = `import mlflow

@mlflow.trace
def chat_completion(message: list[dict], user_id: str, session_id: str):
    """Process a chat message with user and session tracking."""

    # Add user and session context to the current trace
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,  # Links trace to specific user
            "mlflow.trace.session": session_id,  # Groups trace with conversation
        }
    )

    # Your chat logic here
    return generate_response(message)`;

const TS_INSTALL_CODE = 'npm install mlflow-tracing';

const TS_CODE = `import * as mlflow from "@mlflow/core";

const chatCompletion = mlflow.trace(
    (message: Array<Record<string, any>>, userId: string, sessionId: string) => {
        // Add user and session context to the current trace
        mlflow.updateCurrentTrace({
            metadata: {
                "mlflow.trace.user": userId,
                "mlflow.trace.session": sessionId,
            },
        });

        // Your chat logic here
        return generateResponse(message);
    },
    { name: "chat_completion" }
);`;

export const SessionsEmptyState = () => {
  const { theme } = useDesignSystemTheme();
  const [language, setLanguage] = useState<Language>('python');

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: `${theme.spacing.lg * 2}px ${theme.spacing.lg}px ${theme.spacing.lg * 3}px`,
        maxWidth: 860,
        margin: '0 auto',
      }}
    >
      {/* Hero */}
      <Typography.Title level={2} css={{ textAlign: 'center', marginBottom: theme.spacing.xs }}>
        <FormattedMessage defaultMessage="Track users and sessions" description="Title for the sessions empty state" />
      </Typography.Title>
      <Typography.Paragraph
        color="secondary"
        css={{ textAlign: 'center', maxWidth: 560, marginBottom: theme.spacing.lg * 2 }}
      >
        <FormattedMessage
          defaultMessage="Group related traces into sessions to track multi-turn conversations, user journeys, and agent workflows over time."
          description="Subtitle for the sessions empty state"
        />
      </Typography.Paragraph>

      {/* Video preview */}
      <div
        css={{
          width: '100%',
          marginBottom: theme.spacing.lg * 2,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
          border: `1px solid ${theme.colors.border}`,
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.08)',
        }}
      >
        <video src={sessionsVideo} autoPlay loop muted playsInline css={{ width: '100%', display: 'block' }} />
      </div>

      {/* Language selector */}
      <div css={{ width: '100%', marginBottom: theme.spacing.lg }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Select your development language"
            description="Language selector heading for sessions onboarding"
          />
        </Typography.Title>
        <LanguageTab theme={theme} language={language} setLanguage={setLanguage} />
      </div>

      {/* Steps */}
      <div css={{ width: '100%', display: 'flex', flexDirection: 'column', gap: theme.spacing.lg * 1.5 }}>
        {/* Step 1 */}
        <StepSection
          theme={theme}
          stepNumber={1}
          title={
            <FormattedMessage
              defaultMessage="Track users and sessions in your traces"
              description="Step 1 title for sessions onboarding"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Use {sessionId} to group multi-turn conversations and {userId} to attribute traces to specific users. For more details, visit the <a>sessions documentation</a>."
              description="Step 1 description for sessions onboarding"
              values={{
                sessionId: (
                  <code
                    css={{
                      fontSize: 12,
                      backgroundColor: theme.colors.backgroundSecondary,
                      padding: '1px 4px',
                      borderRadius: 3,
                    }}
                  >
                    mlflow.trace.session
                  </code>
                ),
                userId: (
                  <code
                    css={{
                      fontSize: 12,
                      backgroundColor: theme.colors.backgroundSecondary,
                      padding: '1px 4px',
                      borderRadius: 3,
                    }}
                  >
                    mlflow.trace.user
                  </code>
                ),
                a: (text: string) => (
                  <Typography.Link
                    componentId="mlflow.sessions.onboarding.docs_link"
                    href="https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/"
                    openInNewTab
                  >
                    {text}
                  </Typography.Link>
                ),
              }}
            />
          }
        >
          {language === 'python' ? (
            <CodeBlock
              theme={theme}
              code={PYTHON_CODE}
              language="python"
              componentId="mlflow.sessions.onboarding.step1.python.copy"
            />
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <CodeBlock
                theme={theme}
                code={TS_INSTALL_CODE}
                language="bash"
                componentId="mlflow.sessions.onboarding.step1.ts.install.copy"
              />
              <CodeBlock
                theme={theme}
                code={TS_CODE}
                language="typescript"
                componentId="mlflow.sessions.onboarding.step1.ts.code.copy"
              />
            </div>
          )}
        </StepSection>

        {/* Step 2 */}
        <StepSection
          theme={theme}
          stepNumber={2}
          title={
            <FormattedMessage defaultMessage="View your sessions" description="Step 2 title for sessions onboarding" />
          }
          description={
            <FormattedMessage
              defaultMessage="Run your application and sessions will appear here automatically, grouped by session ID."
              description="Step 2 description for sessions onboarding"
            />
          }
          isPending
        />
      </div>
    </div>
  );
};

const LanguageTab = ({
  theme,
  language,
  setLanguage,
}: {
  theme: ThemeType;
  language: Language;
  setLanguage: (lang: Language) => void;
}) => {
  const tabs: { key: Language; label: string }[] = [
    { key: 'python', label: 'Python' },
    { key: 'typescript', label: 'TypeScript' },
  ];

  return (
    <div
      css={{
        display: 'inline-flex',
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        overflow: 'hidden',
      }}
    >
      {tabs.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => setLanguage(key)}
          css={{
            padding: `${theme.spacing.xs + 2}px ${theme.spacing.md}px`,
            border: 'none',
            backgroundColor: language === key ? theme.colors.actionPrimaryBackgroundDefault : 'transparent',
            color: language === key ? '#fff' : theme.colors.textSecondary,
            fontSize: 13,
            fontWeight: 500,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
            ':hover': language !== key ? { backgroundColor: theme.colors.actionIconBackgroundHover } : {},
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
};

const StepSection = ({
  theme,
  stepNumber,
  title,
  description,
  children,
  isPending,
}: {
  theme: ThemeType;
  stepNumber: number;
  title: React.ReactNode;
  description: React.ReactNode;
  children?: React.ReactNode;
  isPending?: boolean;
}) => {
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 2 }}>
        <div
          css={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: isPending ? 'transparent' : theme.colors.actionPrimaryBackgroundDefault,
            border: isPending ? `2px dashed ${theme.colors.border}` : 'none',
            color: isPending ? theme.colors.textSecondary : '#fff',
            fontSize: 13,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {stepNumber}
        </div>
        {!isPending && (
          <div css={{ width: 2, flex: 1, backgroundColor: theme.colors.border, marginTop: theme.spacing.xs }} />
        )}
      </div>
      <div css={{ flex: 1, minWidth: 0, paddingBottom: isPending ? 0 : theme.spacing.sm }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
          <Typography.Title level={4} withoutMargins>
            {title}
          </Typography.Title>
          {isPending && (
            <span
              css={{
                fontSize: 11,
                fontWeight: 600,
                color: theme.colors.textValidationWarning,
                backgroundColor: `${theme.colors.textValidationWarning}15`,
                padding: '2px 8px',
                borderRadius: 10,
              }}
            >
              <FormattedMessage
                defaultMessage="Pending"
                description="Label indicating this onboarding step is not yet completed"
              />
            </span>
          )}
        </div>
        <Typography.Paragraph color="secondary" css={{ marginBottom: children ? theme.spacing.sm : 0 }}>
          {description}
        </Typography.Paragraph>
        {children}
      </div>
    </div>
  );
};

const CodeBlock = ({
  theme,
  code,
  language,
  componentId,
}: {
  theme: ThemeType;
  code: string;
  language: string;
  componentId: string;
}) => {
  return (
    <div
      css={{
        position: 'relative',
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      <CopyButton
        componentId={componentId}
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        showLineNumbers
        language={language === 'bash' ? 'text' : language === 'typescript' ? 'javascript' : 'python'}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        style={{ fontSize: 12, overflow: 'auto' }}
      >
        {code}
      </CodeSnippet>
    </div>
  );
};
