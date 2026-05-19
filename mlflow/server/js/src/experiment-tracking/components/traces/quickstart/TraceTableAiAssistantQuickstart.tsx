import { Typography, type ThemeType } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AI_ASSISTANT_SKILLS_INSTALL_CODE, getAiAssistantPrompt } from './TraceTableQuickstart.utils';
import { CodeBlock } from './components/CodeBlock';
import { StepSection } from './components/StepSection';

const SUPPORTED_AGENT_NAMES = ['Claude Code', 'Cursor', 'Codex CLI', 'Gemini CLI', 'OpenCode'];

export const TraceTableAiAssistantQuickstart = ({
  theme,
  trackingUri,
  experimentName,
  workspace,
}: {
  theme: ThemeType;
  trackingUri: string;
  experimentName: string;
  workspace?: string | null;
}) => {
  const prompt = getAiAssistantPrompt({ trackingUri, experimentName, workspace });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg * 1.5 }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="Works with"
            description="Label introducing coding agents supported by the MLflow skills quickstart"
          />
        </Typography.Text>
        {SUPPORTED_AGENT_NAMES.map((agentName) => (
          <span
            key={agentName}
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              backgroundColor: theme.colors.backgroundPrimary,
              fontWeight: 600,
            }}
          >
            {agentName}
          </span>
        ))}
      </div>
      <StepSection
        theme={theme}
        stepNumber={1}
        title={
          <FormattedMessage
            defaultMessage="Install the MLflow skills"
            description="Step 1 title for AI assistant traces onboarding"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Add the MLflow skills once, then your coding assistant can use them to instrument tracing workflows."
            description="Step 1 description for AI assistant traces onboarding"
          />
        }
      >
        <CodeBlock
          theme={theme}
          code={AI_ASSISTANT_SKILLS_INSTALL_CODE}
          language="bash"
          componentId="mlflow.traces.onboarding.ai_assistant.install.copy"
        />
      </StepSection>
      <StepSection
        theme={theme}
        stepNumber={2}
        title={
          <FormattedMessage
            defaultMessage="Ask your agent to add tracing"
            description="Step 2 title for AI assistant traces onboarding"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Paste this prompt into your coding agent. It includes this MLflow server and experiment so the generated traces land here."
            description="Step 2 description for AI assistant traces onboarding"
          />
        }
      >
        <CodeBlock
          theme={theme}
          code={prompt}
          language="text"
          componentId="mlflow.traces.onboarding.ai_assistant.prompt.copy"
        />
      </StepSection>
      <StepSection
        theme={theme}
        stepNumber={3}
        title={
          <FormattedMessage
            defaultMessage="View your traces"
            description="Step 3 title for AI assistant traces onboarding"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="After the agent finishes and runs the instrumented workflow, traces will appear here automatically."
            description="Step 3 description for AI assistant traces onboarding"
          />
        }
        isPending
      />
    </div>
  );
};
