import { useState, type ReactNode } from 'react';
import { Button, CopyIcon, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AssistantSparkleIcon, useAssistant } from '../../../assistant';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet, type CodeSnippetLanguage } from '@mlflow/mlflow/src/shared/web-shared/snippet';

const AGENT_SETUP_COMMAND = 'mlflow agent setup';

type TabKey = 'agent-setup' | 'coding-agent' | 'code-snippet' | 'assistant';

export type AgentActionCardCodeSnippet = {
  content: string;
  language: CodeSnippetLanguage;
  label: ReactNode;
};

export const AgentActionCard = ({
  title,
  header,
  codingAgentPrompt,
  assistantPrompt,
  componentId,
  showAgentSetupTab = false,
  codeSnippet,
}: {
  /** Card title — rendered as Typography.Title level 4 above the tabs. Ignored if `header` is set. */
  title?: ReactNode;
  /** Replaces the title slot with custom content (e.g. a primary CTA + divider). */
  header?: ReactNode;
  codingAgentPrompt: string;
  assistantPrompt: string;
  componentId: string;
  showAgentSetupTab?: boolean;
  /** When provided, renders an additional tab with a syntax-highlighted code block between
   *  coding-agent and assistant. */
  codeSnippet?: AgentActionCardCodeSnippet;
}) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, sendMessage, setupComplete } = useAssistant();
  const defaultTab: TabKey = showAgentSetupTab ? 'agent-setup' : 'coding-agent';
  const [activeTab, setActiveTab] = useState<TabKey>(defaultTab);

  const handleAssistantClick = () => {
    openPanel();
    // TODO(ML-66316 follow-up): if setup isn't complete, the prompt is silently dropped.
    // Once the setup wizard finishes the user has to retype it. Plumb a queueMessage()
    // through AssistantContext to defer the send until setupComplete flips true, and
    // fix the stale-sessionId closure inside startChat at the same time.
    if (setupComplete) {
      sendMessage(assistantPrompt);
    }
  };

  return (
    <div
      css={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        marginBottom: theme.spacing.lg * 1.5,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      {header ??
        (title ? (
          <Typography.Title level={4} css={{ margin: 0 }}>
            {title}
          </Typography.Title>
        ) : null)}

      <Tabs.Root
        componentId={`${componentId}.tabs`}
        valueHasNoPii
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as TabKey)}
      >
        <Tabs.List
          // Tabs.List paints scroll-shadow gradients on an inner viewport using
          // theme.colors.backgroundPrimary (white) by default — the slivers show even with no
          // overflow. Passing the card's background color blends them out.
          shadowScrollStylesBackgroundColor={theme.colors.backgroundSecondary}
          css={{
            width: '100%',
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          {showAgentSetupTab && (
            <Tabs.Trigger value="agent-setup">
              <FormattedMessage
                defaultMessage="CLI"
                description="Tab label for the mlflow agent setup CLI path in the agent action card"
              />
            </Tabs.Trigger>
          )}
          <Tabs.Trigger value="coding-agent">
            <FormattedMessage
              defaultMessage="Your coding agent"
              description="Tab label for the copy-prompt-to-local-agent path in the agent action card"
            />
          </Tabs.Trigger>
          {codeSnippet && <Tabs.Trigger value="code-snippet">{codeSnippet.label}</Tabs.Trigger>}
          <Tabs.Trigger value="assistant">
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <AssistantSparkleIcon isHovered={false} iconSize={14} />
              <FormattedMessage
                defaultMessage="MLflow assistant"
                description="Tab label for the in-UI MLflow assistant path in the agent action card"
              />
            </span>
          </Tabs.Trigger>
        </Tabs.List>

        {showAgentSetupTab && (
          <Tabs.Content value="agent-setup" css={{ paddingTop: 0 }}>
            <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="Run this in your terminal — it installs MLflow skills into your project and launches your coding agent (Claude Code, Codex, or opencode) with instructions to instrument your app."
                description="Description above the mlflow agent setup terminal command in the agent action card"
              />
            </Typography.Text>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <code
                css={{
                  flex: 1,
                  fontSize: 13,
                  padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                  borderRadius: theme.borders.borderRadiusSm,
                  backgroundColor: theme.colors.backgroundPrimary,
                  border: `1px solid ${theme.colors.border}`,
                  color: theme.colors.textPrimary,
                }}
              >
                {AGENT_SETUP_COMMAND}
              </code>
              <CopyButton
                componentId={`${componentId}.copy_agent_setup_command`}
                copyText={AGENT_SETUP_COMMAND}
                showLabel={false}
                icon={<CopyIcon />}
              />
            </div>
          </Tabs.Content>
        )}

        <Tabs.Content value="coding-agent" css={{ paddingTop: 0 }}>
          <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Copy this prompt and paste into Claude Code, Codex, Cursor, or any coding agent already running in your project."
              description="Description above the copy-prompt block in the agent action card"
            />
          </Typography.Text>
          <div
            css={{
              position: 'relative',
              borderRadius: theme.borders.borderRadiusMd,
              overflow: 'hidden',
              border: `1px solid ${theme.colors.border}`,
              backgroundColor: theme.colors.backgroundPrimary,
            }}
          >
            <CopyButton
              componentId={`${componentId}.copy_prompt`}
              css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
              showLabel={false}
              copyText={codingAgentPrompt}
              icon={<CopyIcon />}
            />
            <pre
              css={{
                margin: 0,
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                paddingRight: theme.spacing.lg * 1.5,
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
                fontSize: 12,
                lineHeight: 1.55,
                color: theme.colors.textPrimary,
                maxHeight: 160,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {codingAgentPrompt}
            </pre>
          </div>
        </Tabs.Content>

        {codeSnippet && (
          <Tabs.Content value="code-snippet" css={{ paddingTop: 0 }}>
            <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="Copy this snippet into your project, set your API key, and run it."
                description="Description above the runnable code snippet in the agent action card"
              />
            </Typography.Text>
            <div
              css={{
                position: 'relative',
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundPrimary,
                maxHeight: 160,
                overflow: 'auto',
              }}
            >
              <CopyButton
                componentId={`${componentId}.copy_snippet`}
                css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
                showLabel={false}
                copyText={codeSnippet.content}
                icon={<CopyIcon />}
              />
              <CodeSnippet
                theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
                language={codeSnippet.language}
                style={{
                  backgroundColor: theme.colors.backgroundPrimary,
                  padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                  paddingRight: theme.spacing.lg * 1.5,
                  fontSize: 12,
                  lineHeight: 1.55,
                }}
              >
                {codeSnippet.content}
              </CodeSnippet>
            </div>
          </Tabs.Content>
        )}

        <Tabs.Content value="assistant" css={{ paddingTop: 0 }}>
          <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Opens the MLflow assistant in this browser and starts the conversation."
              description="Description above the MLflow assistant button in the agent action card"
            />
          </Typography.Text>
          <Button componentId={`${componentId}.open_assistant`} type="primary" onClick={handleAssistantClick}>
            <FormattedMessage
              defaultMessage="Open assistant"
              description="Canonical button label for opening the MLflow assistant from an agent action card"
            />
          </Button>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
