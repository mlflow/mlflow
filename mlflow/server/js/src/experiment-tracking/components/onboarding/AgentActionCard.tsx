import { useState, type ReactNode } from 'react';
import { Button, CopyIcon, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AssistantSparkleIcon, useAssistant } from '../../../assistant';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

const AGENT_SETUP_COMMAND = 'mlflow agent setup';

type TabKey = 'agent-setup' | 'assistant' | 'coding-agent';

export const AgentActionCard = ({
  title,
  codingAgentPrompt,
  assistantPrompt,
  componentId,
  showAgentSetupTab = false,
}: {
  title: ReactNode;
  codingAgentPrompt: string;
  assistantPrompt: string;
  componentId: string;
  showAgentSetupTab?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, queueMessage } = useAssistant();
  const defaultTab: TabKey = showAgentSetupTab ? 'agent-setup' : 'coding-agent';
  const [activeTab, setActiveTab] = useState<TabKey>(defaultTab);

  const handleAssistantClick = () => {
    openPanel();
    queueMessage(assistantPrompt);
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
      <Typography.Title level={4} css={{ margin: 0 }}>
        {title}
      </Typography.Title>

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
              backgroundColor: theme.isDarkMode ? '#1f2937' : '#eef2f6',
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
                color: theme.isDarkMode ? '#d4d8de' : '#2b3441',
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
