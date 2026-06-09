import { useState } from 'react';
import { Button, CopyIcon, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AssistantSparkleIcon, useAssistant } from '@mlflow/mlflow/src/assistant';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@mlflow/mlflow/src/shared/web-shared/snippet';

import { buildEvaluateAssistantPrompt, buildEvaluatePrompt } from './evalRunsAgentPrompt';
import { getTraceCodeSnippet } from './EvaluationCodeSnippetButton';
import { RunEvaluationButton } from './RunEvaluationButton';

type TabKey = 'coding-agent' | 'python' | 'assistant';

const BACKGROUND_LIGHT = '#eef2f6';
const BACKGROUND_DARK = '#1f2937';
const TEXT_LIGHT = '#2b3441';
const TEXT_DARK = '#d4d8de';

const CopyableBlock = ({ content, componentId }: { content: string; componentId: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        position: 'relative',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: theme.isDarkMode ? BACKGROUND_DARK : BACKGROUND_LIGHT,
      }}
    >
      <CopyButton
        componentId={componentId}
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={content}
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
          color: theme.isDarkMode ? TEXT_DARK : TEXT_LIGHT,
          maxHeight: 480,
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {content}
      </pre>
    </div>
  );
};

const PythonCodeBlock = ({ content, componentId }: { content: string; componentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const backgroundColor = theme.isDarkMode ? BACKGROUND_DARK : BACKGROUND_LIGHT;
  return (
    <div
      css={{
        position: 'relative',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor,
        maxHeight: 480,
        overflow: 'auto',
      }}
    >
      <CopyButton
        componentId={componentId}
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={content}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        language="python"
        style={{
          backgroundColor,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          paddingRight: theme.spacing.lg * 1.5,
          fontSize: 12,
          lineHeight: 1.55,
        }}
      >
        {content}
      </CodeSnippet>
    </div>
  );
};

export const EvalRunsEmptyStateCard = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const { openPanel, reset, sendMessage, setupComplete } = useAssistant();
  const [activeTab, setActiveTab] = useState<TabKey>('coding-agent');
  const codingAgentPrompt = buildEvaluatePrompt(window.location.origin, experimentId);
  const assistantPrompt = buildEvaluateAssistantPrompt(window.location.origin, experimentId);

  const handleAssistantClick = () => {
    openPanel();
    if (setupComplete) {
      reset();
      sendMessage(assistantPrompt);
    }
  };

  return (
    <div
      css={{
        width: '100%',
        maxWidth: 720,
        margin: '0 auto',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <RunEvaluationButton
          experimentId={experimentId}
          type="primary"
          label={
            <FormattedMessage
              defaultMessage="Evaluate traces"
              description="Primary CTA in the eval-runs empty state — opens the trace selection + scorer picker modal"
            />
          }
        />
      </div>

      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          color: theme.colors.textSecondary,
          fontSize: 12,
        }}
      >
        <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.border }} />
        <FormattedMessage
          defaultMessage="Or use code"
          description="Divider label separating the primary UI evaluation flow from the code-based alternatives"
        />
        <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.border }} />
      </div>

      <Tabs.Root
        componentId="mlflow.eval-runs.empty-state.unified.tabs"
        valueHasNoPii
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as TabKey)}
      >
        <Tabs.List
          shadowScrollStylesBackgroundColor={theme.colors.backgroundSecondary}
          css={{ width: '100%', borderBottom: `1px solid ${theme.colors.border}` }}
        >
          <Tabs.Trigger value="coding-agent">
            <FormattedMessage
              defaultMessage="Your coding agent"
              description="Tab label for the copy-prompt-to-local-agent path in the eval-runs empty state"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="python">
            <FormattedMessage
              defaultMessage="Python"
              description="Tab label for the runnable Python code-snippet path in the eval-runs empty state"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="assistant">
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <AssistantSparkleIcon isHovered={false} iconSize={14} />
              <FormattedMessage
                defaultMessage="MLflow assistant"
                description="Tab label for the in-UI MLflow assistant path in the eval-runs empty state"
              />
            </span>
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="coding-agent" css={{ paddingTop: 0 }}>
          <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Paste this prompt into Claude Code, Codex, Cursor, or any coding agent already running in your project."
              description="Description above the agent prompt in the eval-runs empty state"
            />
          </Typography.Text>
          <CopyableBlock content={codingAgentPrompt} componentId="mlflow.eval-runs.empty-state.unified.copy_prompt" />
        </Tabs.Content>

        <Tabs.Content value="python" css={{ paddingTop: 0 }}>
          <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Copy this snippet into your project, set your API key, and run it."
              description="Description above the Python evaluation snippet in the eval-runs empty state"
            />
          </Typography.Text>
          <PythonCodeBlock
            content={getTraceCodeSnippet(experimentId)}
            componentId="mlflow.eval-runs.empty-state.unified.copy_snippet"
          />
        </Tabs.Content>

        <Tabs.Content value="assistant" css={{ paddingTop: 0 }}>
          <Typography.Text color="secondary" css={{ fontSize: 13, display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Opens the MLflow assistant in this browser and starts the conversation."
              description="Description above the open-assistant button in the eval-runs empty state"
            />
          </Typography.Text>
          <Button
            componentId="mlflow.eval-runs.empty-state.unified.ask_assistant"
            type="primary"
            onClick={handleAssistantClick}
          >
            <FormattedMessage
              defaultMessage="Open assistant"
              description="Button label for opening the MLflow assistant from the eval-runs empty state"
            />
          </Button>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
