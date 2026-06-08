import { useState } from 'react';
import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AgentActionCard } from './onboarding/AgentActionCard';
import { buildCreateExperimentPrompt } from './experimentListAgentPrompt';

const TRACING_VIDEO_START_SEC = 0;
const TRACING_VIDEO_URL = `https://mlflow.org/docs/latest/images/llms/tracing/tracing-top.mp4#t=${TRACING_VIDEO_START_SEC}`;

export const ExperimentListEmptyState = ({ onCreateExperiment }: { onCreateExperiment?: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const [videoFailed, setVideoFailed] = useState(false);

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: `${theme.spacing.lg}px ${theme.spacing.lg}px ${theme.spacing.lg * 3}px`,
        maxWidth: 860,
        margin: '0 auto',
      }}
    >
      <Typography.Title level={2} css={{ textAlign: 'center', marginBottom: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Start tracking your ML workflows with Experiments"
          description="Title for the empty state on the experiments list page"
        />
      </Typography.Title>
      <Typography.Paragraph
        color="secondary"
        css={{ textAlign: 'center', maxWidth: 520, marginBottom: theme.spacing.lg * 2 }}
      >
        <FormattedMessage
          defaultMessage="Experiments organize your runs, traces, and evaluation results. Let MLflow's AI assistant set everything up, or create an experiment manually."
          description="Subtitle for the empty state on the experiments list page"
        />
      </Typography.Paragraph>

      {!videoFailed && (
        <div
          css={{
            width: '100%',
            maxWidth: 520,
            marginBottom: theme.spacing.lg * 2,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'hidden',
            border: `1px solid ${theme.colors.border}`,
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.08)',
          }}
        >
          <video
            src={TRACING_VIDEO_URL}
            autoPlay
            muted
            playsInline
            onError={() => setVideoFailed(true)}
            onEnded={(event) => {
              const video = event.currentTarget;
              video.currentTime = TRACING_VIDEO_START_SEC;
              video.play().catch(() => {
                // Autoplay restrictions can reject; ignore since the video is muted.
              });
            }}
            css={{ width: '100%', display: 'block' }}
          />
        </div>
      )}

      <AgentActionCard
        componentId="mlflow.experiments.onboarding.create_with_agent"
        title={
          <FormattedMessage
            defaultMessage="Let MLflow's AI assistant create your first experiment"
            description="Headline for the agent CTA card on the experiments list empty state"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Chat with the assistant to name and create an experiment."
            description="Subline for the agent CTA card on the experiments list empty state"
          />
        }
        buttonLabel={
          <FormattedMessage
            defaultMessage="Create with AI"
            description="Button label for the agent CTA card on the experiments list empty state"
          />
        }
        prompt={buildCreateExperimentPrompt(window.location.origin)}
      />

      {onCreateExperiment && (
        <Button
          componentId="mlflow.experiment_list_table.create_experiment"
          data-testid="create-experiment-table-empty-state-button"
          onClick={onCreateExperiment}
          type="primary"
          icon={<PlusIcon />}
        >
          <FormattedMessage
            defaultMessage="Or create an experiment manually"
            description="CTA on the experiments empty state to create an experiment as an alternative to the agent path"
          />
        </Button>
      )}
    </div>
  );
};
