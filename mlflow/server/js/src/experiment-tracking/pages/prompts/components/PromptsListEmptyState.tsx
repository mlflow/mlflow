import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AgentActionCard } from '../../../components/onboarding/AgentActionCard';
import { buildCreatePromptAssistantPrompt, buildCreatePromptPrompt } from './promptsAgentPrompt';

export const PromptsListEmptyState = ({ onCreatePrompt }: { onCreatePrompt: () => void }) => {
  const { theme } = useDesignSystemTheme();

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
          defaultMessage="Create your first prompt"
          description="Title for the empty state on the prompts list page"
        />
      </Typography.Title>
      <Typography.Paragraph
        color="secondary"
        css={{ textAlign: 'center', maxWidth: 520, marginBottom: theme.spacing.lg * 2 }}
      >
        <FormattedMessage
          defaultMessage="Manage and version prompts using MLflow's Prompt Registry. <link>Learn more</link>"
          description="Subtitle for the empty state on the prompts list page"
          values={{
            link: (content: any) => (
              <Typography.Link
                componentId="mlflow.prompts.list.empty_state.learn_more_link"
                href="https://mlflow.org/docs/latest/genai/prompt-registry/"
                openInNewTab
              >
                {content}
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>

      <AgentActionCard
        componentId="mlflow.prompts.onboarding.create_with_agent"
        title={
          <FormattedMessage
            defaultMessage="Get help registering a prompt"
            description="Headline for the agent CTA card on the prompts list empty state"
          />
        }
        codingAgentPrompt={buildCreatePromptPrompt(window.location.origin)}
        assistantPrompt={buildCreatePromptAssistantPrompt(window.location.origin)}
      />

      <Button
        componentId="mlflow.prompts.list.table.create_prompt"
        data-testid="create-prompt-empty-state-button"
        onClick={onCreatePrompt}
        type="primary"
        icon={<PlusIcon />}
      >
        <FormattedMessage
          defaultMessage="Or create a prompt manually"
          description="CTA on the prompts empty state to create a prompt as an alternative to the agent path"
        />
      </Button>
    </div>
  );
};
