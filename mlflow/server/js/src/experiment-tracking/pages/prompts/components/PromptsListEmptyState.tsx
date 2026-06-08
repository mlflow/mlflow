import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { AgentActionCard } from '../../../components/onboarding/AgentActionCard';

const buildCreatePromptPrompt = (
  trackingUri: string,
): string => `Help me create my first MLflow registered prompt and walk me through testing it.

MLflow is running at ${trackingUri}.

Guide me through this end-to-end — one question at a time, don't dump everything at once:
1. Ask what the prompt is for in plain language (e.g. "summarize an article", "extract action items from a meeting transcript").
2. Ask if the prompt is global or attached to an experiment.
3. Suggest a kebab-case name based on my answer and confirm it with me.
4. Draft a prompt template using \`{{variable}}\` placeholders for the inputs. Show it to me and let me iterate before saving.
5. Once I approve, register it via MLflow's Prompt Registry API. Report the prompt name, version, and a URL I can open in the MLflow UI.
6. Help me test it — either:
   - If I have an existing app in this repo, find the right place to swap in \`mlflow.genai.load_prompt(name, version)\` and offer to make the edit, OR
   - If I don't have an app yet, give me a copy-runnable Python snippet that loads the prompt and runs it against an LLM (default to OpenAI; check that I have the API key first).

Inference rules:
- Read the repo first. Infer the package manager, entry point, and LLM framework from imports.
- Only ask the user when there's genuine ambiguity.
- Make changes, then report what you did and the URLs.

If you get stuck — missing API key, permissions blocked, missing dependencies, or anything else you can't resolve — respond in this format and stop:

**I need:** <plain-language one-liner. Assume the user may not be a developer. Say "your OpenAI API key" not "OPENAI_API_KEY env var".>

**How to fix:** <one specific step. If a terminal command is required, write "Open a terminal in your project folder and run:" then put the command on its own line in a code block. If it's getting an API key, link to where to get one (e.g. https://platform.openai.com/api-keys).>

Use plain language. One line each. Don't paste code snippets unless they need to be run.`;

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
        showTerminalCommand={false}
        title={
          <FormattedMessage
            defaultMessage="Let MLflow's AI assistant create your first prompt"
            description="Headline for the agent CTA card on the prompts list empty state"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="The assistant will ask about your use case, register the prompt, and help you test it end-to-end."
            description="Subline for the agent CTA card on the prompts list empty state"
          />
        }
        buttonLabel={
          <FormattedMessage
            defaultMessage="Create with AI"
            description="Button label for the agent CTA card on the prompts list empty state"
          />
        }
        prompt={buildCreatePromptPrompt(window.location.origin)}
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
