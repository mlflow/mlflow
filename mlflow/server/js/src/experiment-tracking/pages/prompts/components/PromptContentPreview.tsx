import {
  Button,
  LightningIcon,
  Modal,
  PlayIcon,
  Spacer,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import {
  getChatPromptMessagesFromValue,
  getPromptContentTagValue,
  isChatPrompt,
  PROMPT_TYPE_CHAT,
  PROMPT_TYPE_TEXT,
} from '../utils';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { PromptVersionMetadata } from './PromptVersionMetadata';
import { FormattedMessage } from 'react-intl';
import { uniq } from 'lodash';
import { useDeletePromptVersionModal } from '../hooks/useDeletePromptVersionModal';
import { ShowArtifactCodeSnippet } from '../../../components/artifact-view-components/ShowArtifactCodeSnippet';
import { ModelTraceExplorerChatMessage } from '@mlflow/mlflow/src/shared/web-shared/model-trace-explorer/right-pane/ModelTraceExplorerChatMessage';
import type { ModelTraceChatMessage } from '@mlflow/mlflow/src/shared/web-shared/model-trace-explorer/ModelTrace.types';
import { OptimizeModal } from './OptimizeModal';

const PROMPT_VARIABLE_REGEX = /\{\{\s*(.*?)\s*\}\}/g;

export const PromptContentPreview = ({
  promptVersion,
  onUpdatedContent,
  onDeletedVersion,
  aliasesByVersion,
  registeredPrompt,
  showEditAliasesModal,
  showEditPromptVersionMetadataModal,
}: {
  promptVersion?: RegisteredPromptVersion;
  onUpdatedContent?: () => Promise<any>;
  onDeletedVersion?: () => Promise<any>;
  aliasesByVersion: Record<string, string[]>;
  registeredPrompt?: RegisteredPrompt;
  showEditAliasesModal?: (versionNumber: string) => void;
  showEditPromptVersionMetadataModal: (promptVersion: RegisteredPromptVersion) => void;
}) => {
  const value = useMemo(() => (promptVersion ? getPromptContentTagValue(promptVersion) : ''), [promptVersion]);
  const isChatPromptType = useMemo(() => isChatPrompt(promptVersion), [promptVersion]);
  const parsedMessages = useMemo(
    () => (isChatPromptType ? getChatPromptMessagesFromValue(value) : undefined),
    [isChatPromptType, value],
  );

  const { DeletePromptModal, openModal: openDeleteModal } = useDeletePromptVersionModal({
    promptVersion,
    onSuccess: () => onDeletedVersion?.(),
  });

  const [showUsageExample, setShowUsageExample] = useState(false);
  const [showOptimizeModal, setShowOptimizeModal] = useState(false);

  // Find all variables in the prompt content
  const variableNames = useMemo(() => {
    if (!value) {
      return [];
    }

    const variables: string[] = [];
    const source = isChatPromptType ? parsedMessages?.map((m) => m.content).join('\n') || '' : value;

    let match;
    while ((match = PROMPT_VARIABLE_REGEX.exec(source)) !== null) {
      variables.push(match[1]);
    }

    // Sanity check for tricky cases like nested brackets. If the variable name contains
    // a bracket, we consider it as a parsing error and render a placeholder instead.
    if (variables.some((variable) => variable.includes('{') || variable.includes('}'))) {
      return null;
    }

    return uniq(variables);
  }, [value, isChatPromptType, parsedMessages]);
  const codeSnippetContent = buildCodeSnippetContent(
    promptVersion,
    variableNames,
    isChatPromptType ? PROMPT_TYPE_CHAT : undefined,
  );

  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        paddingTop: 0,
        borderRadius: theme.borders.borderRadiusSm,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography.Title level={3}>Viewing version {promptVersion?.version}</Typography.Title>
        <div css={{ display: 'flex', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.prompts.details.delete_version"
            icon={<TrashIcon />}
            type="primary"
            danger
            onClick={openDeleteModal}
          >
            <FormattedMessage
              defaultMessage="Delete version"
              description="A label for a button to delete prompt version on the prompt details page"
            />
          </Button>
          <Button
            componentId="mlflow.prompts.details.preview.optimize"
            icon={<LightningIcon />}
            onClick={() => setShowOptimizeModal(true)}
          >
            <FormattedMessage
              defaultMessage="Optimize"
              description="A label for a button to display the modal with instructions to optimize the prompt"
            />
          </Button>
          <Button
            componentId="mlflow.prompts.details.preview.use"
            icon={<PlayIcon />}
            onClick={() => setShowUsageExample(true)}
          >
            <FormattedMessage
              defaultMessage="Use"
              description="A label for a button to display the modal with the usage example of the prompt"
            />
          </Button>
        </div>
      </div>
      <Spacer shrinks={false} />
      <PromptVersionMetadata
        aliasesByVersion={aliasesByVersion}
        registeredPrompt={registeredPrompt}
        registeredPromptVersion={promptVersion}
        showEditAliasesModal={showEditAliasesModal}
        showEditPromptVersionMetadataModal={showEditPromptVersionMetadataModal}
      />
      <Spacer shrinks={false} />
      <div
        css={{
          backgroundColor: isChatPromptType ? undefined : theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        {isChatPromptType && parsedMessages ? (
          parsedMessages.map((msg: any, index: number) => (
            <ModelTraceExplorerChatMessage
              key={index}
              message={
                {
                  ...msg,
                  content: msg.content,
                } as ModelTraceChatMessage
              }
            />
          ))
        ) : (
          <Typography.Text
            css={{
              whiteSpace: 'pre-wrap',
            }}
          >
            {value || 'Empty'}
          </Typography.Text>
        )}
      </div>
      <Modal
        componentId="mlflow.prompts.details.preview.usage_example_modal"
        title={
          <FormattedMessage
            defaultMessage="Usage example"
            description="A title of the modal showing the usage example of the prompt"
          />
        }
        visible={showUsageExample}
        onCancel={() => setShowUsageExample(false)}
        cancelText={
          <FormattedMessage
            defaultMessage="Dismiss"
            description="A label for the button to dismiss the modal with the usage example of the prompt"
          />
        }
      >
        <ShowArtifactCodeSnippet
          code={buildCodeSnippetContent(promptVersion, variableNames, isChatPromptType ? PROMPT_TYPE_CHAT : undefined)}
        />{' '}
      </Modal>
      <OptimizeModal
        visible={showOptimizeModal}
        promptName={promptVersion?.name || ''}
        promptVersion={promptVersion?.version || ''}
        onCancel={() => setShowOptimizeModal(false)}
      />
      {DeletePromptModal}
    </div>
  );
};

const buildCodeSnippetContent = (
  promptVersion: RegisteredPromptVersion | undefined,
  variables: string[] | null,
  promptType: string = PROMPT_TYPE_TEXT,
) => {
  let codeSnippetContent = `from openai import OpenAI
import mlflow
client = OpenAI(api_key="<YOUR_API_KEY>")

# Set MLflow tracking URI
mlflow.set_tracking_uri("<YOUR_TRACKING_URI>")

# Example of loading and using the prompt
prompt = mlflow.genai.load_prompt("prompts:/${promptVersion?.name}/${promptVersion?.version}")`;

  // Null variables mean that there was a parsing error
  if (variables === null) {
    if (promptType === PROMPT_TYPE_CHAT) {
      codeSnippetContent += `

# Replace the variables with the actual values
variables = {
   "key": "value",
   ...
}

messages = prompt.format(**variables)
response = client.chat.completions.create(
    messages=messages,
    model="gpt-4o-mini",
)`;
    } else {
      codeSnippetContent += `

# Replace the variables with the actual values
variables = {
   "key": "value",
   ...
}

response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": prompt.format(**variables),
    }],
    model="gpt-4o-mini",
)`;
    }
  } else if (promptType === PROMPT_TYPE_CHAT) {
    codeSnippetContent += `
messages = prompt.format_messages(${variables.map((name) => `${name}="<${name}>"`).join(', ')})
response = client.chat.completions.create(
    messages=messages,
    model="gpt-4o-mini",
)`;
  } else {
    codeSnippetContent += `
response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": prompt.format(${variables.map((name) => `${name}="<${name}>"`).join(', ')}),
    }],
    model="gpt-4o-mini",
)`;
  }

  codeSnippetContent += `\n\nprint(response.choices[0].message.content)`;
  return codeSnippetContent;
};
