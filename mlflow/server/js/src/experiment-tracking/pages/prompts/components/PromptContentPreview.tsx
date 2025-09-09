import {
  Button,
  CopyIcon,
  Modal,
  PlayIcon,
  Spacer,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { getPromptContentTagValue } from '../utils';
import { PromptVersionMetadata } from './PromptVersionMetadata';
import { FormattedMessage } from 'react-intl';
import { uniq } from 'lodash';
import { useDeletePromptVersionModal } from '../hooks/useDeletePromptVersionModal';
import { ShowArtifactCodeSnippet } from '../../../components/artifact-view-components/ShowArtifactCodeSnippet';

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

  const { DeletePromptModal, openModal: openDeleteModal } = useDeletePromptVersionModal({
    promptVersion,
    onSuccess: () => onDeletedVersion?.(),
  });

  const [showUsageExample, setShowUsageExample] = useState(false);

  // Find all variables in the prompt content
  const variableNames = useMemo(() => {
    if (!value) {
      return [];
    }

    const variables: string[] = [];
    let match;

    while ((match = PROMPT_VARIABLE_REGEX.exec(value)) !== null) {
      variables.push(match[1]);
    }

    // Sanity check for tricky cases like nested brackets. If the variable name contains
    // a bracket, we consider it as a parsing error and render a placeholder instead.
    if (variables.some((variable) => variable.includes('{') || variable.includes('}'))) {
      return null;
    }

    return uniq(variables);
  }, [value]);
  const codeSnippetContent = buildCodeSnippetContent(promptVersion, variableNames);

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
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
          overflow: 'auto',
        }}
      >
        <Typography.Text
          css={{
            whiteSpace: 'pre-wrap',
          }}
        >
          {value || 'Empty'}
        </Typography.Text>
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
        <ShowArtifactCodeSnippet code={buildCodeSnippetContent(promptVersion, variableNames)} />
      </Modal>
      {DeletePromptModal}
    </div>
  );
};

const buildCodeSnippetContent = (promptVersion: RegisteredPromptVersion | undefined, variables: string[] | null) => {
  let codeSnippetContent = `from openai import OpenAI
import mlflow
client = OpenAI(api_key="<YOUR_API_KEY>")

# Set MLflow tracking URI
mlflow.set_tracking_uri("<YOUR_TRACKING_URI>")

# Example of loading and using the prompt
prompt = mlflow.genai.load_prompt("prompts:/${promptVersion?.name}/${promptVersion?.version}")`;

  // Null variables mean that there was a parsing error
  if (variables === null) {
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
