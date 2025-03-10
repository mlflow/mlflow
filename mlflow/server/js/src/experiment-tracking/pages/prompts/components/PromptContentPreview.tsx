import { Button, Modal, PlayIcon, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { getPromptContentTagValue } from '../utils';
import { PromptVersionMetadata } from './PromptVersionMetadata';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { uniq } from 'lodash';

export const PromptContentPreview = ({
  promptVersion,
  onUpdatedContent,
  aliasesByVersion,
  registeredPrompt,
  showEditAliasesModal,
}: {
  promptVersion?: RegisteredPromptVersion;
  onUpdatedContent?: () => Promise<any>;
  aliasesByVersion: Record<string, string[]>;
  registeredPrompt?: RegisteredPrompt;
  showEditAliasesModal?: (versionNumber: string) => void;
}) => {
  const value = useMemo(() => (promptVersion ? getPromptContentTagValue(promptVersion) : ''), [promptVersion]);

  const [showUsageExample, setShowUsageExample] = useState(false);

  const variableNames = useMemo(() => {
    if (!value) {
      return [];
    }
    const regex = /{([^}]+)}/g;

    const variables: string[] = [];
    let match;

    while ((match = regex.exec(value)) !== null) {
      variables.push(match[1]);
    }

    return uniq(variables);
  }, [value]);

  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        paddingTop: 0,
        borderRadius: '4px',
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography.Title level={3}>Viewing version {promptVersion?.version}</Typography.Title>
        <Button componentId="TODO" icon={<PlayIcon />} onClick={() => setShowUsageExample(true)}>
          Use
        </Button>
      </div>
      <Spacer shrinks={false} />
      <PromptVersionMetadata
        aliasesByVersion={aliasesByVersion}
        registeredPrompt={registeredPrompt}
        registeredPromptVersion={promptVersion}
        showEditAliasesModal={showEditAliasesModal}
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
        componentId="TODO"
        title={<FormattedMessage defaultMessage="Usage example" description="TODO" />}
        visible={showUsageExample}
        onCancel={() => setShowUsageExample(false)}
        cancelText={<FormattedMessage defaultMessage="Dismiss" description="TODO" />}
      >
        <CodeSnippet language="python">
          {`import openai
import mlflow

client = OpenAI(api_key="<YOUR_API_KEY">)

# Example of loading and using the prompt
prompt = mlflow.load_prompt("prompts:/${promptVersion?.name}/${promptVersion?.version}")

response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": prompt.format(${variableNames.map((name) => `${name}="<${name}>"`).join(', ')})
    }]
)

print(response.choices[0].content)`}
        </CodeSnippet>
        {/* "content": prompt.format(question="<question>") */}
      </Modal>
    </div>
  );
};
