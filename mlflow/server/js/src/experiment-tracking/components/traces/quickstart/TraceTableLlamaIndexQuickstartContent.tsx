import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export const TraceTableLlamaIndexQuickstartContent = ({
  baseComponentId,
  experimentId,
  runUuid,
}: {
  baseComponentId: string;
  experimentId: string | null;
  runUuid?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const code = `import mlflow
from llama_index.core import Document, VectorStoreIndex
${experimentId ? `\nmlflow.set_experiment(experiment_id="${experimentId}")` : ''}
mlflow.llama_index.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

# Querying the engine will cause a trace to be logged
${runUuid ? `with mlflow.start_run():\n    ` : ''}query_engine.query("What is LlamaIndex?")`;

  return (
    <div>
      <Typography.Text>
        <FormattedMessage
          defaultMessage="Automatically log traces for LlamaIndex queries by calling the {code} function. For example:"
          description="Description of how to log traces for the LlamaIndex package using MLflow autologging. This message is followed by a code example."
          values={{
            code: <code>mlflow.llama_index.autolog()</code>,
          }}
        />
      </Typography.Text>
      <div css={{ position: 'relative', width: 'min-content' }}>
        <CopyButton
          componentId={`${baseComponentId}.traces_table.llama_index_quickstart_snippet_copy`}
          css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          showLabel={false}
          copyText={code}
          icon={<CopyIcon />}
        />
        <CodeSnippet
          showLineNumbers
          style={{
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            marginTop: theme.spacing.md,
          }}
          language="python"
        >
          {code}
        </CodeSnippet>
      </div>
    </div>
  );
};
