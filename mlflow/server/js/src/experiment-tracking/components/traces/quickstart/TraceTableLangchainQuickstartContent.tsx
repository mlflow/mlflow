import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export const TraceTableLangchainQuickstartContent = ({
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
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
${experimentId ? `\nmlflow.set_experiment(experiment_id="${experimentId}")` : ''}
mlflow.langchain.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
llm = OpenAI()
prompt = PromptTemplate.from_template("Answer the following question: {question}")
chain = prompt | llm

# Invoking the chain will cause a trace to be logged
${runUuid ? `with mlflow.start_run():\n    ` : ''}chain.invoke("What is MLflow?")`;

  return (
    <div>
      <Typography.Text>
        <FormattedMessage
          defaultMessage="Automatically log traces for LangChain or LangGraph invocations by calling the {code} function. For example:"
          description="Description of how to log traces for the LangChain/LangGraph package using MLflow autologging. This message is followed by a code example."
          values={{
            code: <code>mlflow.langchain.autolog()</code>,
          }}
        />
      </Typography.Text>
      <div css={{ position: 'relative', width: 'min-content' }}>
        <CopyButton
          componentId={`${baseComponentId}.traces_table.langchain_quickstart_snippet_copy`}
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
