import { CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export const TraceTableCustomQuickstartContent = ({
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
${experimentId ? `\nmlflow.set_experiment(experiment_id="${experimentId}")\n` : ''}
@mlflow.trace
def foo(a):
    return a + bar(a)

# Various attributes can be passed to the decorator
# to modify the information contained in the span
@mlflow.trace(name="custom_name", attributes={"key": "value"})
def bar(b):
    return b + 1

# Invoking the traced function will cause a trace to be logged
${runUuid ? `with mlflow.start_run():\n    ` : ''}foo(1)`;

  return (
    <div>
      <Typography.Text>
        <FormattedMessage
          defaultMessage={
            'To manually instrument your own traces, the most convenient method is to use the {code} function decorator. ' +
            'This will cause the inputs and outputs of the function to be captured in the trace. For example:'
          }
          description="Description of how to log custom code traces using MLflow. This message is followed by a code example."
          values={{
            code: <code>@mlflow.trace</code>,
          }}
        />
      </Typography.Text>
      <div css={{ position: 'relative', width: 'min-content' }}>
        <CopyButton
          componentId={`${baseComponentId}.traces_table.custom_quickstart_snippet_copy`}
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
            marginBottom: theme.spacing.md,
          }}
          language="python"
        >
          {code}
        </CodeSnippet>
      </div>
      <Typography.Text>
        <FormattedMessage
          defaultMessage={
            'For more complex use cases, MLflow also provides granular APIs that can be used to ' +
            'control tracing behavior. For more information, please visit the <a>official documentation</a> on ' +
            'fluent and client APIs for MLflow Tracing.'
          }
          description={
            'Explanation of alternative APIs for custom tracing in MLflow. ' +
            'The link leads to the MLflow documentation for the user to learn more.'
          }
          values={{
            a: (text: string) => (
              <Typography.Link
                title="official documentation"
                componentId={`${baseComponentId}.traces_table.custom_tracing_docs_link`}
                href="https://mlflow.org/docs/latest/llms/tracing/index.html#tracing-fluent-apis"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
      </Typography.Text>
    </div>
  );
};
