import { Alert, CopyIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';

export const ExperimentLoggedModelDetailsTracesIntroductionText = ({ modelId }: { modelId: string }) => {
  const { theme } = useDesignSystemTheme();
  const code = `import mlflow
          
mlflow.set_active_model(model_id="${modelId}")`;

  return (
    <>
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage="This tab displays all the traces logged to this logged model. MLflow supports automatic tracing for many popular generative AI frameworks. Follow the steps below to log your first trace. For more information about MLflow Tracing, visit the <a>MLflow documentation</a>."
          description="Message that explains the function of the 'Traces' tab in logged model page. This message is followed by a tutorial explaining how to get started with MLflow Tracing."
          values={{
            a: (text: string) => (
              <Typography.Link
                componentId="mlflow.logged_model.traces.traces_table.quickstart_docs_link"
                href="https://mlflow.org/docs/latest/llms/tracing/index.html"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage="You can start logging traces to this logged model by calling {code} first:"
          description='Introductory text for the code example for logging traces to an existing logged model. The code contains reference to "mlflow.set_active_model" function call'
          values={{
            code: <code>mlflow.set_active_model</code>,
          }}
        />
      </Typography.Paragraph>
      <Typography.Paragraph>
        <div css={{ position: 'relative', width: 'min-content' }}>
          <CopyButton
            componentId="mlflow.logged_model.traces.traces_table.set_active_model_quickstart_snippet_copy"
            css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
            showLabel={false}
            copyText={code}
            icon={<CopyIcon />}
          />
          <CodeSnippet
            showLineNumbers
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
            }}
            language="python"
          >
            {code}
          </CodeSnippet>
        </div>
      </Typography.Paragraph>
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage="Next, you can log traces to this logged model depending on your framework:"
          description="Introductory text for the code example for logging traces to an existing logged model. This part is displayed after the code example for setting the active model."
        />
      </Typography.Paragraph>
    </>
  );
};
