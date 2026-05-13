import { Button, Empty, Typography, useDesignSystemTheme, WarningIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode } from './ModelTrace.types';
import { ModelTraceExplorerCodeSnippetBody } from './ModelTraceExplorerCodeSnippetBody';

const getExampleCode = (traceId: string) =>
  JSON.stringify(`import mlflow

# returns an MLflow Trace object
trace = mlflow.get_trace('${traceId}')

# trace.info contains tags and metadata
trace_info = trace.info

# the trace's spans are stored in trace.data
spans = trace.data.spans`);

export const ModelTraceExplorerTraceTooLargeView = ({
  traceId,
  setForceDisplay,
}: {
  traceId: string;
  setForceDisplay: (forceDisplay: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Empty
      image={<WarningIcon />}
      title={
        <FormattedMessage
          defaultMessage="Large trace"
          description="Modal title for when a trace is very large. The modal warns the user that the trace may cause UI slowness, but allows them to attempt to display it anyway by clicking a button."
        />
      }
      description={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, alignItems: 'center' }}>
          <FormattedMessage
            defaultMessage="This trace is very large, and may cause UI slowness when attempting to render. You can click the button below to attempt to display it anyway."
            description="Body text explaining issues with rendering large traces."
          />
          <Button
            componentId="shared.model-trace-explorer.trace-too-large.force-display-button"
            css={{ width: 'min-content' }}
            onClick={() => setForceDisplay(true)}
          >
            <FormattedMessage
              defaultMessage="Attempt to display"
              description="Button label for forcing the display of a large trace"
            />
          </Button>
          <span>
            <FormattedMessage
              defaultMessage="Alternatively, you can interact with this trace via the MLflow Python SDK, using the {code} function. For more information, please check the {documentation_link}."
              description="Body text for when a trace is too large to display"
              values={{
                code: <Typography.Text code>mlflow.get_trace()</Typography.Text>,
                documentation_link: (
                  <Typography.Link
                    href="https://mlflow.org/docs/latest/api_reference/python_api/mlflow.entities.html#mlflow.entities.Trace"
                    target="_blank"
                    openInNewTab
                    componentId="shared.model-trace-explorer.trace-too-large.documentation-link"
                  >
                    <FormattedMessage
                      defaultMessage="MLflow API documentation"
                      description="Link to the MLflow API documentation for the Trace object"
                    />
                  </Typography.Link>
                ),
              }}
            />
          </span>
          <ModelTraceExplorerCodeSnippetBody
            data={getExampleCode(traceId)}
            renderMode={CodeSnippetRenderMode.PYTHON}
            initialExpanded
          />
        </div>
      }
    />
  );
};
