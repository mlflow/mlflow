import { useState, useEffect, useMemo } from 'react';

import {
  useDesignSystemTheme,
  Typography,
  Pagination,
  Empty,
  DangerIcon,
  InfoTooltip,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelTraceExplorer, getTraceArtifact } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';

const MLFLOW_DOCS_URI = 'https://mlflow.org/docs/latest/llms/tracing/index.html?ref=jupyter-notebook-widget';

const getMlflowUILinkForTrace = (traceId: string, experimentId: string) => {
  const queryParams = new URLSearchParams();
  queryParams.append('selectedTraceId', traceId);
  queryParams.append('compareRunsMode', 'TRACES');
  return `/#/experiments/${experimentId}?${queryParams.toString()}`;
};

export const ModelTraceExplorerOSSNotebookRenderer = () => {
  const traceIds = useMemo(() => new URLSearchParams(window.location.search).getAll('trace_id'), []);
  const experimentIds = useMemo(() => new URLSearchParams(window.location.search).getAll('experiment_id'), []);

  const [activeTraceIndex, setActiveTraceIndex] = useState(traceIds.length > 0 ? 0 : null);
  const [traceData, setTraceData] = useState<ModelTrace | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { theme } = useDesignSystemTheme();

  useEffect(() => {
    const fetchTraceData = async () => {
      if (activeTraceIndex === null) {
        return;
      }

      setIsLoading(true);
      const activeTraceId = traceIds[activeTraceIndex];
      const data = await getTraceArtifact(activeTraceId);

      if (typeof data === 'string') {
        setError(data);
      } else {
        setTraceData(data);
      }

      setIsLoading(false);
    };

    fetchTraceData();
  }, [activeTraceIndex, traceIds]);

  if (traceIds.length === 0 || activeTraceIndex === null) {
    return null;
  }

  if (isLoading) {
    return (
      <div css={{ width: 'calc(100% - 2px)' }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Fetching trace data..."
            description="MLflow trace notebook output > loading state"
          />
        </Typography.Text>
      </div>
    );
  }

  // some error occured
  if (!traceData) {
    return (
      <div css={{ paddingTop: theme.spacing.md, width: 'calc(100% - 2px)' }}>
        <Empty
          image={<DangerIcon />}
          description={
            <>
              <Typography.Paragraph>
                <FormattedMessage
                  defaultMessage="An error occurred while attempting to fetch trace data (ID: {traceId}). Please ensure that the MLflow tracking server is running, and that the trace data exists. Error details:"
                  description="An error message explaining that an error occured while fetching trace data"
                  values={{ traceId: traceIds[activeTraceIndex] }}
                />
              </Typography.Paragraph>
              {error && <Typography.Paragraph>{error}</Typography.Paragraph>}
            </>
          }
          title={
            <FormattedMessage defaultMessage="Error" description="MLflow trace notebook output > error state title" />
          }
        />
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        // -2px to prevent bottom border from being cut off
        height: 'calc(100% - 2px)',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Title level={4} withoutMargins>
            MLflow Trace UI
          </Typography.Title>
          <InfoTooltip
            componentId="mlflow.notebook.trace-ui-info"
            iconTitle="More information"
            maxWidth={400}
            content={
              <FormattedMessage
                defaultMessage="To disable or enable this display, call {disableFunction} or {enableFunction} and re-run the cell"
                description="Content in an info popover that instructs the user on how to disable the display. The disable and enable functions are code snippets with the real function names"
                values={{
                  disableFunction: <code>mlflow.tracing.disable_notebook_display()</code>,
                  enableFunction: <code>mlflow.tracing.enable_notebook_display()</code>,
                }}
              />
            }
          />
          <Typography.Link
            componentId="mlflow.notebook.trace-ui-learn-more-link"
            href={MLFLOW_DOCS_URI}
            openInNewTab
            title="Learn More"
          >
            <FormattedMessage
              defaultMessage="Learn More"
              description="Link to MLflow documentation for more information on MLflow tracing"
            />
          </Typography.Link>
        </div>
        <Typography.Link
          componentId="mlflow.notebook.trace-ui-see-in-mlflow-link"
          href={getMlflowUILinkForTrace(traceIds[activeTraceIndex], experimentIds[activeTraceIndex])}
          openInNewTab
          title="View in MLflow UI"
        >
          <FormattedMessage
            defaultMessage="View in MLflow UI"
            description="Link to the MLflow UI for an alternate view of the current trace"
          />
        </Typography.Link>
      </div>
      {traceIds.length > 1 && (
        <Pagination
          componentId="mlflow.notebook.pagination"
          currentPageIndex={activeTraceIndex + 1}
          dangerouslySetAntdProps={{
            showQuickJumper: true,
          }}
          numTotal={traceIds.length}
          onChange={(index) => setActiveTraceIndex(index - 1)}
          pageSize={1}
          style={{ marginBottom: theme.spacing.sm, paddingLeft: theme.spacing.sm, paddingRight: theme.spacing.sm }}
        />
      )}
      <div css={{ flex: 1, overflow: 'hidden' }}>
        <ModelTraceExplorer modelTrace={traceData} initialActiveView="detail" />
      </div>
    </div>
  );
};
