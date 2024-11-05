import { Header, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { TraceTableLangchainQuickstartContent } from './TraceTableLangchainQuickstartContent';
import { TraceTableLlamaIndexQuickstartContent } from './TraceTableLlamaIndexQuickstartContent';
import { TraceTableCustomQuickstartContent } from './TraceTableCustomQuickstartContent';
import { TraceTableAutogenQuickstartContent } from './TraceTableAutogenQuickstartContent';
import { TraceTableOpenAIQuickstartContent } from './TraceTableOpenAIQuickstartContent';
import { isNil } from 'lodash';

export const TracesViewTableNoTracesQuickstart = ({
  baseComponentId,
  experimentIds,
  runUuid,
}: {
  baseComponentId: string;
  experimentIds: string[];
  runUuid?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  // only display the experiment ID if the user is viewing exactly one experiment
  const experimentId = experimentIds.length === 1 ? experimentIds[0] : null;

  return (
    <div css={{ marginLeft: -theme.spacing.md }}>
      <Header
        title={
          <FormattedMessage
            defaultMessage="No traces recorded"
            description="Message displayed when there are no traces logged to the experiment"
          />
        }
        titleElementLevel={3}
      />
      <Typography.Text
        css={{
          display: 'block',
          marginTop: theme.spacing.md,
          marginBottom: theme.spacing.md,
        }}
      >
        <FormattedMessage
          defaultMessage={
            'This tab displays all the traces logged to this {isRun, select, true {run} other {experiment}}. ' +
            'MLflow supports automatic tracing for many popular generative AI frameworks. Follow the steps below to log ' +
            'your first trace. For more information about MLflow Tracing, visit the <a>MLflow documentation</a>.'
          }
          description={
            "Message that explains the function of the 'Traces' tab in the MLflow UI." +
            'This message is followed by a tutorial explaining how to get started with MLflow Tracing.'
          }
          values={{
            isRun: !isNil(runUuid),
            a: (text: string) => (
              <Typography.Link
                componentId={`${baseComponentId}.traces_table.quickstart_docs_link`}
                href="https://mlflow.org/docs/latest/llms/tracing/index.html"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
      </Typography.Text>
      <Tabs.Root componentId={`${baseComponentId}.traces_table.quickstart`} defaultValue="langchain">
        <Tabs.List>
          <Tabs.Trigger value="langchain">
            <FormattedMessage
              defaultMessage="LangChain / LangGraph"
              description="Header for LangChain / LangGraph tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="llama-index">
            <FormattedMessage
              defaultMessage="LlamaIndex"
              description="Header for LlamaIndex tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="autogen">
            <FormattedMessage
              defaultMessage="AutoGen"
              description="Header for AutoGen tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="openai">
            <FormattedMessage
              defaultMessage="OpenAI"
              description="Header for OpenAI tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="custom">
            <FormattedMessage
              defaultMessage="Custom"
              description="Header for custom tracing tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="langchain">
          <TraceTableLangchainQuickstartContent
            baseComponentId={baseComponentId}
            experimentId={experimentId}
            runUuid={runUuid}
          />
        </Tabs.Content>
        <Tabs.Content value="llama-index">
          <TraceTableLlamaIndexQuickstartContent
            baseComponentId={baseComponentId}
            experimentId={experimentId}
            runUuid={runUuid}
          />
        </Tabs.Content>
        <Tabs.Content value="autogen">
          <TraceTableAutogenQuickstartContent
            baseComponentId={baseComponentId}
            experimentId={experimentId}
            runUuid={runUuid}
          />
        </Tabs.Content>
        <Tabs.Content value="openai">
          <TraceTableOpenAIQuickstartContent
            baseComponentId={baseComponentId}
            experimentId={experimentId}
            runUuid={runUuid}
          />
        </Tabs.Content>
        <Tabs.Content value="custom">
          <TraceTableCustomQuickstartContent
            baseComponentId={baseComponentId}
            experimentId={experimentId}
            runUuid={runUuid}
          />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
