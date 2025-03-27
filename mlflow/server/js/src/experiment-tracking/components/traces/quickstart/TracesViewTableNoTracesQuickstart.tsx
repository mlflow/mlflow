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

  const tabs = [
    {
      value: 'langchain',
      title: (
        <FormattedMessage
          defaultMessage="LangChain / LangGraph"
          description="Header for LangChain / LangGraph tab in the MLflow Tracing quickstart guide"
        />
      ),
      component: TraceTableLangchainQuickstartContent,
    },
    {
      value: 'llama-index',
      title: (
        <FormattedMessage
          defaultMessage="LlamaIndex"
          description="Header for LlamaIndex tab in the MLflow Tracing quickstart guide"
        />
      ),
      component: TraceTableLlamaIndexQuickstartContent,
    },
    {
      value: 'autogen',
      title: (
        <FormattedMessage
          defaultMessage="AutoGen"
          description="Header for AutoGen tab in the MLflow Tracing quickstart guide"
        />
      ),
      component: TraceTableAutogenQuickstartContent,
    },
    {
      value: 'openai',
      title: (
        <FormattedMessage
          defaultMessage="OpenAI"
          description="Header for OpenAI tab in the MLflow Tracing quickstart guide"
        />
      ),
      component: TraceTableOpenAIQuickstartContent,
    },
    {
      value: 'custom',
      title: (
        <FormattedMessage
          defaultMessage="Custom"
          description="Header for custom tracing tab in the MLflow Tracing quickstart guide"
        />
      ),
      component: TraceTableCustomQuickstartContent,
    },
  ];

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
          {tabs.map((tab) => (
            <Tabs.Trigger key={tab.value} value={tab.value}>
              {tab.title}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        {tabs.map((tab) => {
          const Component = tab.component;
          return (
            <Tabs.Content key={tab.value} value={tab.value}>
              <Component baseComponentId={baseComponentId} experimentId={experimentId} runUuid={runUuid} />
            </Tabs.Content>
          );
        })}
      </Tabs.Root>
    </div>
  );
};
