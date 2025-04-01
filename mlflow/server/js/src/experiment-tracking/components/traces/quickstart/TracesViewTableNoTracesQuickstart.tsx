import { Header, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { isNil, keys } from 'lodash';
import { TraceTableGenericQuickstart } from './TraceTableGenericQuickstart';
import { QUICKSTART_CONTENT, QUICKSTART_FLAVOR } from './TraceTableQuickstart.utils';

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
      <Tabs.Root componentId={`${baseComponentId}.traces_table.quickstart`} defaultValue="openai">
        <Tabs.List>
          <Tabs.Trigger value="openai">
            <FormattedMessage
              defaultMessage="OpenAI"
              description="Header for OpenAI tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="langchain">
            <FormattedMessage
              defaultMessage="LangChain / LangGraph"
              description="Header for LangChain / LangGraph tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="llama_index">
            <FormattedMessage
              defaultMessage="LlamaIndex"
              description="Header for LlamaIndex tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="dspy">
            <FormattedMessage
              defaultMessage="DSPy"
              description="Header for DSPy tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="crewai">
            <FormattedMessage
              defaultMessage="CrewAI"
              description="Header for CrewAI tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="autogen">
            <FormattedMessage
              defaultMessage="AutoGen"
              description="Header for AutoGen tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="anthropic">
            <FormattedMessage
              defaultMessage="Anthropic"
              description="Header for Anthropic tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="bedrock">
            <FormattedMessage
              defaultMessage="Bedrock"
              description="Header for Bedrock tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="litellm">
            <FormattedMessage
              defaultMessage="LiteLLM"
              description="Header for LiteLLM tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="gemini">
            <FormattedMessage
              defaultMessage="Gemini"
              description="Header for Gemini tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value="custom">
            <FormattedMessage
              defaultMessage="Custom"
              description="Header for custom tracing tab in the MLflow Tracing quickstart guide"
            />
          </Tabs.Trigger>
        </Tabs.List>
        {keys(QUICKSTART_CONTENT).map((flavorName) => (
          <Tabs.Content value={flavorName as QUICKSTART_FLAVOR} key={flavorName + '_content'}>
            <TraceTableGenericQuickstart
              flavorName={flavorName as QUICKSTART_FLAVOR}
              baseComponentId={baseComponentId}
            />
          </Tabs.Content>
        ))}
      </Tabs.Root>
    </div>
  );
};
