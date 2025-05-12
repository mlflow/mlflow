import { Header, Tabs, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { isNil, keys } from 'lodash';
import { TraceTableGenericQuickstart } from './TraceTableGenericQuickstart';
import { QUICKSTART_CONTENT, QUICKSTART_FLAVOR, QUICKSTART_TAB_MESSAGES } from './TraceTableQuickstart.utils';

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
          {(Object.keys(QUICKSTART_TAB_MESSAGES) as QUICKSTART_FLAVOR[]).map((flavor) => (
            <Tabs.Trigger key={flavor} value={flavor}>
              <FormattedMessage {...QUICKSTART_TAB_MESSAGES[flavor]} />
            </Tabs.Trigger>
          ))}
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
