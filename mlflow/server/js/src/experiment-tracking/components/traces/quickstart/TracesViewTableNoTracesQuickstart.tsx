import { Header, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { isNil } from 'lodash';
import { TraceTableGenericQuickstart } from './TraceTableGenericQuickstart';
import { useTracesViewTableNoTracesQuickstartContext } from './TracesViewTableNoTracesQuickstartContext';

export const TracesViewTableNoTracesQuickstart = ({
  baseComponentId,
  runUuid,
}: {
  baseComponentId: string;
  runUuid?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { introductionText } = useTracesViewTableNoTracesQuickstartContext();

  return (
    <div css={{ overflow: 'auto', paddingBottom: theme.spacing.lg }}>
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
          maxWidth: 800,
        }}
      >
        {introductionText ? (
          introductionText
        ) : (
          <FormattedMessage
            defaultMessage="This tab displays all the traces logged to this {isRun, select, true {run} other {experiment}}. Follow the steps below to log your first trace. For more information about MLflow Tracing, visit the <a>MLflow documentation</a>."
            description="Message that explains the function of the 'Traces' tab in the MLflow UI. This message is followed by a tutorial explaining how to get started with MLflow Tracing."
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
        )}
      </Typography.Text>
      <TraceTableGenericQuickstart flavorName="custom" baseComponentId={baseComponentId} />
    </div>
  );
};
