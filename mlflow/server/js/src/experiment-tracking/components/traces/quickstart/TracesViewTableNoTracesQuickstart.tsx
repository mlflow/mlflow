import { Typography, useDesignSystemTheme } from '@databricks/design-system';
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
    <div
      css={{ flex: 1, flexDirection: 'column', display: 'flex', alignItems: 'center', paddingTop: theme.spacing.lg }}
    >
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="No traces recorded"
          description="Message displayed when there are no traces logged to the experiment"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
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
                  componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_components_traces_quickstart_tracesviewtablenotracesquickstart_46"
                  href="https://mlflow.org/docs/latest/genai/tracing/"
                  openInNewTab
                >
                  {text}
                </Typography.Link>
              ),
            }}
          />
        )}
      </Typography.Paragraph>
      <TraceTableGenericQuickstart flavorName="custom" baseComponentId={baseComponentId} />
    </div>
  );
};
