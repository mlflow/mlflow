import { isNil } from 'lodash';

import { Drawer, Empty, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { ModelTraceExplorer } from '@databricks/web-shared/model-trace-explorer';

export const EvaluationTraceDataDrawer = ({
  requestId,
  onClose,
  trace,
}: {
  requestId: string;
  onClose: () => void;
  trace: ModelTrace;
}) => {
  const { theme } = useDesignSystemTheme();
  const title = (
    <Typography.Title level={2} withoutMargins>
      {requestId}
    </Typography.Title>
  );

  const renderContent = () => {
    const containsSpans = trace.data.spans.length > 0;
    if (isNil(trace) || !containsSpans) {
      return (
        <>
          <Spacer size="lg" />
          <Empty
            description={null}
            title={
              <FormattedMessage
                defaultMessage="No trace data recorded"
                description="Experiment page > traces data drawer > no trace data recorded empty state"
              />
            }
          />
        </>
      );
    } else {
      return (
        <div
          css={{
            height: '100%',
            marginLeft: -theme.spacing.lg,
            marginRight: -theme.spacing.lg,
            marginBottom: -theme.spacing.lg,
          }}
          // This is required for mousewheel scrolling within `Drawer`
          onWheel={(e) => e.stopPropagation()}
        >
          <ModelTraceExplorer modelTrace={trace} />
        </div>
      );
    }
  };

  return (
    <Drawer.Root
      modal
      open
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <Drawer.Content
        componentId="mlflow.evaluations_review.trace_data_drawer"
        width="85vw"
        title={title}
        expandContentToFullHeight
      >
        {renderContent()}
      </Drawer.Content>
    </Drawer.Root>
  );
};
