import { Drawer, Empty, Spacer, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ModelTraceExplorer } from '@databricks/web-shared/model-trace-explorer';
import { useGetTracesById } from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';

export interface TraceModalProps {
  visible: boolean;
  onClose: () => void;
  /** v3 (tr-…) or v4 (trace:/…) trace id to load. */
  traceId: string;
}

/**
 * Drawer wrapping `ModelTraceExplorer` for the dataset record side panel's "open trace"
 * affordance. Mirrors the pattern in `EvaluationTraceDataDrawer` from genai-traces-table.
 */
export const TraceModal = ({ visible, onClose, traceId }: TraceModalProps) => {
  const { theme } = useDesignSystemTheme();
  const traceIds = visible && traceId ? [traceId] : [];
  const { data, isLoading } = useGetTracesById(traceIds);
  const trace: ModelTrace | undefined = data[0];
  const hasSpans = Boolean(trace?.data?.spans?.length);

  return (
    <Drawer.Root open={visible} onOpenChange={(open) => !open && onClose()}>
      <Drawer.Content
        componentId="mlflow.eval-datasets-v2.trace-modal"
        title={
          <Typography.Title level={3} withoutMargins>
            {traceId || (
              <FormattedMessage
                defaultMessage="Trace"
                description="Title shown in the v2 dataset record side-panel trace modal before a trace id is known"
              />
            )}
          </Typography.Title>
        }
        width="60vw"
      >
        {isLoading ? (
          <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Spinner />
          </div>
        ) : !trace || !hasSpans ? (
          <>
            <Spacer size="lg" />
            <Empty
              description={null}
              title={
                <FormattedMessage
                  defaultMessage="No trace data recorded"
                  description="Empty state in the v2 dataset record side-panel trace modal when the trace has no spans"
                />
              }
            />
          </>
        ) : (
          <div
            css={{
              height: '100%',
              marginLeft: -theme.spacing.lg,
              marginRight: -theme.spacing.lg,
              marginBottom: -theme.spacing.lg,
            }}
            onWheel={(e) => e.stopPropagation()}
          >
            <ModelTraceExplorer modelTrace={trace} />
          </div>
        )}
      </Drawer.Content>
    </Drawer.Root>
  );
};
