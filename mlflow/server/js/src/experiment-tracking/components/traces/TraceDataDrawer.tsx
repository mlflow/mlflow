import {
  DangerIcon,
  Drawer,
  Empty,
  Spacer,
  TableSkeleton,
  TitleSkeleton,
  Typography,
  WarningIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { getTraceDisplayName } from './TracesView.utils';
import { useExperimentTraceData } from './hooks/useExperimentTraceData';
import {
  type ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerSkeleton,
} from '@databricks/web-shared/model-trace-explorer';
import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { useExperimentTraceInfo } from './hooks/useExperimentTraceInfo';

export const TraceDataDrawer = ({
  requestId,
  traceInfo,
  loadingTraceInfo,
  onClose,
  selectedSpanId,
  onSelectSpan,
}: {
  requestId: string;
  traceInfo?: ModelTraceInfo;
  loadingTraceInfo?: boolean;
  onClose: () => void;
  selectedSpanId?: string;
  onSelectSpan?: (selectedSpanId?: string) => void;
}) => {
  const {
    traceData,
    loading: loadingTraceData,
    error,
  } = useExperimentTraceData(
    requestId,
    // skip fetching trace data if trace is in progress
    traceInfo?.status === 'IN_PROGRESS',
  );
  const { theme } = useDesignSystemTheme();

  // Usually, we rely on the parent component to provide trace info object (when clicked in a table row).
  // But in some cases it's not available (e.g. when deep linking to a trace when the entity is not on the same page)
  // and then we fetch it independently here.
  const shouldFetchTraceInfo = !loadingTraceInfo && !traceInfo;

  const { traceInfo: internalTraceInfo, loading: loadingInternalTracingInfo } = useExperimentTraceInfo(
    requestId,
    shouldFetchTraceInfo,
  );

  const traceInfoToUse = traceInfo || internalTraceInfo;

  const title = useMemo(() => {
    if (loadingTraceInfo || loadingInternalTracingInfo) {
      return <TitleSkeleton />;
    }
    if (traceInfoToUse) {
      return (
        <Typography.Title level={2} withoutMargins>
          {getTraceDisplayName(traceInfoToUse as ModelTraceInfo)}
        </Typography.Title>
      );
    }
    return requestId;
  }, [
    // Memo dependency list
    loadingTraceInfo,
    loadingInternalTracingInfo,
    traceInfoToUse,
    requestId,
  ]);

  // Construct the model trace object with the trace info and trace data
  const combinedModelTrace = useMemo(
    () =>
      traceData
        ? {
            info: traceInfoToUse || {},
            data: traceData,
          }
        : undefined,
    [traceData, traceInfoToUse],
  );

  const containsSpans = (traceData?.spans || []).length > 0;

  const renderContent = () => {
    if (loadingTraceData || loadingTraceInfo || loadingInternalTracingInfo) {
      return <ModelTraceExplorerSkeleton />;
    }
    if (traceInfo?.status === 'IN_PROGRESS') {
      return (
        <>
          <Spacer size="lg" />
          <Empty
            image={<WarningIcon />}
            description={
              <FormattedMessage
                defaultMessage="Trace data is not available for in-progress traces. Please wait for the trace to complete."
                description="Experiment page > traces data drawer > in-progress description"
              />
            }
            title={
              <FormattedMessage
                defaultMessage="Trace data not available"
                description="Experiment page > traces data drawer > in-progress title"
              />
            }
          />
        </>
      );
    }
    if (error) {
      return (
        <>
          <Spacer size="lg" />
          <Empty
            image={<DangerIcon />}
            description={
              <FormattedMessage
                defaultMessage="An error occurred while attempting to fetch the trace data. Please wait a moment and try again."
                description="Experiment page > traces data drawer > error state description"
              />
            }
            title={
              <FormattedMessage
                defaultMessage="Error"
                description="Experiment page > traces data drawer > error state title"
              />
            }
          />
        </>
      );
    }
    if (!containsSpans) {
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
    }
    if (combinedModelTrace) {
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
          <ModelTraceExplorer
            modelTrace={combinedModelTrace}
            selectedSpanId={selectedSpanId}
            onSelectSpan={onSelectSpan}
          />
        </div>
      );
    }
    return null;
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
        componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracedatadrawer.tsx_222"
        width="90vw"
        title={title}
        expandContentToFullHeight
      >
        {renderContent()}
      </Drawer.Content>
    </Drawer.Root>
  );
};
