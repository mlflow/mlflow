import {
  DangerIcon,
  Drawer,
  Empty,
  Spacer,
  TableSkeleton,
  TitleSkeleton,
  WarningIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { getTraceDisplayName } from './TracesView.utils';
import { useExperimentTraceData } from './hooks/useExperimentTraceData';
import {
  type ModelTrace,
  ModelTraceInfo,
  ModelTraceExplorerFrameRenderer,
} from '@databricks/web-shared/model-trace-explorer';
import { useMemo } from 'react';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { FormattedMessage } from 'react-intl';
import { useExperimentTraceInfo } from './hooks/useExperimentTraceInfo';

export const TraceDataDrawer = ({
  requestId,
  traceInfo,
  loadingTraceInfo,
  onClose,
}: {
  requestId: string;
  traceInfo?: ModelTraceInfo;
  loadingTraceInfo?: boolean;
  onClose: () => void;
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
      return getTraceDisplayName(traceInfoToUse);
    }
    return requestId;
  }, [traceInfoToUse, loadingTraceInfo, loadingInternalTracingInfo, requestId]);

  // Construct the model trace object with the trace info and trace data
  const combinedModelTrace = useMemo(
    () =>
      traceData
        ? {
            // We're assigning values redunantly due to a name change in the upstream interface,
            // will be cleaned up shortly
            trace_info: traceInfoToUse || {},
            info: traceInfoToUse || {},
            trace_data: traceData,
            data: traceData,
          }
        : undefined,
    [traceData, traceInfoToUse],
  );

  const containsSpans = (traceData?.spans || []).length > 0;

  const renderContent = () => {
    if (loadingTraceData || loadingTraceInfo || loadingInternalTracingInfo) {
      return (
        <>
          <TitleSkeleton />
          <TableSkeleton lines={5} />
        </>
      );
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
                defaultMessage="An error occurred while attemptying to fetch the trace data. Please wait a moment and try again."
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
            overflow: 'hidden',
          }}
          onWheel={(e) => e.stopPropagation()}
        >
          <ModelTraceExplorerFrameRenderer modelTrace={combinedModelTrace as ModelTrace} height="100%" />
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
      <Drawer.Content width="85vw" title={title} expandContentToFullHeight>
        {renderContent()}
      </Drawer.Content>
    </Drawer.Root>
  );
};
