import { useDesignSystemTheme } from '@databricks/design-system';
import {
  isSessionLevelAssessment,
  ModelTraceExplorerUpdateTraceContextProvider,
  useModelTraceExplorerUpdateTraceContext,
  isV3ModelTraceInfo,
  type ModelTrace,
  AssessmentsPane,
  ASSESSMENT_SESSION_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { first } from 'lodash';
import { useMemo } from 'react';
import { ResizableBox } from 'react-resizable';

const initialWidth = 300;
const maxWidth = 600;

export const ExperimentSingleChatSessionScoreResults = ({
  traces,
  sessionId,
}: {
  traces: ModelTrace[];
  sessionId: string;
}) => {
  const { theme } = useDesignSystemTheme();

  const firstTraceInfoInSession = useMemo(() => {
    const traceInfo = first(traces)?.info;
    if (!traceInfo || !isV3ModelTraceInfo(traceInfo)) {
      return undefined;
    }
    return traceInfo;
  }, [traces]);

  const sessionAssessments = useMemo(
    () => firstTraceInfoInSession?.assessments?.filter(isSessionLevelAssessment) ?? [],
    [firstTraceInfoInSession],
  );

  const defaultMetadata = useMemo(() => ({ [ASSESSMENT_SESSION_METADATA_KEY]: sessionId }), [sessionId]);

  const traceUpdateContext = useModelTraceExplorerUpdateTraceContext();

  if (!firstTraceInfoInSession) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        paddingTop: theme.spacing.sm,
        overflow: 'auto',
      }}
    >
      <ResizableBox
        width={initialWidth}
        height={undefined}
        axis="x"
        resizeHandles={['w']}
        minConstraints={[initialWidth, 150]}
        maxConstraints={[maxWidth, 150]}
        handle={
          <div
            css={{
              width: theme.spacing.sm,
              left: -(theme.spacing.sm / 2),
              height: '100%',
              position: 'absolute',
              top: 0,
              cursor: 'ew-resize',
              '&:hover': {
                backgroundColor: theme.colors.border,
                opacity: 0.5,
              },
            }}
          />
        }
        css={{
          position: 'relative',
          display: 'flex',
          borderLeft: `1px solid ${theme.colors.border}`,
          marginLeft: theme.spacing.sm,
          paddingLeft: theme.spacing.sm,
          flex: 1,
        }}
      >
        {/* Repeat the context from the level above, additionally adding proper trace info and chat session ID */}
        <ModelTraceExplorerUpdateTraceContextProvider
          {...traceUpdateContext}
          modelTraceInfo={firstTraceInfoInSession}
          chatSessionId={sessionId}
        >
          <div
            css={{
              paddingLeft: 0,
              border: 0,
            }}
          >
            <AssessmentsPane
              assessments={sessionAssessments}
              traceId={firstTraceInfoInSession.trace_id}
              defaultMetadata={defaultMetadata}
            />
          </div>
        </ModelTraceExplorerUpdateTraceContextProvider>
      </ResizableBox>
    </div>
  );
};
