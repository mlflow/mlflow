import { useDesignSystemTheme } from '@databricks/design-system';
// BEGIN-EDGE
import { AssessmentsPaneV2, shouldEnableTracesTabLabelingSchemas } from '@databricks/web-shared/model-trace-explorer';
// END-EDGE
import {
  isSessionLevelAssessment,
  isV3ModelTraceInfo,
  type ModelTrace,
  AssessmentsPane,
  ASSESSMENT_SESSION_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { first, last } from 'lodash';
import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { ResizableBox } from 'react-resizable';

const initialWidth = 300;
const maxWidth = 600;

const getAssessmentsPaneComponent = () => {
  // BEGIN-EDGE
  if (shouldEnableTracesTabLabelingSchemas()) {
    return AssessmentsPaneV2;
  }
  // END-EDGE
  return AssessmentsPane;
};

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

  const AssessmentsPaneComponent = getAssessmentsPaneComponent();

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
        <AssessmentsPaneComponent
          assessments={sessionAssessments}
          traceId={firstTraceInfoInSession.trace_id}
          defaultMetadata={defaultMetadata}
          css={{
            paddingLeft: 0,
            border: 0,
          }}
          assessmentsTitleOverride={AssessmentsTitleOverride}
        />
      </ResizableBox>
    </div>
  );
};

const AssessmentsTitleOverride = (count?: number) => (
  <FormattedMessage
    defaultMessage="Session scorers{count, plural, =0 {} other { (#)}}"
    values={{ count: count ?? 0 }}
    description="Section title in a side panel that displays session-level scorers"
  />
);
