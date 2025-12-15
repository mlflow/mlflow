import { useDesignSystemTheme } from '@databricks/design-system';
import {
  isSessionLevelAssessment,
  isV3ModelTraceInfo,
  type ModelTrace,
  AssessmentsPane,
} from '@databricks/web-shared/model-trace-explorer';
import { first } from 'lodash';
import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { ResizableBox } from 'react-resizable';

const initialWidth = 300;
const maxWidth = 600;

export const ExperimentSingleChatSessionScoreResults = ({ traces }: { traces: ModelTrace[] }) => {
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
        <div
          css={{
            '& > div': {
              paddingLeft: 0,
              border: 0,
            },
          }}
        >
          <AssessmentsPane assessments={sessionAssessments} traceId={firstTraceInfoInSession.trace_id} />
        </div>
      </ResizableBox>
    </div>
  );
};
