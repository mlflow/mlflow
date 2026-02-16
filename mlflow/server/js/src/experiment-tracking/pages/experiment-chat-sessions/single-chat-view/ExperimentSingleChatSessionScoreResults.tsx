import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import {
  isSessionLevelAssessment,
  ModelTraceExplorerUpdateTraceContextProvider,
  useModelTraceExplorerUpdateTraceContext,
  isV3ModelTraceInfo,
  type ModelTrace,
  AssessmentsPane,
  ModelTraceExplorerRunJudgesContextProvider,
} from '@databricks/web-shared/model-trace-explorer';
import { first, last } from 'lodash';
import { useEffect, useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { ResizableBox } from 'react-resizable';
import { isEvaluatingTracesInDetailsViewEnabled } from '../../../../shared/web-shared/model-trace-explorer/FeatureUtils';
import { useRunScorerInTracesViewConfiguration } from '../../experiment-scorers/hooks/useRunScorerInTracesViewConfiguration';
import { ScorerEvaluationScope } from '../../experiment-scorers/constants';

const initialWidth = 300;
const maxWidth = 600;

const getAssessmentsPaneComponent = () => {
  return AssessmentsPane;
};

export const ExperimentSingleChatSessionScoreResults = ({
  traces,
  sessionId,
  onRefreshSession,
}: {
  traces: ModelTrace[];
  sessionId: string;
  onRefreshSession?: () => void;
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

  const AssessmentsPaneComponent = getAssessmentsPaneComponent();

  const traceUpdateContext = useModelTraceExplorerUpdateTraceContext();

  if (!firstTraceInfoInSession) {
    return null;
  }

  const assessmentPaneElement = (
    <AssessmentsPaneComponent
      assessments={sessionAssessments}
      traceId={firstTraceInfoInSession.trace_id}
      css={{
        paddingLeft: 0,
        border: 0,
      }}
      assessmentsTitleOverride={AssessmentsTitleOverride}
      disableCloseButton
      enableRunScorer={isEvaluatingTracesInDetailsViewEnabled()}
      sessionId={sessionId}
    />
  );

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
          {isEvaluatingTracesInDetailsViewEnabled() ? (
            <RunJudgeContextProvider onRefreshSession={onRefreshSession} sessionId={sessionId}>
              {assessmentPaneElement}
            </RunJudgeContextProvider>
          ) : (
            assessmentPaneElement
          )}
        </ModelTraceExplorerUpdateTraceContextProvider>
      </ResizableBox>
    </div>
  );
};

const AssessmentsTitleOverride = (count?: number) => (
  <Typography.Title level={3} withoutMargins css={{ flexShrink: 0 }}>
    <FormattedMessage
      defaultMessage="Session scorers{count, plural, =0 {} other { (#)}}"
      values={{ count: count ?? 0 }}
      description="Section title in a side panel that displays session-level scorers"
    />
  </Typography.Title>
);

const RunJudgeContextProvider = ({
  children,
  onRefreshSession,
  sessionId,
}: {
  children: React.ReactNode;
  onRefreshSession?: () => void;
  sessionId: string;
}) => {
  const runJudgesConfiguration = useRunScorerInTracesViewConfiguration(ScorerEvaluationScope.SESSIONS);

  useEffect(() => {
    return runJudgesConfiguration.subscribeToScorerFinished?.((event) => {
      if (event.results?.some((result) => 'sessionId' in result && result.sessionId === sessionId)) {
        onRefreshSession?.();
      }
    });
  }, [runJudgesConfiguration, sessionId, onRefreshSession]);

  return (
    <ModelTraceExplorerRunJudgesContextProvider {...runJudgesConfiguration}>
      {children}
    </ModelTraceExplorerRunJudgesContextProvider>
  );
};
