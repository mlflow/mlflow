import {
  getModelTraceId,
  SingleChatTurnMessages,
  SingleChatTurnAssessments,
  shouldEnableAssessmentsInSessions,
  type ModelTrace,
} from '@databricks/web-shared/model-trace-explorer';
import {
  Button,
  importantify,
  ParagraphSkeleton,
  Spacer,
  TitleSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { MutableRefObject } from 'react';
import { ExperimentSingleChatIcon } from './ExperimentSingleChatIcon';

export const ExperimentSingleChatConversation = ({
  traces,
  selectedTurnIndex,
  setSelectedTurnIndex,
  setSelectedTrace,
  chatRefs,
  getAssessmentTitle,
}: {
  traces: ModelTrace[];
  selectedTurnIndex: number | null;
  setSelectedTurnIndex: (turnIndex: number | null) => void;
  setSelectedTrace: (trace: ModelTrace) => void;
  chatRefs: MutableRefObject<{ [traceId: string]: HTMLDivElement }>;
  getAssessmentTitle: (assessmentName: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();

  if (!traces) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        height: '100%',
        overflowY: 'auto',
        gap: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingLeft: theme.spacing.sm,
      }}
    >
      {traces.map((trace, index) => {
        const isActive = selectedTurnIndex === index;
        const traceId = getModelTraceId(trace);

        return (
          <div
            ref={(el) => {
              if (el) {
                chatRefs.current[traceId] = el;
              }
            }}
            key={traceId}
            css={{
              display: 'flex',
              flexDirection: 'column',
              position: 'relative',
              gap: theme.spacing.sm,
              backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : undefined,
              border: `1px solid ${theme.colors.border}`,
              padding: theme.spacing.md,
              borderRadius: theme.borders.borderRadiusMd,
            }}
            onMouseEnter={() => setSelectedTurnIndex(index)}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <ExperimentSingleChatIcon />
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Turn {turnNumber}"
                  description="Label for a single turn within an experiment chat session"
                  values={{ turnNumber: index + 1 }}
                />
              </Typography.Text>
              <div css={{ flex: 1 }} />
              <Button
                componentId="mlflow.experiment.chat-session.view-trace"
                size="small"
                color="primary"
                css={[
                  {
                    visibility: isActive ? 'visible' : 'hidden',
                  },
                  // Required for button to have an outstanding background over the chat turn hover state
                  importantify({ backgroundColor: theme.colors.backgroundPrimary }),
                ]}
                onClick={() => setSelectedTrace(trace)}
              >
                <FormattedMessage
                  defaultMessage="View full trace"
                  description="Button to view a full trace within a chat session"
                />
              </Button>
            </div>
            {/* TODO: add turn-level metrics */}
            <SingleChatTurnMessages key={traceId} trace={trace} />
            {shouldEnableAssessmentsInSessions() && (
              <>
                <Spacer size="sm" />
                <SingleChatTurnAssessments
                  trace={trace}
                  getAssessmentTitle={getAssessmentTitle}
                  onAddAssessmentsClick={() => setSelectedTrace(trace)}
                />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
};

export const ExperimentSingleChatConversationSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        height: '100%',
        overflowY: 'auto',
        gap: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingLeft: theme.spacing.sm,
      }}
    >
      <div
        css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, width: '60%', alignSelf: 'flex-start' }}
      >
        <TitleSkeleton css={{ width: '20%' }} />
        {[...Array(5).keys()].map((i) => (
          <ParagraphSkeleton key={i} seed={`s-${i}`} />
        ))}
      </div>
      <div
        css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, width: '60%', alignSelf: 'flex-end' }}
      >
        <TitleSkeleton css={{ width: '20%' }} />
        {[...Array(5).keys()].map((i) => (
          <ParagraphSkeleton key={i} seed={`s-${i}`} />
        ))}
      </div>
    </div>
  );
};
