import { getModelTraceId, SingleChatTurnMessages, type ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { Button, ParagraphSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { MutableRefObject } from 'react';

export const ExperimentSingleChatConversation = ({
  traces,
  selectedTurnIndex,
  setSelectedTurnIndex,
  setSelectedTrace,
  chatRefs,
}: {
  traces: ModelTrace[];
  selectedTurnIndex: number | null;
  setSelectedTurnIndex: (turnIndex: number | null) => void;
  setSelectedTrace: (trace: ModelTrace) => void;
  chatRefs: MutableRefObject<{ [traceId: string]: HTMLDivElement }>;
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
              padding: theme.spacing.md,
            }}
            onMouseEnter={() => setSelectedTurnIndex(index)}
          >
            <SingleChatTurnMessages key={traceId} trace={trace} />
            {isActive && (
              <Button
                componentId="mlflow.experiment.chat-session.view-trace"
                size="small"
                color="primary"
                css={{
                  position: 'absolute',
                  top: theme.spacing.md,
                  right: theme.spacing.md,
                  backgroundColor: theme.colors.actionDefaultBackgroundDefault,
                }}
                onClick={() => setSelectedTrace(trace)}
              >
                <FormattedMessage
                  defaultMessage="View trace"
                  description="Button to view a full trace within a chat session"
                />
              </Button>
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
