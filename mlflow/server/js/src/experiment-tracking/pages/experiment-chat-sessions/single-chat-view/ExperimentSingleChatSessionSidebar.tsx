import { ChainIcon, TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { getModelTraceId, type ModelTrace, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { MutableRefObject } from 'react';
import { useCallback } from 'react';

export const ExperimentSingleChatSessionSidebar = ({
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

  const scrollToTrace = useCallback(
    (trace: ModelTrace) => {
      const traceId = getModelTraceId(trace);
      if (chatRefs.current[traceId]) {
        chatRefs.current[traceId].scrollIntoView({ behavior: 'smooth' });
      }
    },
    [chatRefs],
  );

  if (!traces) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        width: 200,
        borderRight: `1px solid ${theme.colors.border}`,
        gap: theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
      }}
    >
      {traces.map((trace, index) => (
        <div
          key={getModelTraceId(trace)}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            backgroundColor: selectedTurnIndex === index ? theme.colors.actionDefaultBackgroundHover : undefined,
            padding: theme.spacing.xs,
            borderRadius: theme.borders.borderRadiusSm,
            cursor: 'pointer',
          }}
          onMouseEnter={() => setSelectedTurnIndex(index)}
          onClick={() => scrollToTrace(trace)}
        >
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: theme.general.iconSize,
              height: theme.general.iconSize,
              borderRadius: theme.borders.borderRadiusSm,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            <ChainIcon color="ai" />
          </div>
          <Typography.Text>Turn {index + 1}</Typography.Text>
        </div>
      ))}
    </div>
  );
};

export const ExperimentSingleChatSessionSidebarSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        width: 200,
        borderRight: `1px solid ${theme.colors.border}`,
        gap: theme.spacing.xs,
        paddingTop: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
      }}
    >
      <TitleSkeleton css={{ width: '60%' }} />
      <TitleSkeleton css={{ width: '80%' }} />
      <TitleSkeleton css={{ width: '70%' }} />
    </div>
  );
};
