import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject, type ReactNode } from 'react';
import {
  ApplyDesignSystemContextOverrides,
  Button,
  DangerIcon,
  DropdownMenu,
  GenericSkeleton,
  importantify,
  ListIcon,
  ParagraphSkeleton,
  RefreshIcon,
  SlidersIcon,
  SpeechBubbleIcon,
  TitleSkeleton,
  TokenIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl, type IntlShape } from '@databricks/i18n';
import {
  formatCostUSD,
  getModelTraceId,
  getTraceCost,
  getTraceTokenUsage,
  isSessionLevelAssessment,
  isV3ModelTraceInfo,
  ModelTraceExplorerUpdateTraceContextProvider,
  ModelTraceExplorerV2AssessmentPaneToggle,
  ModelTraceExplorerV2AssessmentsPane,
  ModelTraceExplorerV2ResizablePane,
  ModelTraceExplorerV2SingleChatTurnAssessments,
  ModelTraceExplorerV2SingleChatTurnMessages,
  ModelTraceExplorerV2ViewStateProvider,
  shouldEnableAssessmentsInSessions,
  useModelTraceExplorerV2ViewState,
  type ModelTrace,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { formatTraceDuration } from '@databricks/web-shared/traces-table';
import { ExperimentSingleChatIcon } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-chat-sessions/single-chat-view/ExperimentSingleChatIcon';

type TurnMetric = 'timestamp' | 'duration' | 'tokens' | 'cost';

interface TurnMetricDisplay {
  key: TurnMetric;
  value: string;
  title: string;
  icon?: ReactNode;
}

const TURN_METRIC_OPTIONS = ['timestamp', 'duration', 'tokens', 'cost'] as const satisfies readonly TurnMetric[];

const scrollTurnIntoView = (turn: HTMLDivElement | undefined, behavior: ScrollBehavior) => {
  const scrollContainer = turn?.parentElement;
  if (turn && scrollContainer) {
    const offset = turn.getBoundingClientRect().top - scrollContainer.getBoundingClientRect().top;
    scrollContainer.scrollTo({ top: scrollContainer.scrollTop + offset, behavior });
  }
};

const getTurnMetricLabel = (metric: TurnMetric, intl: IntlShape): string => {
  switch (metric) {
    case 'timestamp':
      return intl.formatMessage({ defaultMessage: 'Timestamp', description: 'Session turn metric option' });
    case 'duration':
      return intl.formatMessage({ defaultMessage: 'Duration', description: 'Session turn metric option' });
    case 'tokens':
      return intl.formatMessage({ defaultMessage: 'Total tokens', description: 'Session turn metric option' });
    case 'cost':
      return intl.formatMessage({ defaultMessage: 'Estimated LLM cost', description: 'Session turn metric option' });
  }
};

const SessionTurnDisplaySettings = ({
  visibleMetrics,
  setVisibleMetrics,
  onRefreshSession,
}: {
  visibleMetrics: TurnMetric[];
  setVisibleMetrics: (metrics: TurnMetric[]) => void;
  onRefreshSession: () => void;
}) => {
  const intl = useIntl();
  const settingsLabel = intl.formatMessage({
    defaultMessage: 'Turn display settings',
    description: 'Accessible label for session turn display settings button',
  });
  const selectedMetricSet = new Set(visibleMetrics);

  return (
    <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
      <DropdownMenu.Root>
        <Tooltip componentId="mlflow.traces-v4.session-turn-settings-tooltip" content={settingsLabel}>
          <DropdownMenu.Trigger asChild aria-label={settingsLabel}>
            <Button
              componentId="mlflow.traces-v4.session-turn-settings"
              icon={<SlidersIcon />}
              size="small"
              aria-label={settingsLabel}
            />
          </DropdownMenu.Trigger>
        </Tooltip>
        <DropdownMenu.Content align="end">
          <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger>
              <DropdownMenu.IconWrapper>
                <ListIcon />
              </DropdownMenu.IconWrapper>
              {intl.formatMessage({
                defaultMessage: 'Display metric types',
                description: 'Session turn settings submenu for visible metrics',
              })}
            </DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent>
              {TURN_METRIC_OPTIONS.map((metric) => (
                <DropdownMenu.CheckboxItem
                  key={metric}
                  componentId="mlflow.traces-v4.session-toggle-turn-metric"
                  checked={visibleMetrics.includes(metric)}
                  onSelect={(event) => event.preventDefault()}
                  onCheckedChange={(checked) => {
                    setVisibleMetrics(
                      TURN_METRIC_OPTIONS.filter((option) =>
                        option === metric ? checked : selectedMetricSet.has(option),
                      ),
                    );
                  }}
                >
                  <DropdownMenu.ItemIndicator />
                  {getTurnMetricLabel(metric, intl)}
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.SubContent>
          </DropdownMenu.Sub>
          <DropdownMenu.Separator />
          <DropdownMenu.Item componentId="mlflow.traces-v4.session-refresh" onSelect={onRefreshSession}>
            <DropdownMenu.IconWrapper>
              <RefreshIcon />
            </DropdownMenu.IconWrapper>
            {intl.formatMessage({
              defaultMessage: 'Refresh session',
              description: 'Action to refresh all turns in the open trace session',
            })}
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </ApplyDesignSystemContextOverrides>
  );
};

const SessionTurnMetrics = ({ trace, visibleMetrics }: { trace: ModelTrace; visibleMetrics: TurnMetric[] }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const info = isV3ModelTraceInfo(trace.info) ? trace.info : undefined;
  if (!info) {
    return null;
  }

  const duration = info.execution_duration
    ? (formatTraceDuration(info.execution_duration) ?? info.execution_duration)
    : undefined;
  const totalTokens = getTraceTokenUsage(info)?.total_tokens;
  const totalCost = getTraceCost(info)?.total_cost;
  const metrics = visibleMetrics.flatMap<TurnMetricDisplay>((metric) => {
    switch (metric) {
      case 'timestamp':
        return [];
      case 'duration':
        return duration
          ? [{ key: metric, value: duration, title: getTurnMetricLabel(metric, intl), icon: undefined }]
          : [];
      case 'tokens':
        return typeof totalTokens === 'number'
          ? [
              {
                key: metric,
                value: intl.formatNumber(totalTokens, { notation: 'compact', maximumFractionDigits: 1 }),
                title: intl.formatMessage(
                  { defaultMessage: '{count} total tokens', description: 'Tooltip for a session turn token count' },
                  { count: totalTokens },
                ),
                icon: <TokenIcon />,
              },
            ]
          : [];
      case 'cost':
        return typeof totalCost === 'number'
          ? [
              {
                key: metric,
                value: formatCostUSD(totalCost, 4),
                title: getTurnMetricLabel(metric, intl),
                icon: undefined,
              },
            ]
          : [];
    }
    return [];
  });

  if (metrics.length === 0) {
    return null;
  }

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
      {metrics.map((metric) => (
        <Tooltip key={metric.key} componentId="mlflow.traces-v4.session-turn-metric-tooltip" content={metric.title}>
          <Typography.Text
            size="sm"
            color="secondary"
            css={{ display: 'inline-flex', alignItems: 'center', gap: 2, whiteSpace: 'nowrap' }}
          >
            {metric.icon}
            {metric.value}
          </Typography.Text>
        </Tooltip>
      ))}
    </div>
  );
};

const SessionTurnsPane = ({
  traces,
  selectedTurnIndex,
  setSelectedTurnIndex,
  chatRefs,
  onRefreshSession,
}: {
  traces: ModelTrace[];
  selectedTurnIndex: number | null;
  setSelectedTurnIndex: (turnIndex: number | null) => void;
  chatRefs: MutableRefObject<Record<string, HTMLDivElement>>;
  onRefreshSession: () => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [visibleMetrics, setVisibleMetrics] = useState<TurnMetric[]>([...TURN_METRIC_OPTIONS]);
  const scrollToTrace = useCallback(
    (trace: ModelTrace) => {
      scrollTurnIntoView(chatRefs.current[getModelTraceId(trace)], 'smooth');
    },
    [chatRefs],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        width: '100%',
        minWidth: 0,
        minHeight: 0,
        borderRight: `1px solid ${theme.colors.border}`,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          paddingBottom: 3,
          boxSizing: 'border-box',
          minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.xs,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Turns" description="Header for turns in the trace session drawer" />
          </Typography.Text>
          <Typography.Text size="sm" color="secondary">
            <FormattedMessage
              defaultMessage="{count, plural, one {# turn} other {# turns}}"
              description="Turn count shown beside the turns pane title"
              values={{ count: traces.length }}
            />
          </Typography.Text>
        </div>
        <SessionTurnDisplaySettings
          visibleMetrics={visibleMetrics}
          setVisibleMetrics={setVisibleMetrics}
          onRefreshSession={onRefreshSession}
        />
      </div>
      <div css={{ minHeight: 0, overflowY: 'auto' }}>
        {traces.map((trace, index) => {
          const info = isV3ModelTraceInfo(trace.info) ? trace.info : undefined;
          const requestTime = info?.request_time ? Date.parse(info.request_time) : Number.NaN;
          return (
            <div
              key={getModelTraceId(trace)}
              css={{
                display: 'flex',
                gap: theme.spacing.sm,
                position: 'relative',
                minHeight: 64,
                backgroundColor:
                  selectedTurnIndex === index ? theme.colors.actionDefaultBackgroundHover : 'transparent',
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                boxSizing: 'border-box',
                cursor: 'pointer',
                ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
                ':active': { backgroundColor: theme.colors.actionDefaultBackgroundPress },
                '::before': {
                  content: '""',
                  position: 'absolute',
                  left: theme.spacing.md + theme.general.iconSize / 2,
                  top: index === 0 ? theme.spacing.sm + theme.general.iconSize / 2 : 0,
                  bottom:
                    index === traces.length - 1 ? `calc(100% - ${theme.spacing.sm + theme.general.iconSize / 2}px)` : 0,
                  width: 1,
                  backgroundColor: theme.colors.branded.ai.gradientStart,
                },
              }}
              onClick={() => {
                setSelectedTurnIndex(index);
                scrollToTrace(trace);
              }}
              onMouseEnter={() => setSelectedTurnIndex(index)}
            >
              <div css={{ position: 'relative', zIndex: 1 }}>
                <ExperimentSingleChatIcon />
              </div>
              <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0, flex: 1 }}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                  <Typography.Text bold css={{ whiteSpace: 'nowrap' }}>
                    <FormattedMessage
                      defaultMessage="Turn {turnNumber}"
                      description="Label for a turn in the session navigation pane"
                      values={{ turnNumber: index + 1 }}
                    />
                  </Typography.Text>
                  {visibleMetrics.includes('timestamp') && !Number.isNaN(requestTime) && (
                    <Typography.Text
                      size="sm"
                      color="secondary"
                      ellipsis
                      css={{ minWidth: 0, marginLeft: 'auto', textAlign: 'right' }}
                    >
                      {intl.formatDate(requestTime, {
                        month: '2-digit',
                        day: '2-digit',
                        hour: 'numeric',
                        minute: '2-digit',
                      })}
                    </Typography.Text>
                  )}
                </div>
                <div
                  css={{
                    display: 'flex',
                    marginTop: theme.spacing.sm,
                  }}
                >
                  <SessionTurnMetrics trace={trace} visibleMetrics={visibleMetrics} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const SessionConversation = ({
  traces,
  selectedTurnIndex,
  setSelectedTurnIndex,
  onViewTrace,
  chatRefs,
  getAssessmentTitle,
}: {
  traces: ModelTrace[];
  selectedTurnIndex: number | null;
  setSelectedTurnIndex: (turnIndex: number | null) => void;
  onViewTrace: (trace: ModelTrace) => void;
  chatRefs: MutableRefObject<Record<string, HTMLDivElement>>;
  getAssessmentTitle: (assessmentName: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();
  const assessmentsEnabled = shouldEnableAssessmentsInSessions();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        overflowY: 'auto',
      }}
    >
      {traces.map((trace, index) => {
        const isActive = selectedTurnIndex === index;
        const traceId = getModelTraceId(trace);
        return (
          <div
            ref={(element) => {
              if (element) {
                chatRefs.current[traceId] = element;
              }
            }}
            key={traceId}
            css={{
              display: 'flex',
              position: 'relative',
              padding: theme.spacing.md,
              margin: `${theme.spacing.sm}px ${theme.spacing.sm}px 0`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent',
              ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
            }}
            onMouseEnter={() => setSelectedTurnIndex(index)}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, flex: 1, minWidth: 0 }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Turn {turnNumber}"
                    description="Label for a single turn within an experiment chat session"
                    values={{ turnNumber: index + 1 }}
                  />
                </Typography.Text>
                {assessmentsEnabled && (
                  <div
                    css={{
                      flex: 1,
                      minWidth: 0,
                      '& > div': { marginTop: 0 },
                    }}
                  >
                    <ModelTraceExplorerV2SingleChatTurnAssessments
                      trace={trace}
                      getAssessmentTitle={getAssessmentTitle}
                      visibleItemsCount={1}
                    />
                  </div>
                )}
                {!assessmentsEnabled && <div css={{ flex: 1 }} />}
                <Button
                  componentId="mlflow.traces-v4.session-view-trace"
                  size="small"
                  color="primary"
                  css={[
                    { visibility: isActive ? 'visible' : 'hidden' },
                    importantify({ backgroundColor: theme.colors.backgroundPrimary }),
                  ]}
                  onClick={() => onViewTrace(trace)}
                >
                  <FormattedMessage
                    defaultMessage="View full trace"
                    description="Button to view a full trace within a chat session"
                  />
                </Button>
              </div>
              <ModelTraceExplorerV2SingleChatTurnMessages trace={trace} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

const SessionAssessmentsTitle = () => (
  <Typography.Title level={3} withoutMargins css={{ flexShrink: 0 }}>
    <FormattedMessage defaultMessage="Session assessments" description="Title for the session-level assessments pane" />
  </Typography.Title>
);

const SessionDrawerLoadingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flex: 1, minWidth: 0, minHeight: 0 }}>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          width: 280,
          flexShrink: 0,
          borderRight: `1px solid ${theme.colors.border}`,
        }}
      >
        <div
          css={{
            minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
            display: 'flex',
            alignItems: 'center',
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          }}
        >
          <TitleSkeleton css={{ width: '35%' }} />
        </div>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, padding: theme.spacing.md }}>
          {[60, 75, 68].map((width) => (
            <GenericSkeleton key={width} css={{ width: `${width}%`, height: theme.general.heightBase }} />
          ))}
        </div>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <div
          css={{
            minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
            display: 'flex',
            alignItems: 'center',
            padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          <TitleSkeleton css={{ width: '25%' }} />
        </div>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.lg,
            padding: theme.spacing.md,
          }}
        >
          {[0, 1].map((section) => (
            <div key={section} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, width: '80%' }}>
              <TitleSkeleton css={{ width: '20%' }} />
              <ParagraphSkeleton seed={`session-drawer-${section}`} />
              <ParagraphSkeleton seed={`session-drawer-${section}-messages`} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const SessionConversationPane = ({
  sessionId,
  traces,
  selectedTurnIndex,
  setSelectedTurnIndex,
  onViewTrace,
  chatRefs,
  getAssessmentTitle,
}: {
  sessionId: string;
  traces: ModelTrace[];
  selectedTurnIndex: number | null;
  setSelectedTurnIndex: (turnIndex: number | null) => void;
  onViewTrace: (trace: ModelTrace) => void;
  chatRefs: MutableRefObject<Record<string, HTMLDivElement>>;
  getAssessmentTitle: (assessmentName: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [paneWidth, setPaneWidth] = useState(500);
  const { assessmentsPaneExpanded } = useModelTraceExplorerV2ViewState();
  const firstTrace = traces[0];
  const firstTraceInfo = firstTrace && isV3ModelTraceInfo(firstTrace.info) ? firstTrace.info : undefined;
  const sessionAssessments = useMemo(
    () => firstTraceInfo?.assessments?.filter(isSessionLevelAssessment) ?? [],
    [firstTraceInfo?.assessments],
  );

  const conversation = (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          minHeight: theme.spacing.xl + 2 * theme.spacing.sm,
          padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
          boxSizing: 'border-box',
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexShrink: 0 }}>
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: theme.spacing.xl,
              height: theme.spacing.xl,
              svg: {
                width: theme.typography.fontSizeLg,
                height: theme.typography.fontSizeLg,
              },
            }}
          >
            <SpeechBubbleIcon />
          </span>
          <Typography.Text bold size="lg">
            <FormattedMessage defaultMessage="Session" description="Title for the session conversation pane" />
          </Typography.Text>
        </div>
        <Tooltip componentId="mlflow.traces-v4.session-id-tooltip" content={sessionId}>
          <Typography.Text color="secondary" ellipsis css={{ minWidth: 0 }}>
            {sessionId}
          </Typography.Text>
        </Tooltip>
        <div css={{ flex: 1 }} />
        <ModelTraceExplorerV2AssessmentPaneToggle
          assessmentCount={sessionAssessments.length}
          assessmentTarget="session"
        />
      </div>
      <SessionConversation
        traces={traces}
        selectedTurnIndex={selectedTurnIndex}
        setSelectedTurnIndex={setSelectedTurnIndex}
        onViewTrace={onViewTrace}
        chatRefs={chatRefs}
        getAssessmentTitle={getAssessmentTitle}
      />
    </div>
  );

  if (!assessmentsPaneExpanded || !firstTraceInfo) {
    return conversation;
  }

  return (
    <ModelTraceExplorerV2ResizablePane
      initialRatio={0.65}
      paneWidth={paneWidth}
      setPaneWidth={setPaneWidth}
      leftChild={conversation}
      leftMinWidth={160}
      rightChild={
        <ModelTraceExplorerV2AssessmentsPane
          assessments={sessionAssessments}
          traceId={firstTraceInfo.trace_id}
          sessionId={sessionId}
          assessmentsTitleOverride={SessionAssessmentsTitle}
        />
      }
      rightMinWidth={280}
    />
  );
};

export interface TracesV4SessionDrawerContentProps {
  sessionId: string;
  selectedTraceId?: string;
  onRefreshSession: () => void;
  traces: ModelTrace[];
  isLoading: boolean;
  error?: unknown;
  selectedSqlWarehouseId?: string;
  getAssessmentTitle: (assessmentName: string) => string;
  onViewTrace: (traceInfo: ModelTraceInfoV3) => void;
}

export const TracesV4SessionDrawerContent = ({
  sessionId,
  selectedTraceId,
  onRefreshSession,
  traces,
  isLoading,
  error,
  selectedSqlWarehouseId,
  getAssessmentTitle,
  onViewTrace,
}: TracesV4SessionDrawerContentProps) => {
  const { theme } = useDesignSystemTheme();
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);
  const [turnsPaneWidth, setTurnsPaneWidth] = useState(280);
  const chatRefs = useRef<Record<string, HTMLDivElement>>({});
  const appliedInitialSelectionRef = useRef<string | undefined>(undefined);
  const orderedTraces = useMemo(
    () =>
      [...traces].sort((left, right) => {
        const leftTime = isV3ModelTraceInfo(left.info) ? Date.parse(left.info.request_time) : 0;
        const rightTime = isV3ModelTraceInfo(right.info) ? Date.parse(right.info.request_time) : 0;
        return leftTime - rightTime;
      }),
    [traces],
  );
  useEffect(() => {
    const initialSelectionKey = selectedTraceId ? `${sessionId}:${selectedTraceId}` : undefined;
    if (!selectedTraceId || !initialSelectionKey || appliedInitialSelectionRef.current === initialSelectionKey) {
      return;
    }
    const selectedIndex = orderedTraces.findIndex((trace) => getModelTraceId(trace) === selectedTraceId);
    if (selectedIndex < 0) {
      return;
    }
    appliedInitialSelectionRef.current = initialSelectionKey;
    setSelectedTurnIndex(selectedIndex);
    scrollTurnIntoView(chatRefs.current[selectedTraceId], 'auto');
  }, [orderedTraces, selectedTraceId, sessionId]);
  const traceInfoById = useMemo(
    () =>
      new Map(
        orderedTraces.flatMap((trace) =>
          isV3ModelTraceInfo(trace.info) ? [[trace.info.trace_id, trace.info] as const] : [],
        ),
      ),
    [orderedTraces],
  );
  const handleViewTrace = useCallback(
    (trace: ModelTrace) => {
      const traceInfo = traceInfoById.get(getModelTraceId(trace));
      if (traceInfo) {
        onViewTrace(traceInfo);
      }
    },
    [onViewTrace, traceInfoById],
  );

  return (
    <div
      css={{
        height: '100%',
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        marginLeft: -theme.spacing.lg,
        marginRight: -theme.spacing.lg,
      }}
    >
      {isLoading ? (
        <SessionDrawerLoadingSkeleton />
      ) : error ? (
        <div
          css={{
            display: 'flex',
            flex: 1,
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.md,
          }}
        >
          <DangerIcon css={{ fontSize: 48, color: theme.colors.actionDangerPrimaryBackgroundDefault }} />
          <Typography.Title level={4}>
            <FormattedMessage
              defaultMessage="Failed to load session"
              description="Title shown in the V4 session drawer when the session fails to load"
            />
          </Typography.Title>
          <Button componentId="mlflow.traces-v4.session-load-error.retry" onClick={onRefreshSession}>
            <FormattedMessage
              defaultMessage="Retry"
              description="Button to retry loading a session in the V4 trace drawer"
            />
          </Button>
        </div>
      ) : orderedTraces.length === 0 ? (
        <div css={{ display: 'flex', flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="No conversation turns found in this session."
              description="Empty state in the V4 session drawer"
            />
          </Typography.Text>
        </div>
      ) : (
        <ModelTraceExplorerV2ResizablePane
          initialRatio={0.25}
          paneWidth={turnsPaneWidth}
          setPaneWidth={setTurnsPaneWidth}
          leftMinWidth={280}
          leftChild={
            <SessionTurnsPane
              traces={orderedTraces}
              selectedTurnIndex={selectedTurnIndex}
              setSelectedTurnIndex={setSelectedTurnIndex}
              chatRefs={chatRefs}
              onRefreshSession={onRefreshSession}
            />
          }
          rightMinWidth={320}
          rightChild={
            <ModelTraceExplorerUpdateTraceContextProvider
              sqlWarehouseId={selectedSqlWarehouseId}
              modelTraceInfo={orderedTraces[0].info}
              chatSessionId={sessionId}
            >
              <ModelTraceExplorerV2ViewStateProvider
                key={sessionId}
                modelTrace={orderedTraces[0]}
                assessmentsPaneEnabled
                initialAssessmentsPaneCollapsed
              >
                <SessionConversationPane
                  sessionId={sessionId}
                  traces={orderedTraces}
                  selectedTurnIndex={selectedTurnIndex}
                  setSelectedTurnIndex={setSelectedTurnIndex}
                  onViewTrace={handleViewTrace}
                  chatRefs={chatRefs}
                  getAssessmentTitle={getAssessmentTitle}
                />
              </ModelTraceExplorerV2ViewStateProvider>
            </ModelTraceExplorerUpdateTraceContextProvider>
          }
        />
      )}
    </div>
  );
};
