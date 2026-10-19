import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { TracesV4PageWrapper } from '../../components/experiment-page/components/traces-v4/components/TracesV4PageWrapper';
import { EvalRunCompareSelectionBar } from './EvalRunCompareSelectionBar';
import { EvalRunComparisonOverviewTab } from './EvalRunComparisonOverviewTab';
import { EvalRunComparisonTracesV4Table } from './EvalRunComparisonTracesV4Table';
import type { EvalRunComparisonTraceFilter } from './EvalRunComparisonDelta';
import { normalizeComparisonRunUuids } from './evalRunComparisonData';
import { useEvalRunComparisonBaseline } from './useEvalRunComparisonBaseline';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import {
  BASELINE_TRACE_ID_PARAM,
  CANDIDATE_TRACE_ID_PARAM,
  PAIRWISE_REVIEW_MODE_PARAM,
} from '../../components/experiment-page/components/traces-v4/hooks/useTracesV4UrlState';

enum EvalRunCompareTab {
  Overview = 'overview',
  TraceView = 'traceView',
}

export interface EvalRunCompareViewProps {
  experimentId: string;
  /** The run the comparison is anchored on (the detail page's path runUuid). */
  baselineRunUuid: string;
  /** Other runs being compared against the baseline; empty until the user adds some. */
  compareRunUuids: string[];
  /** Receives the action that starts pairwise review, or undefined when it isn't available. */
  onPairwiseReviewActionChange?: (action: (() => void) | undefined) => void;
}

/**
 * Compare surface rendered on the eval-run detail page when compare mode is on. The run
 * selection bar sits above the tab shell so the Overview and Details tabs share one
 * selection.
 */
export const EvalRunCompareView = ({
  experimentId,
  baselineRunUuid,
  compareRunUuids,
  onPairwiseReviewActionChange,
}: EvalRunCompareViewProps) => {
  const { theme } = useDesignSystemTheme();
  const onSelectBaseline = useEvalRunComparisonBaseline(experimentId, baselineRunUuid, compareRunUuids);
  const runUuids = useMemo(
    () => normalizeComparisonRunUuids(baselineRunUuid, compareRunUuids),
    [baselineRunUuid, compareRunUuids],
  );
  const comparisonKey = JSON.stringify([experimentId, runUuids]);
  const [searchParams, setSearchParams] = useSearchParams();
  const isPairwiseReviewMode = searchParams.get(PAIRWISE_REVIEW_MODE_PARAM) === 'true';
  // A link to a trace comparison (e.g. a shared review) opens on Details, where its drawer lives.
  const hasTraceComparisonParams = Boolean(
    searchParams.get(BASELINE_TRACE_ID_PARAM) && searchParams.get(CANDIDATE_TRACE_ID_PARAM),
  );
  const [activeTab, setActiveTab] = useState<EvalRunCompareTab>(
    hasTraceComparisonParams ? EvalRunCompareTab.TraceView : EvalRunCompareTab.Overview,
  );
  const [outcomeFilter, setOutcomeFilter] = useState<{
    filter: EvalRunComparisonTraceFilter;
    comparisonKey: string;
  }>();
  useEffect(() => setOutcomeFilter(undefined), [comparisonKey]);
  // Each hand-off remounts the Details table so it starts from the requested outcome filter.
  const [filterRequestId, setFilterRequestId] = useState(0);
  const navigateToTraceView = useCallback(
    (filter?: EvalRunComparisonTraceFilter) => {
      setOutcomeFilter(filter ? { filter, comparisonKey } : undefined);
      setFilterRequestId((id) => id + 1);
      setActiveTab(EvalRunCompareTab.TraceView);
    },
    [comparisonKey],
  );
  const hasComparison = runUuids.length > 1;
  // The Details table owns the review action; from Overview, review switches to Details and runs it once it's there.
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;
  const openPairwiseReviewRef = useRef<() => void>();
  const pendingPairwiseReviewRef = useRef(false);
  const handlePairwiseReviewActionChange = useCallback((action: (() => void) | undefined) => {
    openPairwiseReviewRef.current = action;
    if (action && activeTabRef.current === EvalRunCompareTab.TraceView && pendingPairwiseReviewRef.current) {
      pendingPairwiseReviewRef.current = false;
      action();
    }
  }, []);
  const enablePairwiseReview = shouldEnableNewEvalRunsUi() && hasComparison;
  const startPairwiseReview = useCallback(() => {
    if (activeTabRef.current === EvalRunCompareTab.TraceView && openPairwiseReviewRef.current) {
      openPairwiseReviewRef.current();
      return;
    }
    pendingPairwiseReviewRef.current = true;
    setActiveTab(EvalRunCompareTab.TraceView);
  }, []);

  useEffect(() => {
    onPairwiseReviewActionChange?.(enablePairwiseReview ? startPairwiseReview : undefined);
    return () => onPairwiseReviewActionChange?.(undefined);
  }, [onPairwiseReviewActionChange, startPairwiseReview, enablePairwiseReview]);

  // R starts pairwise review, or leaves it while reviewing; typing in a field doesn't count.
  useEffect(() => {
    if (!enablePairwiseReview) {
      return;
    }
    const handleReviewShortcut = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement) {
        const isEditableTarget =
          event.target.matches('textarea, select, input:not([type="radio"]):not([type="checkbox"])') ||
          event.target.isContentEditable ||
          Boolean(event.target.closest('[contenteditable="true"]'));
        if (isEditableTarget) {
          return;
        }
      }
      if (event.key.toLowerCase() === 'r' && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        if (isPairwiseReviewMode) {
          setSearchParams((params) => {
            params.delete(PAIRWISE_REVIEW_MODE_PARAM);
            return params;
          });
        } else {
          startPairwiseReview();
        }
      }
    };
    window.addEventListener('keydown', handleReviewShortcut);
    return () => window.removeEventListener('keydown', handleReviewShortcut);
  }, [isPairwiseReviewMode, setSearchParams, startPairwiseReview, enablePairwiseReview]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        gap: theme.spacing.md,
      }}
    >
      <EvalRunCompareSelectionBar
        experimentId={experimentId}
        baselineRunUuid={baselineRunUuid}
        compareRunUuids={compareRunUuids}
        onSelectBaseline={onSelectBaseline}
      />

      <Tabs.Root
        componentId="mlflow.eval_run_details.compare.tabs"
        value={activeTab}
        onValueChange={(value) => {
          if (value === EvalRunCompareTab.Overview || value === EvalRunCompareTab.TraceView) {
            if (value === EvalRunCompareTab.Overview) setOutcomeFilter(undefined);
            setActiveTab(value);
          }
        }}
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
        }}
      >
        <Tabs.List>
          <Tabs.Trigger value={EvalRunCompareTab.Overview}>
            <FormattedMessage
              defaultMessage="Overview"
              description="Eval-run compare view tab showing the side-by-side run comparison overview"
            />
          </Tabs.Trigger>
          <Tabs.Trigger value={EvalRunCompareTab.TraceView}>
            <FormattedMessage
              defaultMessage="Details"
              description="Eval-run compare view tab showing the row-level side-by-side result comparison"
            />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content
          value={EvalRunCompareTab.Overview}
          mountMode="active"
          css={{ flex: 1, minHeight: 0, overflowY: 'auto' }}
        >
          <EvalRunComparisonOverviewTab
            experimentId={experimentId}
            baselineRunUuid={baselineRunUuid}
            compareRunUuids={compareRunUuids}
            onNavigateToTraceView={hasComparison ? navigateToTraceView : undefined}
            enableOutcomeFilter={hasComparison}
          />
        </Tabs.Content>
        <Tabs.Content
          value={EvalRunCompareTab.TraceView}
          css={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {hasComparison ? (
            <TracesV4PageWrapper resetKey={comparisonKey}>
              <EvalRunComparisonTracesV4Table
                key={`${comparisonKey}:${filterRequestId}`}
                experimentId={experimentId}
                runUuids={runUuids}
                initialOutcomeFilter={outcomeFilter?.comparisonKey === comparisonKey ? outcomeFilter.filter : undefined}
                onPairwiseReviewActionChange={handlePairwiseReviewActionChange}
              />
            </TracesV4PageWrapper>
          ) : (
            <EvalRunComparisonNoRunsState />
          )}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

const EvalRunComparisonNoRunsState = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ padding: theme.spacing.lg, color: theme.colors.textSecondary }}>
      <FormattedMessage
        defaultMessage="Add a run to compare against the baseline."
        description="Details tab of the eval-run comparison when only the baseline run is selected"
      />
    </div>
  );
};
