import { useCallback, useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { TracesV4PageWrapper } from '../../components/experiment-page/components/traces-v4/components/TracesV4PageWrapper';
import { EvalRunCompareSelectionBar } from './EvalRunCompareSelectionBar';
import { EvalRunComparisonOverviewTab } from './EvalRunComparisonOverviewTab';
import { EvalRunComparisonTracesV4Table } from './EvalRunComparisonTracesV4Table';
import type { EvalRunComparisonTraceFilter } from './EvalRunComparisonDelta';
import { normalizeComparisonRunUuids } from './evalRunComparisonData';
import { useEvalRunComparisonBaseline } from './useEvalRunComparisonBaseline';

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
}

/**
 * Compare surface rendered on the eval-run detail page when compare mode is on. The run
 * selection bar sits above the tab shell so the Overview and Details tabs share one
 * selection.
 */
export const EvalRunCompareView = ({ experimentId, baselineRunUuid, compareRunUuids }: EvalRunCompareViewProps) => {
  const { theme } = useDesignSystemTheme();
  const onSelectBaseline = useEvalRunComparisonBaseline(experimentId, baselineRunUuid, compareRunUuids);
  const runUuids = useMemo(
    () => normalizeComparisonRunUuids(baselineRunUuid, compareRunUuids),
    [baselineRunUuid, compareRunUuids],
  );
  const comparisonKey = JSON.stringify([experimentId, runUuids]);
  const [activeTab, setActiveTab] = useState<EvalRunCompareTab>(EvalRunCompareTab.Overview);
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
