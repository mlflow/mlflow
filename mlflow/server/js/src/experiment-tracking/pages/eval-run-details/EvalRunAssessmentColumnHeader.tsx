import { Button, ChevronDownIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { HeaderContext } from '@tanstack/react-table';
import { useIntl } from 'react-intl';
import type { AssessmentFilter, AssessmentValueType } from '@databricks/web-shared/genai-traces-table';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { AssessmentColumnSummary } from '@databricks/web-shared/genai-traces-table/components/charts/AssessmentColumnSummary';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { TraceTableColumn } from '@databricks/web-shared/traces-table';
import {
  assessmentNameFromColumnId,
  isAssessmentColumnId,
} from '../../components/experiment-page/components/traces-v4/utils/assessmentColumns';
import { AssessmentSourceIcon } from '../../components/experiment-page/components/traces-v4/components/AssessmentSourceIcon';
import type { EvalRunAssessmentAggregate } from './useEvalRunAssessmentAggregates';
import { EvalRunBaselineDeltaPill } from './EvalRunBaselineDeltaPill';

interface EvalRunAssessmentColumnMeta {
  evalRunScorer: EvalRunAssessmentAggregate;
  assessmentFilters: AssessmentFilter[];
  toggleAssessmentFilter: (assessmentName: string, filterValue: AssessmentValueType) => void;
  runUuid: string;
  areChartsCollapsed: boolean;
  /** The scorer's saved run metric, shown instead of the summary's score when the run has one. */
  currentScore?: number;
  baselineScore?: number;
  /** Set when the experiment has a baseline run other than this one; shows the delta pill. */
  baselineRunHref?: string;
}

const hasEvalRunAssessmentColumnMeta = (meta: unknown): meta is EvalRunAssessmentColumnMeta =>
  typeof meta === 'object' && meta !== null && 'evalRunScorer' in meta && meta.evalRunScorer !== undefined;

/** Pass/fail and boolean scorers have a chart in the header; the others show only their summary value. */
const hasCollapsibleChart = (scorer: EvalRunAssessmentAggregate) =>
  scorer.dtype === 'pass-fail' || scorer.dtype === 'boolean';

/** Assessment header with the scorer's summary over the run: rate or value, and a chart for pass/fail scores. */
const EvalRunAssessmentColumnHeader = ({ column }: HeaderContext<ModelTraceInfoV3, unknown>) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { meta } = column.columnDef;
  if (!hasEvalRunAssessmentColumnMeta(meta)) {
    return null;
  }
  const {
    evalRunScorer: scorer,
    assessmentFilters,
    toggleAssessmentFilter,
    runUuid,
    areChartsCollapsed,
    currentScore,
    baselineScore,
    baselineRunHref,
  } = meta;
  const isRate = scorer.dtype === 'pass-fail' || scorer.dtype === 'boolean';

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
        <AssessmentSourceIcon
          sourceType={scorer.assessmentInfo.source?.sourceType}
          testIdPrefix="eval-run-assessment"
        />
        <Typography.Text bold size="sm" color="secondary" ellipsis>
          {scorer.displayName}
        </Typography.Text>
      </div>
      <div css={{ width: '100%', whiteSpace: 'nowrap' }}>
        <AssessmentColumnSummary
          theme={theme}
          intl={intl}
          assessmentInfo={scorer.assessmentInfo}
          assessmentAggregates={scorer.aggregates}
          allAssessmentFilters={assessmentFilters}
          toggleAssessmentFilter={(assessmentName, filterValue) => toggleAssessmentFilter(assessmentName, filterValue)}
          currentRunDisplayName={runUuid}
          displayScoreOverride={
            currentScore === undefined
              ? undefined
              : intl.formatNumber(
                  currentScore,
                  isRate ? { style: 'percent', maximumFractionDigits: 0 } : { maximumFractionDigits: 2 },
                )
          }
          baselineDeltaPill={
            baselineRunHref ? (
              <EvalRunBaselineDeltaPill
                isRate={isRate}
                currentScore={currentScore}
                baselineScore={baselineScore}
                baselineRunHref={baselineRunHref}
              />
            ) : undefined
          }
          collapsedHeader={areChartsCollapsed || !hasCollapsibleChart(scorer)}
          categoricalChartLayout="inline"
        />
      </div>
    </div>
  );
};

/** Chevron on the header's bottom edge that collapses or expands the header charts. */
export const EvalRunAssessmentHeaderToggle = ({
  areChartsCollapsed,
  onToggle,
}: {
  areChartsCollapsed: boolean;
  onToggle: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const label = areChartsCollapsed
    ? intl.formatMessage({
        defaultMessage: 'Expand assessment charts',
        description: 'Accessible label for expanding the score charts in the evaluation run table headers',
      })
    : intl.formatMessage({
        defaultMessage: 'Collapse assessment charts',
        description: 'Accessible label for collapsing the score charts in the evaluation run table headers',
      });

  return (
    <>
      {/* Masks the header's bottom border behind the chevron. */}
      <div
        aria-hidden
        css={{
          position: 'absolute',
          left: '50%',
          top: 'calc(100% - 2px)',
          width: 24,
          height: 12,
          backgroundColor: theme.colors.backgroundPrimary,
          transform: 'translate(-50%, -50%)',
        }}
      />
      <div
        css={{
          position: 'absolute',
          left: '50%',
          top: 'calc(100% - 2px)',
          display: 'flex',
          width: 24,
          height: 24,
          boxSizing: 'border-box',
          pointerEvents: 'auto',
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          backgroundColor: theme.colors.backgroundPrimary,
          boxShadow: `inset 0 0 0 2px ${theme.colors.border}, ${theme.shadows.sm}`,
          overflow: 'hidden',
          opacity: 0.5,
          ':hover, :focus-within': {
            boxShadow: `inset 0 0 0 2px ${theme.colors.actionDefaultBorderFocus}, ${theme.shadows.sm}`,
            opacity: 1,
          },
        }}
      >
        <Button
          componentId="mlflow.eval-run-traces-v4.toggle-assessment-charts"
          size="small"
          type="tertiary"
          icon={
            <ChevronDownIcon
              css={{
                color: theme.colors.textSecondary,
                transform: areChartsCollapsed ? 'none' : 'rotate(180deg)',
                transition: 'transform 0.2s',
              }}
            />
          }
          aria-label={label}
          aria-expanded={!areChartsCollapsed}
          onClick={onToggle}
        />
      </div>
    </>
  );
};

/**
 * Gives each assessment column whose scorer has a summary the summary header. Columns without one (the
 * summaries are still loading, or the scorer isn't in the summarized traces) keep their plain header.
 */
export const buildEvalRunAssessmentColumnDefs = ({
  columnDefs,
  scorersByName,
  assessmentFilters,
  toggleAssessmentFilter,
  runUuid,
  areChartsCollapsed,
  currentScores,
  baselineScores,
  baselineRunHref,
}: {
  columnDefs: TraceTableColumn[];
  scorersByName: ReadonlyMap<string, EvalRunAssessmentAggregate>;
  assessmentFilters: AssessmentFilter[];
  toggleAssessmentFilter: (assessmentName: string, filterValue: AssessmentValueType) => void;
  runUuid: string;
  areChartsCollapsed: boolean;
  /** Saved run metrics keyed by scorer name and by metric name (see `getEvalRunScores`). */
  currentScores?: Record<string, number>;
  baselineScores?: Record<string, number>;
  baselineRunHref?: string;
}): TraceTableColumn[] =>
  columnDefs.map((column): TraceTableColumn => {
    const columnId = column.id;
    if (!columnId || !isAssessmentColumnId(columnId)) {
      return column;
    }
    const assessmentName = assessmentNameFromColumnId(columnId);
    const scorer = scorersByName.get(assessmentName);
    if (!scorer) {
      return column;
    }
    // String scorers summarize as a count of values, so a saved metric has no score to replace.
    const showsScore = scorer.dtype === 'pass-fail' || scorer.dtype === 'boolean' || scorer.dtype === 'numeric';
    const meta: EvalRunAssessmentColumnMeta = {
      evalRunScorer: scorer,
      assessmentFilters,
      toggleAssessmentFilter,
      runUuid,
      areChartsCollapsed,
      ...(showsScore && {
        currentScore: currentScores?.[assessmentName] ?? currentScores?.[scorer.assessmentInfo.metricName],
        baselineScore: baselineScores?.[assessmentName] ?? baselineScores?.[scorer.assessmentInfo.metricName],
        baselineRunHref,
      }),
    };
    return {
      ...column,
      id: columnId,
      header: EvalRunAssessmentColumnHeader,
      headerVariant: 'rich',
      meta: { ...column.meta, ...meta },
    };
  });

/** Whether any summary header has a chart, so the collapse toggle has something to do. */
export const hasCollapsibleAssessmentChart = (columnDefs: TraceTableColumn[]): boolean =>
  columnDefs.some(
    (column) => hasEvalRunAssessmentColumnMeta(column.meta) && hasCollapsibleChart(column.meta.evalRunScorer),
  );
