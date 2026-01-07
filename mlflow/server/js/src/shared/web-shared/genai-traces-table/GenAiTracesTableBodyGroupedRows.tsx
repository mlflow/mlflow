import type { Row } from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import React, { useMemo, useState, useCallback } from 'react';

import {
  TableCell,
  TableRow,
  TableRowSelectCell,
  Tag,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
  CheckCircleIcon,
  ClockIcon,
  XCircleIcon,
  SpeechBubbleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type {
  GroupedTracesResult,
  TraceSessionGroup,
  TracesTableColumn,
  AssessmentInfo,
  EvalTraceComparisonEntry,
  SessionAssessmentAggregate,
  RunEvaluationResultAssessment,
} from './types';
import { TracesTableColumnType } from './types';
import { GenAiTracesTableBodyRow } from './GenAiTracesTableBodyRows';
import { EvaluationsReviewAssessmentTag } from './components/EvaluationsReviewAssessmentTag';
import { escapeCssSpecialCharacters } from './utils/DisplayUtils';
import {
  getSessionFirstInput,
  getSessionLastOutput,
  getSessionTotalTokens,
  getSessionTotalDuration,
  getSessionAggregatedState,
  isSessionLevelAssessment,
} from './utils/GroupingUtils';
import {
  TRACE_ID_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  TOKENS_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  STATE_COLUMN_ID,
} from './hooks/useTableColumns';

interface GroupedRowsProps {
  groupedTracesResult: GroupedTracesResult;
  selectedAssessmentInfos: AssessmentInfo[];
  rows: Row<EvalTraceComparisonEntry>[];
  isComparing: boolean;
  enableRowSelection?: boolean;
  selectedColumns: TracesTableColumn[];
}

/**
 * Represents a visible row in the grouped table - either a session header or a trace row
 */
type VisibleRow =
  | { type: 'sessionHeader'; sessionGroup: TraceSessionGroup }
  | { type: 'trace'; rowIndex: number; sessionId?: string; isLastInSession?: boolean };

/**
 * Aggregated data computed for a session row
 */
interface SessionAggregatedData {
  firstInput: Record<string, unknown> | null;
  lastOutput: Record<string, unknown> | null;
  totalTokens: number;
  totalDuration: number;
  aggregatedState: string;
}

export const GenAiTracesTableBodyGroupedRows: React.FC<GroupedRowsProps> = ({
  groupedTracesResult,
  selectedAssessmentInfos,
  rows,
  isComparing,
  enableRowSelection,
  selectedColumns,
}) => {
  const { theme } = useDesignSystemTheme();
  const [expandedSessions, setExpandedSessions] = useState<Set<string>>(new Set());

  // Get sorted columns based on the table's column order from the first row
  const sortedColumns = useMemo(() => {
    if (rows.length === 0) return selectedColumns;

    // Get the column order from the first row's visible cells
    const columnOrder = rows[0].getVisibleCells().map((cell) => cell.column.id);

    // Create a map of column ID to its index in the order
    const orderMap = new Map<string, number>();
    columnOrder.forEach((id, index) => {
      orderMap.set(id, index);
    });

    // Sort selectedColumns to match the table's column order
    return [...selectedColumns].sort((a, b) => {
      const indexA = orderMap.get(a.id) ?? Infinity;
      const indexB = orderMap.get(b.id) ?? Infinity;
      return indexA - indexB;
    });
  }, [rows, selectedColumns]);

  const toggleSession = useCallback((sessionId: string) => {
    setExpandedSessions((prev) => {
      const next = new Set(prev);
      if (next.has(sessionId)) {
        next.delete(sessionId);
      } else {
        next.add(sessionId);
      }
      return next;
    });
  }, []);

  // Build a mapping from trace ID to row index for quick lookup
  const traceIdToRowIndex = useMemo(() => {
    const map = new Map<string, number>();
    rows.forEach((row, index) => {
      const traceId =
        row.original.currentRunValue?.traceInfo?.client_request_id || row.original.currentRunValue?.traceInfo?.trace_id;
      if (traceId) {
        map.set(traceId, index);
      }
    });
    return map;
  }, [rows]);

  // Compute visible rows based on expanded state
  const visibleRows = useMemo((): VisibleRow[] => {
    const result: VisibleRow[] = [];

    for (const row of groupedTracesResult.rows) {
      if (row.type === 'sessionGroup') {
        const sessionGroup = row.data;
        // Add session header
        result.push({ type: 'sessionHeader', sessionGroup });

        // If expanded, add child trace rows
        if (expandedSessions.has(sessionGroup.sessionId)) {
          sessionGroup.traces.forEach((trace, index) => {
            const traceId = trace.client_request_id || trace.trace_id;
            const rowIndex = traceIdToRowIndex.get(traceId);
            if (rowIndex !== undefined) {
              result.push({
                type: 'trace',
                rowIndex,
                sessionId: sessionGroup.sessionId,
                isLastInSession: index === sessionGroup.traces.length - 1,
              });
            }
          });
        }
      } else {
        // Ungrouped trace
        const trace = row.data;
        const traceId = trace.client_request_id || trace.trace_id;
        const rowIndex = traceIdToRowIndex.get(traceId);
        if (rowIndex !== undefined) {
          result.push({ type: 'trace', rowIndex });
        }
      }
    }

    return result;
  }, [groupedTracesResult.rows, expandedSessions, traceIdToRowIndex]);

  const tableContainerRef = React.useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: visibleRows.length,
    estimateSize: (index) => {
      const row = visibleRows[index];
      return row.type === 'sessionHeader' ? 48 : 120;
    },
    getScrollElement: () => tableContainerRef.current,
    overscan: 10,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();

  return (
    <div
      ref={tableContainerRef}
      style={{
        height: `${rowVirtualizer.getTotalSize()}px`,
        position: 'relative',
        display: 'grid',
      }}
    >
      {virtualItems.map((virtualRow) => {
        const visibleRow = visibleRows[virtualRow.index];

        if (visibleRow.type === 'sessionHeader') {
          return (
            <div
              key={`session-${visibleRow.sessionGroup.sessionId}`}
              data-index={virtualRow.index}
              ref={rowVirtualizer.measureElement}
              style={{
                position: 'absolute',
                transform: `translate3d(0, ${virtualRow.start}px, 0)`,
                willChange: 'transform',
                width: '100%',
              }}
            >
              <SessionHeaderRow
                sessionGroup={visibleRow.sessionGroup}
                isExpanded={expandedSessions.has(visibleRow.sessionGroup.sessionId)}
                onToggle={() => toggleSession(visibleRow.sessionGroup.sessionId)}
                selectedAssessmentInfos={selectedAssessmentInfos}
                selectedColumns={sortedColumns}
                enableRowSelection={enableRowSelection}
                isComparing={isComparing}
              />
            </div>
          );
        }

        const row = rows[visibleRow.rowIndex];
        if (!row) return null;

        const exportableTrace = row.original.currentRunValue && !isComparing;

        return (
          <div
            key={virtualRow.key}
            data-index={virtualRow.index}
            ref={rowVirtualizer.measureElement}
            style={{
              position: 'absolute',
              transform: `translate3d(0, ${virtualRow.start}px, 0)`,
              willChange: 'transform',
              width: '100%',
              paddingLeft: visibleRow.sessionId ? theme.spacing.lg : 0,
              borderBottom: visibleRow.isLastInSession ? `2px solid ${theme.colors.borderDecorative}` : undefined,
            }}
          >
            <GenAiTracesTableBodyRow
              row={row}
              exportableTrace={exportableTrace}
              enableRowSelection={enableRowSelection}
              isSelected={enableRowSelection ? row.getIsSelected() : undefined}
              isComparing={isComparing}
              selectedColumns={selectedColumns}
            />
          </div>
        );
      })}
    </div>
  );
};

interface SessionHeaderRowProps {
  sessionGroup: TraceSessionGroup;
  isExpanded: boolean;
  onToggle: () => void;
  selectedAssessmentInfos: AssessmentInfo[];
  selectedColumns: TracesTableColumn[];
  enableRowSelection?: boolean;
  isComparing: boolean;
}

const SessionHeaderRow: React.FC<SessionHeaderRowProps> = ({
  sessionGroup,
  isExpanded,
  onToggle,
  selectedAssessmentInfos,
  selectedColumns,
  enableRowSelection,
  isComparing,
}) => {
  const { theme } = useDesignSystemTheme();

  // Compute aggregated values
  const aggregatedData = useMemo(
    (): SessionAggregatedData => ({
      firstInput: getSessionFirstInput(sessionGroup.evaluationResults),
      lastOutput: getSessionLastOutput(sessionGroup.evaluationResults),
      totalTokens: getSessionTotalTokens(sessionGroup.traces),
      totalDuration: getSessionTotalDuration(sessionGroup.traces),
      aggregatedState: getSessionAggregatedState(sessionGroup.traces),
    }),
    [sessionGroup.traces, sessionGroup.evaluationResults],
  );

  return (
    <TableRow
      css={{
        cursor: 'pointer',
        '&:hover': {
          backgroundColor: theme.colors.actionTertiaryBackgroundHover,
        },
      }}
      onClick={onToggle}
    >
      {/* Selection checkbox cell - match the trace row pattern */}
      {enableRowSelection && <TableRowSelectCell componentId="mlflow.traces.session-row-select" />}

      {/* Map through selectedColumns and render appropriate cell for each */}
      {selectedColumns.map((column) => (
        <SessionCell
          key={column.id}
          column={column}
          sessionGroup={sessionGroup}
          aggregatedData={aggregatedData}
          isExpanded={isExpanded}
          isComparing={isComparing}
        />
      ))}
    </TableRow>
  );
};

interface SessionCellProps {
  column: TracesTableColumn;
  sessionGroup: TraceSessionGroup;
  aggregatedData: SessionAggregatedData;
  isExpanded: boolean;
  isComparing: boolean;
}

const SessionCell: React.FC<SessionCellProps> = ({ column, sessionGroup, aggregatedData, isExpanded, isComparing }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const renderContent = () => {
    // Handle TRACE_ID column - show session ID with expand/collapse icon and session indicator
    if (column.id === TRACE_ID_COLUMN_ID) {
      return (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          {isExpanded ? (
            <ChevronDownIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
          ) : (
            <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
          )}
          <Tag
            componentId="mlflow.traces.session-id-tag"
            color="turquoise"
            css={{ width: 'fit-content', maxWidth: 200 }}
          >
            <span
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
              }}
            >
              <SpeechBubbleIcon css={{ fontSize: 12, flexShrink: 0 }} />
              <span
                css={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {sessionGroup.sessionId}
              </span>
            </span>
          </Tag>
          <span css={{ color: theme.colors.textSecondary, whiteSpace: 'nowrap' }}>
            <FormattedMessage
              defaultMessage="({count} {count, plural, one {trace} other {traces}})"
              description="Count of traces in a session"
              values={{ count: sessionGroup.traceCount }}
            />
          </span>
        </div>
      );
    }

    // Handle INPUT columns (Request) - show first input as JSON like trace rows
    if (column.type === TracesTableColumnType.INPUT) {
      // Show the entire firstInput object as JSON to match trace row format
      const displayValue = aggregatedData.firstInput !== null ? JSON.stringify(aggregatedData.firstInput) : '-';
      return (
        <div
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: theme.colors.textPrimary,
          }}
        >
          {displayValue}
        </div>
      );
    }

    // Handle RESPONSE column - show last output as JSON like trace rows
    if (column.id === RESPONSE_COLUMN_ID) {
      // Show the entire lastOutput object as JSON to match trace row format
      const displayValue = aggregatedData.lastOutput !== null ? JSON.stringify(aggregatedData.lastOutput) : '-';
      return (
        <div
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: theme.colors.textPrimary,
          }}
        >
          {displayValue}
        </div>
      );
    }

    // Handle TOKENS column - show total tokens with gray background like trace rows
    if (column.id === TOKENS_COLUMN_ID) {
      if (!aggregatedData.totalTokens) {
        return <span css={{ color: theme.colors.textSecondary }}>-</span>;
      }
      return (
        <Tag css={{ width: 'fit-content', maxWidth: '100%' }} componentId="mlflow.traces.session-tokens">
          <span
            css={{
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {aggregatedData.totalTokens}
          </span>
        </Tag>
      );
    }

    // Handle EXECUTION_DURATION column - show total duration
    if (column.id === EXECUTION_DURATION_COLUMN_ID) {
      const durationMs = aggregatedData.totalDuration;
      const durationStr = durationMs > 0 ? `${(durationMs / 1000).toFixed(3)}s` : '-';
      return <span css={{ color: theme.colors.textPrimary }}>{durationStr}</span>;
    }

    // Handle STATE column - show aggregated state with icon
    if (column.id === STATE_COLUMN_ID) {
      const state = aggregatedData.aggregatedState;

      // Get icon based on state
      const getStateIcon = () => {
        if (state === 'OK') {
          return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
        }
        if (state === 'ERROR') {
          return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
        }
        // PENDING or IN_PROGRESS
        return <ClockIcon css={{ color: theme.colors.textValidationWarning }} />;
      };

      // Get label based on state
      const getStateLabel = () => {
        if (state === 'OK') {
          return intl.formatMessage({
            defaultMessage: 'OK',
            description: 'Experiment page > traces table > status label > ok',
          });
        }
        if (state === 'ERROR') {
          return intl.formatMessage({
            defaultMessage: 'Error',
            description: 'Experiment page > traces table > status label > error',
          });
        }
        return intl.formatMessage({
          defaultMessage: 'In progress',
          description: 'Experiment page > traces table > status label > in progress',
        });
      };

      return (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          {getStateIcon()}
          {getStateLabel()}
        </div>
      );
    }

    // Handle ASSESSMENT columns - show session-level assessment with badge if available
    if (column.type === TracesTableColumnType.ASSESSMENT) {
      const assessmentName = column.assessmentInfo?.name || column.id;

      // Find session-level assessment from evaluation results (attached to first trace)
      let sessionLevelAssessment: RunEvaluationResultAssessment | undefined;
      for (const evalResult of sessionGroup.evaluationResults) {
        const assessments = evalResult.responseAssessmentsByName?.[assessmentName];
        if (assessments?.length) {
          const assessment = assessments[0];
          if (isSessionLevelAssessment(assessment)) {
            sessionLevelAssessment = assessment;
            break;
          }
        }
      }

      // If there's a session-level assessment, show it with the badge style
      if (sessionLevelAssessment && column.assessmentInfo) {
        return (
          <EvaluationsReviewAssessmentTag
            showRationaleInTooltip
            disableJudgeTypeIcon
            hideAssessmentName
            assessment={sessionLevelAssessment}
            assessmentInfo={column.assessmentInfo}
            type="value"
          />
        );
      }

      // Fall back to showing aggregate if no session-level assessment
      const aggregate = sessionGroup.aggregatedAssessments.get(assessmentName);
      if (!aggregate) {
        return <span css={{ color: theme.colors.textSecondary }}>-</span>;
      }

      return <SessionAssessmentCell aggregate={aggregate} />;
    }

    // Default: empty cell for other column types
    return <span css={{ color: theme.colors.textSecondary }}>-</span>;
  };

  return (
    <TableCell
      css={{
        flex: `1 1 var(--col-${escapeCssSpecialCharacters(column.id)}-size)`,
        overflow: 'hidden',
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
      }}
    >
      {renderContent()}
    </TableCell>
  );
};

interface SessionAssessmentCellProps {
  aggregate: SessionAssessmentAggregate;
}

const SessionAssessmentCell: React.FC<SessionAssessmentCellProps> = ({ aggregate }) => {
  const { theme } = useDesignSystemTheme();

  // For numeric assessments, show the average
  if (aggregate.numericAverage !== null) {
    return (
      <span
        css={{
          display: 'inline-flex',
          alignItems: 'center',
          padding: `2px ${theme.spacing.xs}px`,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: theme.colors.tagDefault,
          color: theme.colors.textPrimary,
          fontSize: theme.typography.fontSizeSm,
        }}
      >
        {aggregate.numericAverage.toFixed(2)}
      </span>
    );
  }

  // For pass/fail assessments, show a bar chart with ratio
  const { passCount, totalCount } = aggregate;
  const failCount = totalCount - passCount;
  const passRatio = totalCount > 0 ? passCount / totalCount : 0;
  const failRatio = totalCount > 0 ? failCount / totalCount : 0;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      {/* Stacked bar chart */}
      <div
        css={{
          display: 'flex',
          width: 60,
          height: 8,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        {passRatio > 0 && (
          <div
            css={{
              width: `${passRatio * 100}%`,
              backgroundColor: theme.colors.green300,
            }}
          />
        )}
        {failRatio > 0 && (
          <div
            css={{
              width: `${failRatio * 100}%`,
              backgroundColor: theme.colors.red300,
            }}
          />
        )}
      </div>
      {/* Ratio text */}
      <span
        css={{
          fontSize: theme.typography.fontSizeSm,
          color: theme.colors.textSecondary,
          whiteSpace: 'nowrap',
        }}
      >
        {passCount}/{totalCount}
      </span>
    </div>
  );
};
