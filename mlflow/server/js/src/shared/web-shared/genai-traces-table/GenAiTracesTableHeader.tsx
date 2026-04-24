import type { HeaderGroup } from '@tanstack/react-table';
import { flexRender } from '@tanstack/react-table';
import { isNil } from 'lodash';
import React, { useState } from 'react';

import {
  HoverCard,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  Tooltip,
  useDesignSystemTheme,
  ChevronDownIcon,
} from '@databricks/design-system';
import { defineMessage, useIntl, type MessageDescriptor } from '@databricks/i18n';

import { EvaluationsAssessmentHoverCard } from './components/EvaluationsAssessmentHoverCard';
import { AssessmentColumnSummary } from './components/charts/AssessmentColumnSummary';
import {
  createAssessmentColumnId,
  TRACE_ID_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
  SESSION_COLUMN_ID,
  USER_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  TOKENS_COLUMN_ID,
  EXECUTION_DURATION_COLUMN_ID,
  REQUEST_TIME_COLUMN_ID,
  STATE_COLUMN_ID,
  SOURCE_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  TAGS_COLUMN_ID,
  ISSUES_COLUMN_ID,
} from './hooks/useTableColumns';
import {
  TracesTableColumnGroup,
  type AssessmentAggregates,
  type AssessmentFilter,
  type AssessmentInfo,
  type AssessmentValueType,
  type EvalTraceComparisonEntry,
} from './types';
import { escapeCssSpecialCharacters } from './utils/DisplayUtils';
import { getDocsLink } from './utils/DocUtils';

const COLUMN_TOOLTIPS = {
  [TRACE_ID_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Unique identifier for the trace',
      description: 'Tooltip description for the Trace ID column in the traces table',
    }),
  },
  [INPUTS_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: "Inputs of the trace, derived from the root span's inputs",
      description: 'Tooltip description for the Inputs column in the traces table',
    }),
  },
  [RESPONSE_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: "Outputs of the trace, derived from the root span's outputs",
      description: 'Tooltip description for the Response column in the traces table',
    }),
  },
  [SESSION_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'A unique identifier for the session or conversation for grouping traces',
      description: 'Tooltip description for the Session column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/tracing/track-users-sessions/'),
  },
  [USER_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Application user who triggered the trace',
      description: 'Tooltip description for the User column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/tracing/track-users-sessions/'),
  },
  [TRACE_NAME_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: "Name of the trace, derived from the root span's name",
      description: 'Tooltip description for the Trace Name column in the traces table',
    }),
  },
  [TOKENS_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Aggregated input/output/total token usage across all spans in the trace',
      description: 'Tooltip description for the Tokens column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/tracing/token-usage-cost'),
  },
  [EXECUTION_DURATION_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Total trace duration',
      description: 'Tooltip description for the Execution Duration column in the traces table',
    }),
  },
  [REQUEST_TIME_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'When the trace was created',
      description: 'Tooltip description for the Request Time column in the traces table',
    }),
  },
  [STATE_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Trace status: OK, ERROR, or IN_PROGRESS',
      description: 'Tooltip description for the State column in the traces table',
    }),
  },
  [SOURCE_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Entry point or script that generated the trace',
      description: 'Tooltip description for the Source column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/tracing/track-environments-context'),
  },
  [LOGGED_MODEL_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Associated model version',
      description: 'Tooltip description for the Logged Model column in the traces table',
    }),
  },
  [LINKED_PROMPTS_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Linked prompt versions from Prompt Registry',
      description: 'Tooltip description for the Linked Prompts column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/prompt-registry/use-prompts-in-apps#linking-with-trace'),
  },
  [RUN_NAME_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Associated MLflow Run if the trace was generated within an MLflow run context',
      description: 'Tooltip description for the Run Name column in the traces table',
    }),
  },
  [TAGS_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'User-defined key-value pairs',
      description: 'Tooltip description for the Tags column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/tracing/attach-tags'),
  },
  [ISSUES_COLUMN_ID]: {
    description: defineMessage({
      defaultMessage: 'Issues detected on the trace by automatic issue detection',
      description: 'Tooltip description for the Issues column in the traces table',
    }),
    docsUrl: getDocsLink('/genai/eval-monitor/ai-insights/detect-issues'),
  },
  [`${TracesTableColumnGroup.ASSESSMENT}-group`]: {
    description: defineMessage({
      defaultMessage: 'Feedback scores logged to the trace',
      description: 'Tooltip description for the Assessment column group in the traces table',
    }),
    docsUrl: getDocsLink('/genai/eval-monitor'),
  },
  [`${TracesTableColumnGroup.EXPECTATION}-group`]: {
    description: defineMessage({
      defaultMessage: 'Ground truth values annotated to the trace',
      description: 'Tooltip description for the Expectation column group in the traces table',
    }),
    docsUrl: getDocsLink('/genai/assessments/expectations'),
  },
} satisfies Record<string, { description: MessageDescriptor; docsUrl?: string }>;

interface GenAiTracesTableHeaderProps {
  enableRowSelection?: boolean;
  enableGrouping?: boolean;
  selectedAssessmentInfos: AssessmentInfo[];
  assessmentNameToAggregates: Record<string, AssessmentAggregates>;
  assessmentFilters: AssessmentFilter[];
  toggleAssessmentFilter: (
    assessmentName: string,
    filterValue: AssessmentValueType,
    run: string,
    filterType?: AssessmentFilter['filterType'],
  ) => void;
  runDisplayName?: string;
  compareToRunUuid?: string;
  compareToRunDisplayName?: string;
  disableAssessmentTooltips?: boolean;
  collapsedHeader: boolean;
  setCollapsedHeader: (collapsed: boolean) => void;
  isComparing: boolean;
  headerGroups: HeaderGroup<EvalTraceComparisonEntry>[];
  allRowSelected: boolean;
  someRowSelected: boolean;
  toggleAllRowsSelectedHandler: () => (event: unknown) => void;
}

export const GenAiTracesTableHeader = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    enableRowSelection,
    enableGrouping,
    selectedAssessmentInfos,
    assessmentNameToAggregates,
    assessmentFilters,
    toggleAssessmentFilter,
    runDisplayName,
    compareToRunUuid,
    compareToRunDisplayName,
    disableAssessmentTooltips,
    collapsedHeader,
    setCollapsedHeader,
    isComparing,
    headerGroups,
    allRowSelected,
    someRowSelected,
    toggleAllRowsSelectedHandler,
  }: GenAiTracesTableHeaderProps) => {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const [isChevronHovered, setIsChevronHovered] = useState(false);

    // super hacky way to get the border to show between the header and the row
    const borderCss = enableGrouping
      ? {
          position: 'relative' as const,
          '&::before': {
            content: '""',
            position: 'absolute' as const,
            top: '32px',
            left: 0,
            right: 0,
            borderTop: `1px solid ${theme.colors.border}`,
          },
        }
      : {};

    return (
      <>
        {headerGroups.map((headerGroup, depth) => (
          <TableRow
            isHeader
            key={headerGroup.id}
            css={{
              position: 'sticky',
              top: depth * 40,
              zIndex: 100,
              // hack to hide the bottom border of the first row
              ...(enableGrouping && {
                '& > *': {
                  borderBottom: depth === 0 ? 'none' : undefined,
                },
              }),
              ...(depth === headerGroups.length - 1 && {
                borderBottom: `1px solid ${isChevronHovered ? theme.colors.blue500 : theme.colors.border}`,
                transition: 'border-color 0.2s',
                // Remove default cell borders in the last header row
                '& > *': {
                  borderBottom: 'none',
                },
              }),
            }}
          >
            {enableRowSelection && (
              <div css={selectedAssessmentInfos.length === 0 && depth === 1 ? {} : borderCss}>
                <TableRowSelectCell
                  componentId="mlflow.experiment-evaluation-monitoring.evals-logs-table-header-select-cell"
                  checked={allRowSelected}
                  indeterminate={someRowSelected}
                  onChange={toggleAllRowsSelectedHandler()}
                  checkboxLabel={intl.formatMessage({
                    defaultMessage: 'Select all',
                    description: 'Description for button to select all rows in table',
                  })}
                  noCheckbox={depth === 0}
                  isDisabled={isComparing}
                />
              </div>
            )}
            {headerGroup.headers.map((header) => {
              if (header.isPlaceholder) return null; // skip spacer cells

              const assessmentInfo = selectedAssessmentInfos.find(
                (info) => createAssessmentColumnId(info.name) === header.id,
              );

              const title = header.isPlaceholder ? null : (
                <div
                  css={{
                    display: 'inline-block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    wordBreak: 'normal',
                    overflowWrap: 'normal',
                  }}
                  title={String(flexRender(header.column.columnDef.header, header.getContext()))}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </div>
              );
              const tooltipInfo = COLUMN_TOOLTIPS[header.column.id as keyof typeof COLUMN_TOOLTIPS];
              const titleElement =
                assessmentInfo && !disableAssessmentTooltips ? (
                  <HoverCard
                    key={header.id}
                    content={
                      <>
                        <EvaluationsAssessmentHoverCard
                          assessmentInfo={assessmentInfo}
                          assessmentNameToAggregates={assessmentNameToAggregates}
                          allAssessmentFilters={assessmentFilters}
                          toggleAssessmentFilter={toggleAssessmentFilter}
                          runUuid={runDisplayName}
                          compareToRunUuid={compareToRunUuid ? compareToRunDisplayName : undefined}
                        />
                      </>
                    }
                    trigger={title}
                  />
                ) : !isNil(title) && tooltipInfo ? (
                  <Tooltip
                    componentId="mlflow.traces-table.column-header-tooltip"
                    content={
                      <span>
                        {intl.formatMessage(tooltipInfo.description)}.
                        {'docsUrl' in tooltipInfo && tooltipInfo.docsUrl && (
                          <>
                            {' '}
                            <a
                              href={tooltipInfo.docsUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              css={{ color: theme.colors.blue400 }}
                            >
                              {intl.formatMessage({
                                defaultMessage: 'Learn more',
                                description: 'Link text in column header tooltip that opens documentation in a new tab',
                              })}
                            </a>
                          </>
                        )}
                      </span>
                    }
                  >
                    {title}
                  </Tooltip>
                ) : !isNil(title) ? (
                  title
                ) : null;

              return (
                <TableHeader
                  key={header.column.id}
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluations_evaluationsoverview.tsx_576"
                  css={{
                    '> span:first-of-type': {
                      width: '100%',
                      height: '100%',
                      marginTop: 'auto',
                      marginBottom: 'auto',
                    },
                    ...(selectedAssessmentInfos.length === 0 && depth === 1 ? {} : borderCss),
                  }}
                  style={{
                    flex: `${header.colSpan} 1 var(--header-${escapeCssSpecialCharacters(header?.column.id)}-size)`,
                  }}
                  header={header}
                  column={header.column}
                >
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    {titleElement}
                    <div
                      css={{
                        borderTop:
                          !enableGrouping && selectedAssessmentInfos.length > 0
                            ? `1px solid ${theme.colors.border}`
                            : '',
                        width: '100%',
                      }}
                    >
                      {assessmentInfo && (
                        <AssessmentColumnSummary
                          theme={theme}
                          intl={intl}
                          assessmentInfo={assessmentInfo}
                          assessmentAggregates={assessmentNameToAggregates[assessmentInfo.name]}
                          allAssessmentFilters={assessmentFilters}
                          toggleAssessmentFilter={toggleAssessmentFilter}
                          currentRunDisplayName={runDisplayName}
                          compareToRunDisplayName={compareToRunUuid ? compareToRunDisplayName : undefined}
                          collapsedHeader={collapsedHeader}
                        />
                      )}
                    </div>
                  </div>
                </TableHeader>
              );
            })}
            {depth === headerGroups.length - 1 && (
              <div
                css={{
                  paddingTop: '0px !important',
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  bottom: 0,
                  zIndex: 101,
                  padding: 0,
                  pointerEvents: 'none',
                }}
              >
                {/* Mask the border under the chevron */}
                <div
                  css={{
                    position: 'absolute',
                    left: '50%',
                    top: '100%',
                    width: '24px',
                    height: '12px',
                    background: theme.colors.backgroundPrimary,
                    transform: 'translate(-50%, -50%)',
                    zIndex: 101,
                  }}
                />
                {/* Chevron circle, pointer events enabled */}
                <div
                  css={{
                    position: 'absolute',
                    left: '50%',
                    top: '100%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 102,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '22px',
                    height: '22px',
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                  }}
                  onClick={() => setCollapsedHeader(!collapsedHeader)}
                  onMouseEnter={() => setIsChevronHovered(true)}
                  onMouseLeave={() => setIsChevronHovered(false)}
                >
                  <div
                    css={{
                      width: '22px',
                      height: '22px',
                      borderRadius: '50%',
                      background: theme.colors.backgroundPrimary,
                      border: `2px solid ${theme.colors.border}`,
                      boxShadow: '0 2px 8px 0 rgba(0,0,0,0.10)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'border-color 0.2s, box-shadow 0.2s',
                      ':hover': {
                        borderColor: theme.colors.blue500,
                        boxShadow: '0 2px 12px 0 rgba(0,0,0,0.18)',
                      },
                      opacity: isChevronHovered ? 1 : 0.5,
                    }}
                  >
                    <ChevronDownIcon
                      css={{
                        transform: !collapsedHeader ? 'rotate(180deg)' : 'none',
                        transition: 'transform 0.2s, opacity 0.2s',
                        color: theme.colors.textSecondary,
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </TableRow>
        ))}
      </>
    );
  },
);
