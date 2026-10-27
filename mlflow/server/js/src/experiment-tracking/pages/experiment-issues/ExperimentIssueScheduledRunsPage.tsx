import { useCallback, useMemo } from 'react';

import {
  Button,
  DangerIcon,
  Empty,
  Spacer,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { TimeAgo } from '@databricks/web-shared/browse';
import type { MessageDescriptor } from 'react-intl';
import { FormattedMessage, defineMessage, useIntl } from 'react-intl';

import { PageNotFoundView } from '../../../common/components/PageNotFoundView';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { Link, useParams } from '../../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import Routes from '../../routes';
import { formatScheduleInstant, formatScheduleWindowBound, weekdayLabels } from './issueDetectionSchedule';
import {
  SUMMARY_WINDOW_DAYS,
  type ScheduledRunOutcome,
  type ScheduledRunRow,
  scheduledRunDurationMs,
} from './scheduledRunHistory';
import { ExperimentIssuesScheduleControl } from './ExperimentIssuesScheduleControl';
import { useSetExperimentBreadcrumbTitle } from '../experiment-page-tabs/ExperimentPageRevampContext';
import { useIssueDetectionSchedule } from './useIssueDetectionSchedule';
import { useScheduledDetectionRuns } from './useScheduledDetectionRuns';

const COMPONENT_ID = 'mlflow.experiment-issues.scheduled-runs';

/** How close to the bottom of the list fetches the next page, matching the Evaluation runs tab. */
const OFFSET_FROM_BOTTOM_TO_FETCH_MORE = 200;

/** Column widths, flex-basis style, since the design system's table is a flex grid. */
const COLUMN_WIDTHS = {
  createdAt: 120,
  window: 215,
  outcome: 165,
  scannedTraces: 120,
  newIssues: 95,
  model: 185,
  cost: 85,
  duration: 95,
} as const;

const outcomeLabels = {
  running: defineMessage({
    defaultMessage: 'Running',
    description: 'Experiment issues tab > Scheduled runs > Outcome of an occurrence still going',
  }),
  succeeded: defineMessage({
    defaultMessage: 'Succeeded',
    description: 'Experiment issues tab > Scheduled runs > Outcome of an occurrence that analyzed traces',
  }),
  quiet: defineMessage({
    defaultMessage: 'Nothing to analyze',
    description:
      'Experiment issues tab > Scheduled runs > Outcome of an occurrence that fired over a window holding no traces it could use',
  }),
  failed: defineMessage({
    defaultMessage: 'Failed',
    description: 'Experiment issues tab > Scheduled runs > Outcome of an occurrence that broke',
  }),
  cancelled: defineMessage({
    defaultMessage: 'Cancelled',
    description: 'Experiment issues tab > Scheduled runs > Outcome of an occurrence somebody stopped',
  }),
} satisfies Record<ScheduledRunOutcome, MessageDescriptor>;

const OutcomeCell = ({ outcome }: { outcome: ScheduledRunOutcome }) => {
  const { theme } = useDesignSystemTheme();
  const color = {
    running: theme.colors.textValidationInfo,
    succeeded: theme.colors.textValidationSuccess,
    quiet: theme.colors.textValidationWarning,
    failed: theme.colors.textValidationDanger,
    cancelled: theme.colors.textSecondary,
  }[outcome];
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, color, whiteSpace: 'nowrap' }}>
      <span css={{ width: 7, height: 7, borderRadius: '50%', background: 'currentColor', flex: 'none' }} />
      <Typography.Text css={{ color: `${color} !important` }} bold size="sm">
        <FormattedMessage {...outcomeLabels[outcome]} />
      </Typography.Text>
    </span>
  );
};

/** `3m 12s`, or `4s` under a minute: a scheduled occurrence is minutes long, never hours. */
const formatDuration = (ms: number): string => {
  const seconds = Math.round(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${String(seconds % 60).padStart(2, '0')}s` : `${seconds}s`;
};

const formatCost = (usd: number): string => `$${usd.toFixed(2)}`;

const Dash = () => {
  const { theme } = useDesignSystemTheme();
  return <span css={{ color: theme.colors.textPlaceholder }}>&mdash;</span>;
};

/**
 * The stretch of trace history an occurrence covered.
 *
 * Both bounds, because neither alone says it: the end is the fire time rather than the moment the
 * run began, and the start is wherever the previous occurrence left its watermark -- which after a
 * quiet or failed occurrence can be much further back than one cadence interval.
 */
const WindowCell = ({ row, timezone, locale }: { row: ScheduledRunRow; timezone?: string; locale: string }) => {
  if (row.windowStartMs === undefined || row.windowEndMs === undefined || !timezone) {
    // An occurrence that ran before the bounds were recorded, or a schedule whose zone could not
    // be read: the rest of the row still stands.
    return <Dash />;
  }
  const start = formatScheduleWindowBound(row.windowStartMs, timezone, locale);
  const end = formatScheduleWindowBound(row.windowEndMs, timezone, locale);
  if (start === undefined || end === undefined) {
    return <Dash />;
  }
  return (
    <Typography.Text size="sm" css={{ whiteSpace: 'nowrap' }}>
      {start} &rarr; {end}
    </Typography.Text>
  );
};

const Figure = ({ label, children }: { label: React.ReactNode; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ minWidth: 0 }}>
      <Typography.Text size="sm" color="secondary" bold css={{ display: 'block', whiteSpace: 'nowrap' }}>
        {label}
      </Typography.Text>
      <div css={{ marginTop: theme.spacing.xs }}>{children}</div>
    </div>
  );
};

const ScheduledRunsContent = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const schedule = useIssueDetectionSchedule({ experimentId });
  // Names this route's breadcrumb leaf, the way every other sub-page does -- eval run details
  // publishes its run name here. The page title below is the same string, as it is there.
  useSetExperimentBreadcrumbTitle(
    intl.formatMessage({
      defaultMessage: 'Scheduled detection runs',
      description: "Experiment issues tab > Breadcrumb leaf for a schedule's past runs",
    }),
  );
  const { rows, newIssueCounts, summary, hasNextPage, fetchNextPage, isLoading, isFetching, error, refetch } =
    useScheduledDetectionRuns(experimentId);

  const nowMs = Date.now();
  // What the schedule does, in the words the dialog uses for the same thing, so the page and the
  // dialog cannot describe one schedule two ways.
  const scheduleSummary = useMemo(() => {
    const at = { time: schedule.startTime, zone: schedule.timezone ?? '' };
    if (schedule.setting === 'off') {
      return intl.formatMessage({
        defaultMessage: 'Schedule is off',
        description: 'Experiment issues tab > Scheduled runs > Summary of a paused schedule',
      });
    }
    if (schedule.setting === 'weekly') {
      return intl.formatMessage(
        {
          defaultMessage: 'Weekly on {weekday} at {time} ({zone})',
          description: 'Experiment issues tab > Scheduled runs > Summary of a weekly schedule',
        },
        { ...at, weekday: weekdayLabels(intl.locale)[schedule.weekday] },
      );
    }
    if (schedule.setting === 'daily') {
      return intl.formatMessage(
        {
          defaultMessage: 'Daily at {time} ({zone})',
          description: 'Experiment issues tab > Scheduled runs > Summary of a daily schedule',
        },
        at,
      );
    }
    // A schedule stored as hourly before the dialog stopped offering it. The dispatcher still
    // runs it, so the page still has to be able to describe it.
    return intl.formatMessage({
      defaultMessage: 'Hourly',
      description: 'Experiment issues tab > Scheduled runs > Summary of an hourly schedule',
    });
  }, [schedule.setting, schedule.startTime, schedule.timezone, schedule.weekday, intl]);

  const nextRun = useMemo(
    () =>
      // A paused schedule can still carry a next fire time in its dispatch state, written before
      // it was turned off. The dispatcher will not honour it, so naming it here -- beside a line
      // that says the schedule is off -- would promise a run that is never coming.
      schedule.setting !== 'off' && schedule.nextRunMs !== undefined && schedule.timezone
        ? formatScheduleInstant(schedule.nextRunMs, schedule.timezone, intl.locale)
        : undefined,
    [schedule.setting, schedule.nextRunMs, schedule.timezone, intl.locale],
  );

  const onScroll = useCallback(
    (event: React.UIEvent<HTMLDivElement>) => {
      const { scrollHeight, scrollTop, clientHeight } = event.currentTarget;
      if (scrollHeight - scrollTop - clientHeight < OFFSET_FROM_BOTTOM_TO_FETCH_MORE && !isFetching && hasNextPage) {
        fetchNextPage();
      }
    },
    [fetchNextPage, isFetching, hasNextPage],
  );

  const columns: { id: keyof typeof COLUMN_WIDTHS; label: React.ReactNode }[] = [
    {
      id: 'createdAt',
      label: (
        <FormattedMessage
          defaultMessage="Created at"
          description="Experiment issues tab > Scheduled runs > Column header for when an occurrence started"
        />
      ),
    },
    {
      id: 'window',
      label: (
        <FormattedMessage
          defaultMessage="Window covered"
          description="Experiment issues tab > Scheduled runs > Column header for the trace window an occurrence covered"
        />
      ),
    },
    {
      id: 'outcome',
      label: (
        <FormattedMessage
          defaultMessage="Outcome"
          description="Experiment issues tab > Scheduled runs > Column header for how an occurrence ended"
        />
      ),
    },
    {
      id: 'scannedTraces',
      label: (
        <FormattedMessage
          defaultMessage="Scanned traces"
          description="Experiment issues tab > Scheduled runs > Column header for how many traces an occurrence analyzed"
        />
      ),
    },
    {
      id: 'newIssues',
      label: (
        <FormattedMessage
          defaultMessage="New issues"
          description="Experiment issues tab > Scheduled runs > Column header for issues an occurrence filed for the first time"
        />
      ),
    },
    {
      id: 'model',
      label: (
        <FormattedMessage
          defaultMessage="Model"
          description="Experiment issues tab > Scheduled runs > Column header for the model an occurrence ran against"
        />
      ),
    },
    {
      id: 'cost',
      label: (
        <FormattedMessage
          defaultMessage="Cost"
          description="Experiment issues tab > Scheduled runs > Column header for what an occurrence spent"
        />
      ),
    },
    {
      id: 'duration',
      label: (
        <FormattedMessage
          defaultMessage="Duration"
          description="Experiment issues tab > Scheduled runs > Column header for how long an occurrence took"
        />
      ),
    },
  ];

  const FILLER_CELL_STYLE = { flexGrow: 1, flexShrink: 0, flexBasis: 0, minWidth: 0, padding: 0 } as const;

  const cellStyle = (id: keyof typeof COLUMN_WIDTHS) => ({
    flexGrow: 0,
    flexShrink: 0,
    flexBasis: COLUMN_WIDTHS[id],
  });

  // While pages remain, every figure below counts only what has loaded, so each is a lower bound
  // rather than the 30-day total its label promises. Marked, not silently approximated.
  const atLeast = (value: string) => (hasNextPage ? `${value}+` : value);

  if (isLoading) {
    return <TableSkeleton lines={5} />;
  }

  // A failed read is not an empty history: left to the empty state, a transient outage would tell
  // someone their schedule has never fired. The schedule's own read is checked the same way,
  // because its fallback reads as "Schedule is off".
  if (error || schedule.loadError) {
    return (
      <Empty
        image={<DangerIcon />}
        title={
          <FormattedMessage
            defaultMessage="Could not load this schedule's history"
            description="Experiment issues tab > Scheduled runs > Title when the run history could not be read"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="The occurrences this schedule has recorded could not be read. Nothing has been lost."
            description="Experiment issues tab > Scheduled runs > Body when the run history could not be read"
          />
        }
        button={
          <Button
            componentId={`${COMPONENT_ID}.retry`}
            onClick={() => {
              void refetch();
              schedule.reload();
            }}
          >
            <FormattedMessage
              defaultMessage="Retry"
              description="Experiment issues tab > Scheduled runs > Action retrying a failed read"
            />
          </Button>
        }
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div css={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: theme.spacing.sm }}>
        <div css={{ minWidth: 0 }}>
          <Typography.Title level={2} withoutMargins>
            <FormattedMessage
              defaultMessage="Scheduled detection runs"
              description="Experiment issues tab > Scheduled runs > Page title"
            />
          </Typography.Title>
          {scheduleSummary && (
            <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
              {scheduleSummary}
            </Typography.Text>
          )}
        </div>
        {/* The same dialog the Issues tab opens, rendered here so editing a schedule does not
            mean leaving the page that shows what it has done. */}
        <ExperimentIssuesScheduleControl experimentId={experimentId} triggerVariant="editButton" />
      </div>
      <Spacer size="sm" />
      <div
        css={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.lg,
          alignItems: 'flex-start',
          // Each figure shares the row's width so the strip spans the page rather than leaving
          // its right half empty, but never narrows past its own label: a basis small enough to
          // stretch from also let the longest label break across two lines and push the figure
          // after it onto a second row. Whole figures wrap instead, when there is no room.
          '& > div': { flex: '1 1 auto', minWidth: 'fit-content' },
        }}
      >
        <Figure
          label={
            <FormattedMessage
              defaultMessage="Next run"
              description="Experiment issues tab > Scheduled runs > Figure for when the schedule next fires"
            />
          }
        >
          {/* A dash rather than "Schedule is off" again: the line under the title already says
              so, and repeating it here read as two separate statements of the same fact. */}
          {nextRun ? <Typography.Text>{nextRun}</Typography.Text> : <Dash />}
        </Figure>
        <Figure
          label={
            <FormattedMessage
              defaultMessage="Last outcome"
              description="Experiment issues tab > Scheduled runs > Figure for how the newest settled occurrence ended"
            />
          }
        >
          {summary.lastOutcome ? <OutcomeCell outcome={summary.lastOutcome} /> : <Dash />}
        </Figure>
        <Figure
          label={
            <FormattedMessage
              defaultMessage="Runs, last {days} days"
              description="Experiment issues tab > Scheduled runs > Figure counting recent occurrences"
              values={{ days: SUMMARY_WINDOW_DAYS }}
            />
          }
        >
          <Typography.Text>{atLeast(String(summary.runCount))}</Typography.Text>
        </Figure>
        <Figure
          label={
            // Named with its window like the figures either side of it: the total is scoped to
            // the same 30 days, and "Traces scanned" alone read as an all-time count.
            <FormattedMessage
              defaultMessage="Traces scanned, last {days} days"
              description="Experiment issues tab > Scheduled runs > Figure totalling traces analyzed over the summary window"
              values={{ days: SUMMARY_WINDOW_DAYS }}
            />
          }
        >
          <Typography.Text>{atLeast(String(summary.tracesScanned))}</Typography.Text>
        </Figure>
        <Figure
          label={
            <FormattedMessage
              defaultMessage="Spend, last {days} days"
              description="Experiment issues tab > Scheduled runs > Figure totalling recent model spend"
              values={{ days: SUMMARY_WINDOW_DAYS }}
            />
          }
        >
          <Typography.Text>{atLeast(formatCost(summary.spendUsd))}</Typography.Text>
        </Figure>
      </div>
      <Spacer size="md" />
      <Table
        scrollable
        onScroll={onScroll}
        css={{ flex: 1 }}
        empty={
          rows.length === 0 ? (
            <Empty
              title={
                schedule.setting === 'off' ? (
                  <FormattedMessage
                    defaultMessage="This schedule is off"
                    description="Experiment issues tab > Scheduled runs > Empty state title for a paused schedule"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="No occurrences yet"
                    description="Experiment issues tab > Scheduled runs > Empty state title before the first occurrence"
                  />
                )
              }
              description={
                schedule.setting === 'off' ? (
                  <FormattedMessage
                    defaultMessage="Past occurrences stay here. Turn the schedule back on to add to them."
                    description="Experiment issues tab > Scheduled runs > Empty state body for a paused schedule"
                  />
                ) : nextRun ? (
                  <FormattedMessage
                    defaultMessage="The first one runs {when}."
                    description="Experiment issues tab > Scheduled runs > Empty state body naming the first run"
                    values={{ when: nextRun }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Occurrences appear here once the schedule has fired."
                    description="Experiment issues tab > Scheduled runs > Empty state body before the first run"
                  />
                )
              }
            />
          ) : undefined
        }
      >
        <TableRow isHeader>
          {columns.map(({ id, label }) => (
            <TableHeader componentId={`${COMPONENT_ID}.header`} key={id} style={cellStyle(id)}>
              {label}
            </TableHeader>
          ))}
          <TableHeader componentId={`${COMPONENT_ID}.header.filler`} aria-hidden style={FILLER_CELL_STYLE} />
        </TableRow>
        {rows.flatMap((row) => [
          <TableRow key={row.runId}>
            <TableCell style={cellStyle('createdAt')} css={{ alignItems: 'center' }}>
              <TimeAgo date={new Date(row.createdAtMs)} />
            </TableCell>
            <TableCell style={cellStyle('window')} css={{ alignItems: 'center' }}>
              <WindowCell row={row} timezone={schedule.timezone} locale={intl.locale} />
            </TableCell>
            <TableCell style={cellStyle('outcome')} css={{ alignItems: 'center' }}>
              <OutcomeCell outcome={row.outcome} />
            </TableCell>
            <TableCell style={cellStyle('scannedTraces')} css={{ alignItems: 'center' }}>
              {row.scannedTraces === undefined ? <Dash /> : row.scannedTraces}
            </TableCell>
            <TableCell style={cellStyle('newIssues')} css={{ alignItems: 'center' }}>
              {/* An occurrence still going has not finished filing, so its count so far would
                  read as a result. A count that could not be read is blank for the same reason:
                  zero is the claim that it found nothing. */}
              {row.outcome === 'running' || newIssueCounts?.[row.runId] === undefined ? (
                <Dash />
              ) : (
                newIssueCounts[row.runId]
              )}
            </TableCell>
            <TableCell style={cellStyle('model')} css={{ alignItems: 'center' }}>
              {row.model ? (
                <Typography.Text size="sm" ellipsis title={row.model}>
                  {row.model}
                </Typography.Text>
              ) : (
                <Dash />
              )}
            </TableCell>
            <TableCell style={cellStyle('cost')} css={{ alignItems: 'center' }}>
              {row.costUsd === undefined ? <Dash /> : formatCost(row.costUsd)}
            </TableCell>
            <TableCell style={cellStyle('duration')} css={{ alignItems: 'center' }}>
              {formatDuration(scheduledRunDurationMs(row, nowMs))}
            </TableCell>
            <TableCell aria-hidden style={FILLER_CELL_STYLE} />
          </TableRow>,
          // Why it failed, on its own full-width row under the occurrence it belongs to. The
          // message is the provider's own wording, which is the part worth acting on, so it is
          // shown rather than summarized -- and only when the run recorded one, since occurrences
          // from before that was recorded have nothing to show.
          ...(row.outcome === 'failed' && row.error
            ? [
                <TableRow key={`${row.runId}-error`} css={{ height: 'auto' }}>
                  <TableCell
                    css={{
                      flex: 1,
                      alignItems: 'flex-start',
                      backgroundColor: theme.colors.backgroundValidationDanger,
                      paddingTop: theme.spacing.sm,
                      paddingBottom: theme.spacing.sm,
                    }}
                  >
                    <div css={{ minWidth: 0 }}>
                      <Typography.Text size="sm" bold color="error" css={{ display: 'block' }}>
                        <FormattedMessage
                          defaultMessage="Why it failed"
                          description="Experiment issues tab > Scheduled runs > Label above the reason an occurrence failed"
                        />
                      </Typography.Text>
                      <Typography.Text size="sm" css={{ wordBreak: 'break-word' }}>
                        {row.error}
                      </Typography.Text>
                    </div>
                  </TableCell>
                </TableRow>,
              ]
            : []),
        ])}
      </Table>
      {rows.length > 0 && (
        <Typography.Hint css={{ marginLeft: 'auto', paddingTop: theme.spacing.xs, whiteSpace: 'nowrap' }}>
          {hasNextPage ? (
            <FormattedMessage
              defaultMessage="{count}+ runs loaded"
              description="Experiment issues tab > Scheduled runs > Count of loaded occurrences while more pages remain"
              values={{ count: rows.length }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="{count, plural, one {# run} other {# runs}}"
              description="Experiment issues tab > Scheduled runs > Total count of occurrences"
              values={{ count: rows.length }}
            />
          )}
        </Typography.Hint>
      )}
    </div>
  );
};

const ExperimentIssueScheduledRunsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams<{ experimentId: string }>();
  const workspace = useActiveWorkspace();
  if (!shouldEnableExperimentIssuesTab()) return <PageNotFoundView />;
  if (!experimentId) return null;
  return (
    // The route's place in the hierarchy is carried by the shell's breadcrumb -- Experiments >
    // <experiment> > Issues > Scheduled detection runs -- rather than by a link of this page's own.
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, paddingTop: theme.spacing.sm }}>
      <ScheduledRunsContent key={`${workspace}:${experimentId}`} experimentId={experimentId} />
    </div>
  );
};

export default ExperimentIssueScheduledRunsPage;
