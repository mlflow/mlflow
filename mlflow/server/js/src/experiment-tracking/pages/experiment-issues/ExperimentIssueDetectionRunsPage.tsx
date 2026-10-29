import { useMemo } from 'react';

import {
  Button,
  CalendarClockIcon,
  CursorPagination,
  DangerIcon,
  Empty,
  Spacer,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  Tag,
  Tooltip,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { TimeAgo } from '@databricks/web-shared/browse';
import { FormattedMessage, useIntl } from 'react-intl';

import { PageNotFoundView } from '../../../common/components/PageNotFoundView';
import { shouldEnableExperimentIssuesTab } from '../../../common/utils/FeatureUtils';
import { useParams } from '../../../common/utils/RoutingUtils';
import { useActiveWorkspace } from '../../../workspaces/utils/WorkspaceUtils';
import { RunStatusIcon } from '../../components/RunStatusIcon';
import { formatScheduleInstant, formatScheduleWindowBound, weekdayLabels } from './issueDetectionSchedule';
import {
  type DetectionRunRow,
  type DetectionRunTrigger,
  detectionRunDurationMs,
  isDetectionRunSettled,
} from './detectionRunHistory';
import { ExperimentIssuesScheduleControl } from './ExperimentIssuesScheduleControl';
import { useSetExperimentBreadcrumbTitle } from '../experiment-page-tabs/ExperimentPageRevampContext';
import { useIssueDetectionSchedule } from './useIssueDetectionSchedule';
import { useDetectionRuns } from './useDetectionRuns';

const COMPONENT_ID = 'mlflow.experiment-issues.detection-runs';

/** Column widths, flex-basis style, since the design system's table is a flex grid. */
const COLUMN_WIDTHS = {
  createdAt: 120,
  status: 100,
  trigger: 105,
  window: 200,
  analyzedTraces: 130,
  newIssues: 95,
  model: 170,
  cost: 85,
  duration: 95,
} as const;

/** How wide the failure tooltip may grow, sized for the 500 characters a run tag can hold. */
const ERROR_TOOLTIP_MAX_WIDTH = 420;

/**
 * The run's status, shown the way the Evaluation runs table shows it.
 *
 * The same `RunStatusIcon` and the same five words, so a run listed here and the same run listed
 * there cannot read as two different things. `other` covers a status a newer server introduces,
 * which `RunStatusIcon` draws nothing for.
 *
 * A run that recorded why it failed carries that reason in a tooltip here, rather than in a row
 * of its own under the table: the message is the provider's own wording and can run to a few
 * hundred characters, which as a row pushed the next run down the page.
 */
const StatusCell = ({ status, error }: { status: string; error?: string }) => {
  const { theme } = useDesignSystemTheme();
  const label = (
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, whiteSpace: 'nowrap' }}>
      <RunStatusIcon status={status} />
      <Typography.Text size="sm">
        <FormattedMessage
          defaultMessage="{status, select, SCHEDULED {Scheduled} RUNNING {Running} FINISHED {Finished} FAILED {Failed} KILLED {Killed} other {Unknown}}"
          description="Experiment issues tab > Run history > Status of one detection run"
          values={{ status }}
        />
      </Typography.Text>
    </span>
  );
  if (!error) {
    return label;
  }
  return (
    <Tooltip
      componentId={`${COMPONENT_ID}.status.error`}
      maxWidth={ERROR_TOOLTIP_MAX_WIDTH}
      content={<span css={{ wordBreak: 'break-word' }}>{error}</span>}
    >
      {label}
    </Tooltip>
  );
};

/**
 * Whether the run was fired by the schedule or asked for by a person.
 *
 * The icon goes in the `Tag`'s own slot rather than among its children, which is how the traces
 * table builds the same pill: the slot sizes the icon to the label and spaces it, where a child
 * icon renders full size against the text.
 */
const TriggerCell = ({ trigger }: { trigger: DetectionRunTrigger }) => {
  const pillCss = { width: 'fit-content', maxWidth: '100%' } as const;
  return trigger === 'schedule' ? (
    <Tag componentId={`${COMPONENT_ID}.trigger.schedule`} color="indigo" icon={<CalendarClockIcon />} css={pillCss}>
      <FormattedMessage
        defaultMessage="Scheduled"
        description="Experiment issues tab > Run history > Trigger pill for a run the schedule fired"
      />
    </Tag>
  ) : (
    <Tag componentId={`${COMPONENT_ID}.trigger.adhoc`} icon={<UserIcon />} css={pillCss}>
      <FormattedMessage
        defaultMessage="Ad hoc"
        description="Experiment issues tab > Run history > Trigger pill for a run somebody started by hand"
      />
    </Tag>
  );
};

/**
 * `3m 12s`, or `4s` under a minute, or `15h 02m` over an hour.
 *
 * A detection run takes minutes, so the hours are not for the runs that work: they are for a run
 * whose worker died, which stays `RUNNING` and is therefore timed against now. Without them an
 * abandoned run from yesterday reads as `915m 04s`.
 */
const formatDuration = (ms: number): string => {
  const seconds = Math.round(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  if (hours > 0) {
    return `${hours}h ${String(minutes % 60).padStart(2, '0')}m`;
  }
  return minutes > 0 ? `${minutes}m ${String(seconds % 60).padStart(2, '0')}s` : `${seconds}s`;
};

const formatCost = (usd: number): string => `$${usd.toFixed(2)}`;

const Dash = () => {
  const { theme } = useDesignSystemTheme();
  return <span css={{ color: theme.colors.textPlaceholder }}>&mdash;</span>;
};

/**
 * The stretch of trace history a scheduled run covered.
 *
 * Both bounds, because neither alone says it: the end is the fire time rather than the moment the
 * run began, and the start is wherever the previous occurrence left its watermark -- which after a
 * quiet or failed occurrence can be much further back than one cadence interval.
 *
 * Blank on an ad-hoc run, and truthfully so: that run was handed a chosen set of traces, so there
 * is no window it covered.
 */
const WindowCell = ({ row, timezone, locale }: { row: DetectionRunRow; timezone?: string; locale: string }) => {
  if (row.windowStartMs === undefined || row.windowEndMs === undefined || !timezone) {
    // An ad-hoc run, an occurrence that ran before the bounds were recorded, or a schedule whose
    // zone could not be read: the rest of the row still stands.
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

/**
 * What the page says instead of a table when the experiment has no detection runs at all.
 *
 * Centred on the page, the way the Issues tab's own empty state is, and shown on its own: a column
 * header over no rows labels nothing, so it is withheld until there is a row to describe. A
 * paused schedule is not a reason to be here -- the ad-hoc runs behind it would still be rows --
 * so the only thing the schedule changes is whether there is a first run to name.
 */
const EmptyHistory = ({ nextRun }: { nextRun?: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flex: 1,
        minHeight: 400,
        paddingBottom: theme.spacing.lg,
      }}
    >
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No detection runs yet"
            description="Experiment issues tab > Run history > Empty state title before the first detection run"
          />
        }
        description={
          nextRun ? (
            <FormattedMessage
              defaultMessage="Runs appear here once you detect issues from the Issues tab, or once the schedule fires. The schedule next runs {when}."
              description="Experiment issues tab > Run history > Empty state body naming the first scheduled run"
              values={{ when: nextRun }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="Runs appear here once you detect issues from the Issues tab, or once you turn on a schedule."
              description="Experiment issues tab > Run history > Empty state body with no schedule to name"
            />
          )
        }
      />
    </div>
  );
};

const DetectionRunsContent = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const schedule = useIssueDetectionSchedule({ experimentId });
  // Names this route's breadcrumb leaf, the way every other sub-page does -- eval run details
  // publishes its run name here. The page title below is the same string, as it is there.
  useSetExperimentBreadcrumbTitle(
    intl.formatMessage({
      defaultMessage: 'Issue detection run history',
      description: 'Experiment issues tab > Breadcrumb leaf for the detection runs an experiment has recorded',
    }),
  );
  const {
    rows,
    newIssueCounts,
    hasNextPage,
    hasPreviousPage,
    goToNextPage,
    goToPreviousPage,
    isLoading,
    isFetching,
    error,
    refetch,
  } = useDetectionRuns(experimentId);

  // Read once per render rather than per row, so every duration on screen is measured from the
  // same instant. An unfinished run is timed against it, so its duration sits still between
  // renders -- bounded by `RUNNING_POLL_INTERVAL_MS`, since polling is what renders the page
  // again. A ticking clock would buy a second-accurate figure for a run that takes minutes.
  const nowMs = Date.now();
  // What the schedule does, in the words the dialog uses for the same thing, so the page and the
  // dialog cannot describe one schedule two ways.
  const scheduleSummary = useMemo(() => {
    const at = { time: schedule.startTime, zone: schedule.timezone ?? '' };
    if (schedule.setting === 'off') {
      return intl.formatMessage({
        defaultMessage: 'Schedule is off',
        description: 'Experiment issues tab > Run history > Summary of a paused schedule',
      });
    }
    if (schedule.setting === 'weekly') {
      return intl.formatMessage(
        {
          defaultMessage: 'Scheduled weekly on {weekday} at {time} ({zone})',
          description: 'Experiment issues tab > Run history > Summary of a weekly schedule',
        },
        { ...at, weekday: weekdayLabels(intl.locale)[schedule.weekday] },
      );
    }
    if (schedule.setting === 'daily') {
      return intl.formatMessage(
        {
          defaultMessage: 'Scheduled daily at {time} ({zone})',
          description: 'Experiment issues tab > Run history > Summary of a daily schedule',
        },
        at,
      );
    }
    // A schedule stored as hourly before the dialog stopped offering it. The dispatcher still
    // runs it, so the page still has to be able to describe it.
    return intl.formatMessage({
      defaultMessage: 'Scheduled hourly',
      description: 'Experiment issues tab > Run history > Summary of an hourly schedule',
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

  const columns: { id: keyof typeof COLUMN_WIDTHS; label: React.ReactNode }[] = [
    {
      id: 'createdAt',
      label: (
        <FormattedMessage
          defaultMessage="Created at"
          description="Experiment issues tab > Run history > Column header for when a run started"
        />
      ),
    },
    {
      id: 'status',
      label: (
        <FormattedMessage
          defaultMessage="Status"
          description="Experiment issues tab > Run history > Column header for how a run ended"
        />
      ),
    },
    {
      id: 'trigger',
      label: (
        <FormattedMessage
          defaultMessage="Trigger"
          description="Experiment issues tab > Run history > Column header for what started a run"
        />
      ),
    },
    {
      id: 'window',
      label: (
        <FormattedMessage
          defaultMessage="Window covered"
          description="Experiment issues tab > Run history > Column header for the trace window a scheduled run covered"
        />
      ),
    },
    {
      id: 'analyzedTraces',
      label: (
        <FormattedMessage
          defaultMessage="Analyzed traces"
          description="Experiment issues tab > Run history > Column header for how many traces a run analyzed"
        />
      ),
    },
    {
      id: 'newIssues',
      label: (
        <FormattedMessage
          defaultMessage="New issues"
          description="Experiment issues tab > Run history > Column header for issues a run filed for the first time"
        />
      ),
    },
    {
      id: 'model',
      label: (
        <FormattedMessage
          defaultMessage="Model"
          description="Experiment issues tab > Run history > Column header for the model a run ran against"
        />
      ),
    },
    {
      id: 'cost',
      label: (
        <FormattedMessage
          defaultMessage="Cost"
          description="Experiment issues tab > Run history > Column header for what a run spent"
        />
      ),
    },
    {
      id: 'duration',
      label: (
        <FormattedMessage
          defaultMessage="Duration"
          description="Experiment issues tab > Run history > Column header for how long a run took"
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

  const isEmpty = rows.length === 0;

  /**
   * Paged rather than scrolled, the way the experiment, prompt, scorer and dataset lists page: a
   * run search answers with a cursor and no total, so there is no count to put beside the rows
   * and nothing a page number could be worked out from.
   *
   * Shown beside the failure state as well as beside the table. A page that could not be read
   * leaves no rows, so the error takes the table over -- and without this the page the user was
   * reading a moment ago would have nothing left to click.
   */
  const pagination =
    hasNextPage || hasPreviousPage ? (
      <div css={{ display: 'flex', justifyContent: 'flex-end', paddingTop: theme.spacing.xs }}>
        <CursorPagination
          componentId={`${COMPONENT_ID}.pagination`}
          hasNextPage={hasNextPage && !isFetching}
          hasPreviousPage={hasPreviousPage && !isFetching}
          onNextPage={goToNextPage}
          onPreviousPage={goToPreviousPage}
        />
      </div>
    ) : null;

  if (isLoading) {
    return <TableSkeleton lines={5} />;
  }

  // A failed read is not an empty history: left to the empty state, a transient outage would tell
  // someone nothing has ever run here. The schedule's own read is checked the same way, because
  // its fallback reads as "Schedule is off".
  if ((error || schedule.loadError) && isEmpty) {
    return (
      <>
        <Empty
          image={<DangerIcon />}
          title={
            <FormattedMessage
              defaultMessage="Could not load this experiment's run history"
              description="Experiment issues tab > Run history > Title when the run history could not be read"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="The detection runs this experiment has recorded could not be read. Nothing has been lost."
              description="Experiment issues tab > Run history > Body when the run history could not be read"
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
                description="Experiment issues tab > Run history > Action retrying a failed read"
              />
            </Button>
          }
        />
        {pagination}
      </>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <div css={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: theme.spacing.sm }}>
        <div css={{ minWidth: 0 }}>
          <Typography.Title level={2} withoutMargins>
            <FormattedMessage
              defaultMessage="Issue detection run history"
              description="Experiment issues tab > Run history > Page title"
            />
          </Typography.Title>
          {/* What the schedule does and when it next fires, on one line. The next run is the only
              thing about this page's contents that the table itself cannot say, which is why it
              survived the figures strip that used to sit above the table. */}
          <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            {scheduleSummary}
            {nextRun && (
              <>
                {' · '}
                <FormattedMessage
                  defaultMessage="Next run {when}"
                  description="Experiment issues tab > Run history > When the schedule next fires, beside what it does"
                  values={{ when: <Typography.Text bold>{nextRun}</Typography.Text> }}
                />
              </>
            )}
          </Typography.Text>
        </div>
        {/* The same dialog the Issues tab opens, rendered here so editing a schedule does not
            mean leaving the page that shows what it has done. */}
        <ExperimentIssuesScheduleControl experimentId={experimentId} />
      </div>
      {isEmpty ? (
        <EmptyHistory nextRun={nextRun} />
      ) : (
        <>
          <Spacer size="md" />
          <Table scrollable css={{ flex: 1 }}>
            <TableRow isHeader>
              {columns.map(({ id, label }) => (
                <TableHeader componentId={`${COMPONENT_ID}.header`} key={id} style={cellStyle(id)}>
                  {label}
                </TableHeader>
              ))}
              <TableHeader componentId={`${COMPONENT_ID}.header.filler`} aria-hidden style={FILLER_CELL_STYLE} />
            </TableRow>
            {rows.map((row) => (
              <TableRow key={row.runId}>
                <TableCell style={cellStyle('createdAt')} css={{ alignItems: 'center' }}>
                  <TimeAgo date={new Date(row.createdAtMs)} />
                </TableCell>
                <TableCell style={cellStyle('status')} css={{ alignItems: 'center' }}>
                  <StatusCell status={row.status} error={row.status === 'FAILED' ? row.error : undefined} />
                </TableCell>
                <TableCell style={cellStyle('trigger')} css={{ alignItems: 'center' }}>
                  <TriggerCell trigger={row.trigger} />
                </TableCell>
                <TableCell style={cellStyle('window')} css={{ alignItems: 'center' }}>
                  <WindowCell row={row} timezone={schedule.timezone} locale={intl.locale} />
                </TableCell>
                <TableCell style={cellStyle('analyzedTraces')} css={{ alignItems: 'center' }}>
                  {/* The count the run recorded when it finished, which is what the model was
                    shown rather than what was handed in: a run still going, or one from before
                    that count was recorded, has none. */}
                  {row.analyzedTraces === undefined ? <Dash /> : row.analyzedTraces}
                </TableCell>
                <TableCell style={cellStyle('newIssues')} css={{ alignItems: 'center' }}>
                  {/* A run still going has not finished filing, so its count so far would read as
                    a result. A count that could not be read is blank for the same reason: zero is
                    the claim that it found nothing. */}
                  {!isDetectionRunSettled(row.status) || newIssueCounts?.[row.runId] === undefined ? (
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
                  {formatDuration(detectionRunDurationMs(row, nowMs))}
                </TableCell>
                <TableCell aria-hidden style={FILLER_CELL_STYLE} />
              </TableRow>
            ))}
          </Table>
          {pagination}
        </>
      )}
    </div>
  );
};

const ExperimentIssueDetectionRunsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams<{ experimentId: string }>();
  const workspace = useActiveWorkspace();
  if (!shouldEnableExperimentIssuesTab()) return <PageNotFoundView />;
  if (!experimentId) return null;
  return (
    // The route's place in the hierarchy is carried by the shell's breadcrumb -- Experiments >
    // <experiment> > Issues > Issue detection run history -- rather than by a link of this page's own.
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, paddingTop: theme.spacing.sm }}>
      <DetectionRunsContent key={`${workspace}:${experimentId}`} experimentId={experimentId} />
    </div>
  );
};

export default ExperimentIssueDetectionRunsPage;
