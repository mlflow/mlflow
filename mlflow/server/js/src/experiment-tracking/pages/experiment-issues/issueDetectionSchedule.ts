import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { MlflowService } from '../../sdk/MlflowService';
import {
  MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
  MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
} from '../../constants';

/**
 * The experiment's recurring issue-detection schedule, as stored.
 *
 * Upstream reaches a managed scheduler service here. OSS has its own dispatcher reading the two
 * experiment tags below, so this module is the adapter between the control and those tags; the
 * types it exports are the ones the control is written against.
 */

/** Which trace namespace a filter clause matches on: request metadata or a tag. */
export type ScheduleTraceFilterField = 'metadata' | 'tag';

/**
 * The comparison a filter clause applies. Only equality is offered today; the operator is persisted
 * anyway so the stored schema is ready to carry other comparisons (e.g. `!=`) without a migration.
 */
export type ScheduleTraceFilterOperator = '=';

/**
 * One equality clause attached to an experiment's schedule, matching either request metadata
 * (`request_metadata.<key>`) or a tag (`tags.<key>`). A schedule may carry several (ANDed): a
 * scheduled run only analyzes traces matching every clause (`key` = `value`). Stored as a JSON
 * array in its own experiment tag so the clauses can be recovered to prefill the schedule modal.
 */
export interface ScheduleTraceFilter {
  field: ScheduleTraceFilterField;
  operator: ScheduleTraceFilterOperator;
  key: string;
  value: string;
}

/**
 * The model a scheduled run analyzes with: either an AI Gateway endpoint or a provider's model
 * directly. Stored with the cadence so the dispatcher knows what to run against; the credential is
 * not stored, and is resolved from the provider at each fire time.
 */
export interface IssueDetectionScheduleModel {
  provider?: string;
  model?: string;
  endpointName?: string;
  /**
   * Which stored credential a direct selection runs with. The dispatcher uses this and nothing
   * else, so a schedule never runs on a key its owner did not choose. Absent on an endpoint
   * selection, which carries its own credential, and on a schedule saved before it was recorded.
   */
  secretId?: string;
}

/**
 * The cadences offered. Hourly fires at the schedule's chosen minute past every hour; daily fires
 * at its chosen time of day, weekly on Monday at that time.
 */
export type IssueDetectionScheduleFrequency = 'hourly' | 'daily' | 'weekly';

/**
 * Time of day a schedule fires when it stores none, `HH:MM` in its own timezone. This is the hour
 * every schedule fired at before a start time could be chosen, so one saved before then reads back
 * firing exactly when it already did. Must stay in step with `DEFAULT_FIRE_TIME` on the server.
 */
export const DEFAULT_ISSUE_DETECTION_START_TIME = '09:00';

/**
 * Minutes between the start times offered.
 *
 * The server's dispatcher polls on a 15-minute grid, so a schedule set to 09:08 cannot fire at
 * 09:08 -- the earliest occurrence after it is 09:15. Offering quarter hours is what keeps what
 * this control promises and what the server does the same thing. A time already stored off the
 * grid is still shown and still fires (at the first poll after it) rather than being rounded
 * behind its owner's back.
 */
export const START_TIME_STEP_MINUTES = 15;

/**
 * Weekday a weekly schedule fires on when it stores none: Monday, numbered the way the server
 * numbers it (`datetime.weekday()`), which is also the numbering the tag carries.
 */
export const DEFAULT_ISSUE_DETECTION_WEEKDAY = 0;

/** The user-facing schedule setting: no schedule, or one of the cadences. */
export type IssueDetectionScheduleSetting = IssueDetectionScheduleFrequency | 'off';

/**
 * Schema version of the stored envelopes. The server reads only a version it knows, so this must
 * stay in step with the version it writes and expects.
 */
export const ISSUE_DETECTION_SCHEDULE_VERSION = 1;

const FREQUENCIES: IssueDetectionScheduleFrequency[] = ['hourly', 'daily', 'weekly'];

const isFrequency = (value: unknown): value is IssueDetectionScheduleFrequency =>
  FREQUENCIES.some((frequency) => frequency === value);

export interface IssueDetectionSchedule {
  /** A paused schedule reads back as 'off' while keeping its cadence and clauses. */
  setting: IssueDetectionScheduleSetting;
  /** The cadence the schedule keeps while paused, so turning it back on restores it. */
  cadence: IssueDetectionScheduleFrequency | undefined;
  /**
   * Time of day the cadence fires, `HH:MM` in `timezone`. Always a readable time: a schedule
   * storing none, or one storing a value this reader cannot use, reads back as the default.
   */
  startTime: string;
  /** Weekday a weekly cadence fires on, Monday as 0. Meaningless for the other cadences. */
  weekday: number;
  /**
   * When the server says this schedule next runs, or undefined when it has not said yet.
   *
   * Taken from the server's own state tag rather than worked out here: the arithmetic behind it
   * is DST-aware, and a second implementation in the browser would be a second answer. Undefined
   * until the dispatcher's next pass after a schedule is saved, and deliberately so -- a value
   * recorded against different timing would describe a schedule that no longer exists.
   */
  nextRunMs: number | undefined;
  traceFilters: ScheduleTraceFilter[];
  /** Undefined on a schedule saved before a model could be chosen: the server default applies. */
  model: IssueDetectionScheduleModel | undefined;
  /**
   * IANA zone the cadence's hour is interpreted in. Undefined only on a schedule with none
   * stored, where saving adopts the editor's browser zone.
   */
  timezone: string | undefined;
}

export const noIssueDetectionSchedule = (): IssueDetectionSchedule => ({
  setting: 'off',
  cadence: undefined,
  startTime: DEFAULT_ISSUE_DETECTION_START_TIME,
  weekday: DEFAULT_ISSUE_DETECTION_WEEKDAY,
  nextRunMs: undefined,
  traceFilters: [],
  model: undefined,
  timezone: undefined,
});

const pad = (value: number): string => String(value).padStart(2, '0');

/** `HH:MM`, the form both this module and the server store and compare. */
const formatStartTime = (hour: number, minute: number): string => `${pad(hour)}:${pad(minute)}`;

/** `00`-`23`, the hours the modal offers. */
export const startTimeHourOptions = (): string[] => Array.from({ length: 24 }, (_, hour) => pad(hour));

/**
 * The minutes the modal offers: `00`, `15`, `30`, `45`.
 *
 * Not every minute, because the dispatcher polls on this same grid -- a schedule set to 09:08
 * could not fire at 09:08, only at the 09:15 poll after it.
 */
export const startTimeMinuteOptions = (): string[] =>
  Array.from({ length: 60 / START_TIME_STEP_MINUTES }, (_, index) => pad(index * START_TIME_STEP_MINUTES));

/** `HH:MM` split for the two selects that edit it. */
export const splitStartTime = (startTime: string): { hour: string; minute: string } => {
  const [hour, minute] = startTime.split(':');
  return { hour, minute };
};

export const joinStartTime = (hour: string, minute: string): string => `${hour}:${minute}`;

// ASCII digits only, and the same hour/minute bounds the server applies. Both readers have to
// agree: a value only one of them accepted would have the modal showing an hour the dispatcher
// does not fire at.
const START_TIME_PATTERN = /^(\d{1,2}):(\d{1,2})$/;

/**
 * The stored weekday, or the default when it is missing or not a day.
 *
 * Mirrors the server's reader, including its refusal of anything but a whole 0-6.
 */
export const parseIssueDetectionWeekday = (raw: unknown): number =>
  typeof raw === 'number' && Number.isInteger(raw) && raw >= 0 && raw <= 6 ? raw : DEFAULT_ISSUE_DETECTION_WEEKDAY;

/**
 * The stored start time as `HH:MM`, or the default when it is missing or unreadable.
 *
 * Falling back costs the owner the hour they chose rather than every run, which is the same
 * trade-off the server makes for the same tag.
 */
export const parseIssueDetectionStartTime = (raw: unknown): string => {
  if (typeof raw !== 'string') {
    return DEFAULT_ISSUE_DETECTION_START_TIME;
  }
  const match = START_TIME_PATTERN.exec(raw.trim());
  if (!match) {
    return DEFAULT_ISSUE_DETECTION_START_TIME;
  }
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (hour > 23 || minute > 59) {
    return DEFAULT_ISSUE_DETECTION_START_TIME;
  }
  return formatStartTime(hour, minute);
};

interface ExperimentTag {
  key: string;
  value?: string | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseJson = (raw: string | null | undefined): unknown => {
  if (!raw) {
    return undefined;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
};

const parseTraceFilter = (entry: unknown): ScheduleTraceFilter | undefined => {
  if (!isRecord(entry)) {
    return undefined;
  }
  const { field, key, value, operator } = entry;
  if (typeof key !== 'string' || key.trim().length === 0) {
    return undefined;
  }
  // Only equality exists today; an operator this version cannot apply drops the clause rather
  // than being coerced to '=', which would select a set disjoint from the one it asked for.
  if (operator !== undefined && operator !== '=') {
    return undefined;
  }
  // Anything that is not 'tag' is request metadata, so a clause written before the field existed
  // still applies; a missing or non-text value reads as empty.
  return {
    field: field === 'tag' ? 'tag' : 'metadata',
    operator: '=',
    key: key.trim(),
    value: typeof value === 'string' ? value : '',
  };
};

const parseScheduleModel = (entry: unknown): IssueDetectionScheduleModel | undefined => {
  if (!isRecord(entry)) {
    return undefined;
  }
  const { provider, model, endpointName, secretId } = entry;
  if (typeof endpointName === 'string' && endpointName.length > 0) {
    return { endpointName };
  }
  if (typeof provider === 'string' && provider.length > 0 && typeof model === 'string' && model.length > 0) {
    return {
      provider,
      model,
      ...(typeof secretId === 'string' && secretId.trim().length > 0 ? { secretId } : {}),
    };
  }
  return undefined;
};

export const parseScheduleTraceFilters = (raw: string | null | undefined): ScheduleTraceFilter[] => {
  const parsed = parseJson(raw);
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed.flatMap((entry) => {
    const filter = parseTraceFilter(entry);
    return filter ? [filter] : [];
  });
};

/**
 * A fingerprint of everything that decides *when* a schedule fires, in the form the server writes
 * alongside its dispatch state. Must stay in step with `DetectionSchedule.timing` on the server:
 * a mismatch only hides the next-run line, never changes when anything fires.
 */
export const scheduleTiming = (
  cadence: IssueDetectionScheduleFrequency,
  startTime: string,
  weekday: number,
  timezone: string,
): string => `${cadence}|${startTime}|${weekday}|${timezone}`;

/**
 * When the server says this schedule next runs, if what it recorded still describes it.
 *
 * The state tag carries the timing it was written against. A schedule edited since then has a
 * next run the server has not computed yet, and showing the old one would name a time nothing
 * will happen at -- so this answers undefined until the next dispatcher pass catches up.
 */
const parseNextRun = (
  raw: string | null | undefined,
  schedule: {
    cadence: IssueDetectionScheduleFrequency | undefined;
    startTime: string;
    weekday: number;
    timezone: string | undefined;
  },
): number | undefined => {
  const parsed = parseJson(raw);
  if (!isRecord(parsed) || parsed['version'] !== ISSUE_DETECTION_SCHEDULE_VERSION) {
    return undefined;
  }
  const nextRun = parsed['nextFireTimeMs'];
  if (typeof nextRun !== 'number' || !Number.isFinite(nextRun) || !schedule.cadence || !schedule.timezone) {
    return undefined;
  }
  const timing = scheduleTiming(schedule.cadence, schedule.startTime, schedule.weekday, schedule.timezone);
  return parsed['timing'] === timing ? nextRun : undefined;
};

/**
 * Read the schedule from an experiment's tags.
 *
 * Only an envelope this reader recognizes is interpreted, so a tag written by a newer server (or
 * by hand) reads as "no schedule" rather than being shown as a cadence nobody chose.
 */
export const readIssueDetectionScheduleFromTags = (
  tags: readonly ExperimentTag[] | undefined | null,
): IssueDetectionSchedule => {
  const scheduleTag = tags?.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG)?.value;
  const traceFilters = parseScheduleTraceFilters(
    tags?.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG)?.value,
  );
  const parsed = parseJson(scheduleTag);
  if (!isRecord(parsed) || parsed['version'] !== ISSUE_DETECTION_SCHEDULE_VERSION) {
    return { ...noIssueDetectionSchedule(), traceFilters };
  }
  const cadence = isFrequency(parsed['cadence']) ? parsed['cadence'] : undefined;
  const paused = parsed['paused'] === true;
  const storedTimezone = parsed['timezone'];
  const schedule = {
    setting: (!cadence || paused ? 'off' : cadence) as IssueDetectionScheduleSetting,
    cadence,
    startTime: parseIssueDetectionStartTime(parsed['fireTime']),
    weekday: parseIssueDetectionWeekday(parsed['weekday']),
    traceFilters,
    model: parseScheduleModel(parsed['model']),
    timezone: typeof storedTimezone === 'string' && storedTimezone.trim().length > 0 ? storedTimezone : undefined,
  };
  return {
    ...schedule,
    nextRunMs: parseNextRun(
      tags?.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_SCHEDULE_STATE_TAG)?.value,
      schedule,
    ),
  };
};

export const loadIssueDetectionSchedule = async (experimentId: string): Promise<IssueDetectionSchedule> => {
  const { experiment } = await MlflowService.getExperiment({ experiment_id: experimentId });
  return readIssueDetectionScheduleFromTags(experiment?.tags);
};

/**
 * Characters the server drops a clause for. A clause is spliced into a trace-search expression
 * where a backtick or quote breaks it, and that grammar has no escape, so the server has no way to
 * apply such a clause. Rejecting the draft here is what stops a schedule being saved with a filter
 * that looks active in this modal and is silently ignored by every run.
 */
const UNSUPPORTED_FILTER_CHARACTERS = ['`', "'", '"', '\\'];

/**
 * Also dropped by the server: a control character or DEL makes a malformed clause, so
 * `_has_control_char` in `mlflow/genai/discovery/schedule.py` refuses it the same way, and this
 * applies the same bounds. Easy to arrive here by paste rather than by typing -- a tab or a
 * newline carried along with a copied value -- and invisible once in the field, which is exactly
 * why the modal has to name it.
 */
const isControlCharacter = (character: string): boolean => {
  const code = character.codePointAt(0) ?? 0;
  return code < 0x20 || code === 0x7f;
};

/**
 * How many clauses a schedule may carry. The server applies only this many and ignores the rest,
 * so offering a further one here would show a filter as active that no run ever applies. Must
 * stay in step with `MAX_TRACE_FILTERS` on the server.
 */
export const MAX_TRACE_FILTERS = 20;

export const unsupportedFilterCharacter = (filter: ScheduleTraceFilter): string | undefined => {
  const punctuation = UNSUPPORTED_FILTER_CHARACTERS.find(
    (character) => filter.key.includes(character) || filter.value.includes(character),
  );
  if (punctuation) {
    return punctuation;
  }
  // Spread rather than indexed, so a character outside the basic plane is one item and cannot be
  // split into surrogates that look like something else.
  const control = [...filter.key, ...filter.value].find(isControlCharacter);
  // Named by code point, because the character itself has nothing to show: "Remove the  character
  // from the filter" would name nothing at all.
  return control === undefined ? undefined : `U+${control.codePointAt(0)?.toString(16).toUpperCase().padStart(4, '0')}`;
};

/** Blank-key clauses are dropped and keys trimmed, so the stored model stays clean. */
export const normalizeTraceFilters = (filters: ScheduleTraceFilter[]): ScheduleTraceFilter[] =>
  filters
    .map((filter) => ({ field: filter.field, operator: filter.operator, key: filter.key.trim(), value: filter.value }))
    .filter((filter) => filter.key.length > 0);

/**
 * The zone a save stores: the one already on the schedule, or the editor's browser zone when it
 * has none. Preserving it is what stops an edit made elsewhere moving when the schedule fires --
 * pausing a Tokyo daily schedule from Los Angeles would otherwise resume it at 09:00 Pacific.
 * There is no timezone control in the modal, so a stored zone is only ever changed by the server.
 *
 * The modal now offers a timezone control, so a zone the owner picked is passed in and wins; this
 * is the fallback chain behind it. A schedule saved before the control existed keeps the zone it
 * was given, which is what stops an edit made elsewhere moving when it fires.
 */
export const resolveScheduleTimezone = (chosen?: string, stored?: string): string =>
  chosen?.trim() || stored?.trim() || Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

export interface IssueDetectionScheduleUpdate {
  setting: IssueDetectionScheduleSetting;
  traceFilters: ScheduleTraceFilter[];
  model?: IssueDetectionScheduleModel;
  /** Omitted keeps the stored start time, the way omitting the model keeps the stored model. */
  startTime?: string;
  /** Omitted keeps the stored weekday. */
  weekday?: number;
  /** Omitted keeps the stored timezone, or adopts the editor's browser zone if there is none. */
  timezone?: string;
}

/**
 * Ask the server to bring the schedule's dispatch state in line with what was just written, and
 * to say when it next runs.
 *
 * The two tag writes above go through the generic tag endpoint, so nothing on the server learns a
 * schedule was saved; without this the next run was computed only by the following dispatcher
 * pass, up to a quarter of an hour later, and the dialog had nothing to show for exactly as long
 * as someone was waiting to be told what they had just scheduled.
 *
 * Best-effort on purpose. A failure, an older server with no such route, or an authorization
 * refusal all leave the state as the dispatcher would have found it, and its next pass still
 * corrects it -- so this must never fail a save that has already persisted.
 */
const syncScheduleState = async (experimentId: string): Promise<number | undefined> => {
  try {
    const response = (await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/issues/schedule/state'), {
      method: 'POST',
      body: { experiment_id: experimentId },
    })) as { next_fire_time_ms?: number | null };
    return typeof response?.next_fire_time_ms === 'number' ? response.next_fire_time_ms : undefined;
  } catch {
    return undefined;
  }
};

/**
 * Persist the schedule.
 *
 * "Off" pauses rather than deletes: the cadence and the clauses stay stored, so turning it back on
 * restores what was chosen before. Turning it off with nothing stored writes a paused schedule with
 * no cadence, which the server reads as nothing to run.
 *
 * The clauses are written **before** the cadence, and the order is the safety property rather than
 * a detail: the cadence tag is what makes the dispatcher pick the schedule up, so writing it first
 * opened a window -- a failed second write, or a dispatcher poll landing between the two -- where a
 * run could analyze traces the clauses were meant to exclude. Written in this order, a narrowed
 * scope is in force before anything can run under it, and a failure before the cadence write
 * leaves the schedule as it was.
 *
 * The two writes are still not atomic, so a failure between them can leave the stored clauses
 * ahead of the cadence. That failure is surfaced to the user, both tags are rewritten on every
 * save, and the direction it can fail in is the safe one.
 */
export const saveIssueDetectionSchedule = async (
  experimentId: string,
  { setting, traceFilters, model, startTime, weekday, timezone }: IssueDetectionScheduleUpdate,
  current?: IssueDetectionSchedule,
): Promise<IssueDetectionSchedule> => {
  const normalized = normalizeTraceFilters(traceFilters);
  const cadence = setting === 'off' ? current?.cadence : setting;
  // Like the cadence, the model, start time, weekday and timezone are kept while paused so
  // turning the schedule back on restores what it had.
  const savedModel = model ?? current?.model;
  // Parsed on the way out as well as in, so a draft that somehow holds an unusable value stores
  // the default rather than one the dispatcher would read as the default anyway.
  const savedStartTime = parseIssueDetectionStartTime(startTime ?? current?.startTime);
  const savedWeekday = parseIssueDetectionWeekday(weekday ?? current?.weekday);
  const savedTimezone = resolveScheduleTimezone(timezone, current?.timezone);
  await MlflowService.setExperimentTag({
    experiment_id: experimentId,
    key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG,
    value: JSON.stringify(normalized),
  });
  await MlflowService.setExperimentTag({
    experiment_id: experimentId,
    key: MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG,
    value: JSON.stringify({
      version: ISSUE_DETECTION_SCHEDULE_VERSION,
      cadence: cadence ?? null,
      timezone: savedTimezone,
      paused: setting === 'off',
      fireTime: savedStartTime,
      weekday: savedWeekday,
      ...(savedModel ? { model: savedModel } : {}),
    }),
  });
  return {
    setting,
    cadence,
    startTime: savedStartTime,
    weekday: savedWeekday,
    // Answered by the server now that the tags are written, so the dialog can say when the
    // schedule runs next the moment it is saved. Undefined if that call did not get through,
    // which is the old behaviour: no line until the dispatcher's next pass.
    nextRunMs: await syncScheduleState(experimentId),
    traceFilters: normalized,
    model: savedModel,
    timezone: savedTimezone,
  };
};
