import { MlflowService } from '../../sdk/MlflowService';
import { MLFLOW_ISSUE_DETECTION_SCHEDULE_TAG, MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER_TAG } from '../../constants';

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

/** The cadences offered. Hourly fires on the hour, daily at 09:00, weekly on Monday at 09:00. */
export type IssueDetectionScheduleFrequency = 'hourly' | 'daily' | 'weekly';

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
  traceFilters: [],
  model: undefined,
  timezone: undefined,
});

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
  const timezone = parsed['timezone'];
  return {
    setting: !cadence || paused ? 'off' : cadence,
    cadence,
    traceFilters,
    model: parseScheduleModel(parsed['model']),
    timezone: typeof timezone === 'string' && timezone.trim().length > 0 ? timezone : undefined,
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
 */
const resolveTimezone = (stored?: string): string =>
  stored?.trim() || Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

export interface IssueDetectionScheduleUpdate {
  setting: IssueDetectionScheduleSetting;
  traceFilters: ScheduleTraceFilter[];
  model?: IssueDetectionScheduleModel;
}

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
  { setting, traceFilters, model }: IssueDetectionScheduleUpdate,
  current?: IssueDetectionSchedule,
): Promise<IssueDetectionSchedule> => {
  const normalized = normalizeTraceFilters(traceFilters);
  const cadence = setting === 'off' ? current?.cadence : setting;
  // Like the cadence, the model and timezone are kept while paused so turning the schedule back
  // on restores what it had.
  const savedModel = model ?? current?.model;
  const timezone = resolveTimezone(current?.timezone);
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
      timezone,
      paused: setting === 'off',
      ...(savedModel ? { model: savedModel } : {}),
    }),
  });
  return { setting, cadence, traceFilters: normalized, model: savedModel, timezone };
};
