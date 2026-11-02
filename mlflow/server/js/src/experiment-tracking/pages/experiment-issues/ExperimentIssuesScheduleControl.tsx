import {
  Button,
  CheckCircleIcon,
  ClockIcon,
  CloseSmallIcon,
  FormUI,
  Input,
  Modal,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { Link, matchPath, useLocation } from '../../../common/utils/RoutingUtils';
import { generateRandomName } from '../../../common/utils/NameUtils';
import Routes, { RoutePaths } from '../../routes';
import { useApiKeyConfiguration } from '../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { useCreateSecret } from '../../../gateway/hooks/useCreateSecret';
import { useEndpointsQuery } from '../../../gateway/hooks/useEndpointsQuery';
import {
  ISSUE_DETECTION_PROVIDERS,
  IssueDetectionModelDropdown,
  type IssueDetectionModelSelection,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionModelDropdown';
import {
  DEFAULT_ISSUE_DETECTION_START_TIME,
  DEFAULT_ISSUE_DETECTION_WEEKDAY,
  MAX_TRACE_FILTERS,
  formatScheduleInstant,
  joinStartTime,
  resolveScheduleTimezone,
  splitStartTime,
  startTimeHourOptions,
  startTimeMinuteOptions,
  unsupportedFilterCharacter,
  weekdayLabels,
} from './issueDetectionSchedule';
import { allTimezones, recommendedTimezones, timezoneChoice } from './scheduleTimezones';
import { NavigableCombobox } from '../../../common/components/navigable-combobox/NavigableCombobox';
import type { ComboboxMenuItem, NavigableComboboxConfig } from '../../../common/components/navigable-combobox/types';
import type {
  IssueDetectionScheduleModel,
  ScheduleTraceFilter,
  ScheduleTraceFilterField,
} from './issueDetectionSchedule';
import {
  useIssueDetectionSchedule,
  useScheduledDetectionAvailable,
  type IssueDetectionScheduleSetting,
} from './useIssueDetectionSchedule';

const COMPONENT_ID = 'mlflow.experiment-issues.schedule';

// 'off' first so the select reads Off -> Weekly -> Daily -> Hourly (weekly/daily emphasized over hourly).
// Daily and weekly only. An hourly cadence is more detection than an experiment needs and
// multiplies the spend; the dispatcher still runs a schedule already stored as hourly, so
// dropping the option here does not strand one.
const SETTING_OPTIONS: IssueDetectionScheduleSetting[] = ['off', 'weekly', 'daily'];

const isScheduleSetting = (value: string): value is IssueDetectionScheduleSetting =>
  SETTING_OPTIONS.some((option) => option === value);

/** The modal's two steps: the schedule itself, and the API key the chosen provider needs. */
type ModalView = 'main' | 'apiKey';

/**
 * Weekday names in the browser's own language, Monday first -- which is the numbering the
 * schedule stores and the server reads (`datetime.weekday()`), not the locale's first day.
 * Localized through `Intl` rather than through seven translated messages, since every locale
 * already has these names.
 */
/**
 * Width of the dialog.
 *
 * Narrower than the design system's default, which left every row ending less than half way
 * across. What stops it going narrower still is the clause row below -- `Field | Key | Operator |
 * Value` plus its remove button -- which is the widest thing here and has to stay on one line.
 */
const MODAL_WIDTH = 580;

/** Widths of the schedule row's controls, sized to fit one line inside the dialog. */
const CADENCE_WIDTH = 104;
const WEEKDAY_WIDTH = 116;
const TIME_PART_WIDTH = 72;

/**
 * Width of the model picker, matching the one in the ad-hoc detection dialog: that modal is 520
 * wide with 32 of padding each side, and its picker takes half of what is left. Picked over
 * filling the row so the two dialogs present the same control at the same size.
 */
const MODEL_PICKER_WIDTH = 220;

// Widths of one clause row, which together with its gaps and remove button set the floor under
// `MODAL_WIDTH`. Trimmed from 140/160/56/160 when the dialog was narrowed; the key and value
// still hold a tag name and a value like `production` without scrolling.
const FILTER_FIELD_WIDTH = 130;
const FILTER_KEY_WIDTH = 140;
const FILTER_OPERATOR_WIDTH = 48;
const FILTER_VALUE_WIDTH = 140;

// The filterable namespaces (Field dropdown) and the only operator offered today. Kept as constants
// so the row mirrors the traces-tab filter shape (Field | Key | Operator | Value) while the operator
// stays locked to equals.
const METADATA_FIELD_VALUE: ScheduleTraceFilterField = 'metadata';
const TAG_FIELD_VALUE: ScheduleTraceFilterField = 'tag';
const EQUALS_OPERATOR = '=';

const isTraceFilterField = (value: string): value is ScheduleTraceFilterField =>
  value === METADATA_FIELD_VALUE || value === TAG_FIELD_VALUE;

interface TraceFilterRowProps {
  index: number;
  filter: ScheduleTraceFilter;
  disabled: boolean;
  onChange: (index: number, next: ScheduleTraceFilter) => void;
  onRemove: (index: number) => void;
}

/**
 * One Field | Key | Operator | Value row of the scheduled-run trace filter, mirroring the traces-tab
 * filter clause. Field selects the namespace (Metadata or Tag); the operator is locked to "=", so
 * only the field, key, and value are editable. Declared at module scope rather than nested, so it is not
 * redefined on every render of the modal.
 */
const TraceFilterRow = ({ index, filter, disabled, onChange, onRemove }: TraceFilterRowProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const fieldLabel = intl.formatMessage({
    defaultMessage: 'Field',
    description: 'Experiment issues tab > Schedule modal label for the trace filter field',
  });
  const keyLabel = intl.formatMessage({
    defaultMessage: 'Key',
    description: 'Experiment issues tab > Schedule modal label for the trace filter key',
  });
  const operatorLabel = intl.formatMessage({
    defaultMessage: 'Operator',
    description: 'Experiment issues tab > Schedule modal label for the trace filter operator',
  });
  const valueLabel = intl.formatMessage({
    defaultMessage: 'Value',
    description: 'Experiment issues tab > Schedule modal label for the trace filter value',
  });
  const metadataFieldOptionLabel = intl.formatMessage({
    defaultMessage: 'Metadata',
    description: 'Experiment issues tab > Schedule modal trace filter field option for request metadata',
  });
  const tagFieldOptionLabel = intl.formatMessage({
    defaultMessage: 'Tag',
    description: 'Experiment issues tab > Schedule modal trace filter field option for tags',
  });

  const fieldId = `${COMPONENT_ID}.modal.filter-field-${index}`;
  const keyId = `${COMPONENT_ID}.modal.filter-key-${index}`;
  const operatorId = `${COMPONENT_ID}.modal.filter-operator-${index}`;
  const valueId = `${COMPONENT_ID}.modal.filter-value-${index}`;
  // Lift the portaled option lists above the modal, which otherwise stacks over them.
  const contentProps = { style: { zIndex: theme.options.zIndexBase + 40 } };

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={fieldId}>{fieldLabel}</FormUI.Label>
        <SimpleSelect
          id={fieldId}
          componentId={`${COMPONENT_ID}.modal.filter-field`}
          aria-label={fieldLabel}
          width={FILTER_FIELD_WIDTH}
          value={filter.field}
          disabled={disabled}
          contentProps={contentProps}
          onChange={({ target }) => {
            if (isTraceFilterField(target.value)) {
              onChange(index, { ...filter, field: target.value });
            }
          }}
        >
          <SimpleSelectOption value={METADATA_FIELD_VALUE}>{metadataFieldOptionLabel}</SimpleSelectOption>
          <SimpleSelectOption value={TAG_FIELD_VALUE}>{tagFieldOptionLabel}</SimpleSelectOption>
        </SimpleSelect>
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={keyId}>{keyLabel}</FormUI.Label>
        <Input
          id={keyId}
          componentId={`${COMPONENT_ID}.modal.filter-key`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter key',
            description: 'Experiment issues tab > Schedule modal aria label for the trace filter key input',
          })}
          css={{ width: FILTER_KEY_WIDTH }}
          placeholder={intl.formatMessage({
            defaultMessage: 'e.g. environment',
            description: 'Experiment issues tab > Schedule modal placeholder for the metadata filter key',
          })}
          value={filter.key}
          disabled={disabled}
          onChange={(e) => onChange(index, { ...filter, key: e.target.value })}
        />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={operatorId}>{operatorLabel}</FormUI.Label>
        <Input
          id={operatorId}
          componentId={`${COMPONENT_ID}.modal.filter-operator`}
          aria-label={operatorLabel}
          css={{ width: FILTER_OPERATOR_WIDTH }}
          value={EQUALS_OPERATOR}
          // Only equality is supported for now, so the operator is a fixed, read-only "=".
          readOnly
          disabled
        />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={valueId}>{valueLabel}</FormUI.Label>
        <Input
          id={valueId}
          componentId={`${COMPONENT_ID}.modal.filter-value`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter value',
            description: 'Experiment issues tab > Schedule modal aria label for the trace filter value input',
          })}
          css={{ width: FILTER_VALUE_WIDTH }}
          placeholder={intl.formatMessage({
            defaultMessage: 'e.g. production',
            description: 'Experiment issues tab > Schedule modal placeholder for the metadata filter value',
          })}
          value={filter.value}
          disabled={disabled}
          onChange={(e) => onChange(index, { ...filter, value: e.target.value })}
        />
      </div>
      <Button
        componentId={`${COMPONENT_ID}.modal.filter-remove`}
        type="tertiary"
        icon={<CloseSmallIcon />}
        disabled={disabled}
        aria-label={intl.formatMessage({
          defaultMessage: 'Remove filter',
          description: 'Experiment issues tab > Schedule modal aria label for the remove-filter button',
        })}
        onClick={() => onRemove(index)}
      />
    </div>
  );
};

export interface ExperimentIssuesScheduleControlProps {
  experimentId: string;
}

const emptyFilterRow = (): ScheduleTraceFilter => ({
  field: METADATA_FIELD_VALUE,
  operator: EQUALS_OPERATOR,
  key: '',
  value: '',
});

/**
 * The picker's selection in the shape the schedule stores: an endpoint, or a provider's model
 * plus the stored credential it runs with.
 *
 * A direct selection names its secret, the same one the ad-hoc dialog sends as `secret_id`. The
 * dispatcher uses it and nothing else: resolving a key by provider at fire time would pick
 * whichever secret was saved most recently, which on a server with more than one user is a key
 * somebody else saved for their own use.
 */
const toStoredModel = (
  selection: IssueDetectionModelSelection,
  secretId: string | undefined,
): IssueDetectionScheduleModel =>
  selection.mode === 'endpoint' && selection.endpointName
    ? { endpointName: selection.endpointName }
    : { provider: selection.provider, model: selection.model, ...(secretId ? { secretId } : {}) };

/**
 * Which stored credential a save pins the schedule to.
 *
 * A key just created in this dialog wins. Otherwise the schedule keeps the secret it was already
 * pinned to, as long as the selection still names that provider: re-reading `existingSecrets[0]`
 * on an unrelated edit -- a cadence change, a pause -- would re-point the schedule at whichever
 * secret that provider happens to list first, which on a server with more than one user is a key
 * somebody else saved for their own use. That is the hazard #27003's credential fix closed on the
 * dispatcher side, and saving from here reopened it.
 *
 * Only a selection that moved to another provider falls back to that provider's first secret: the
 * stored pin cannot apply to it, and this dialog has no secret picker to ask with.
 */
const resolveSecretId = (
  selection: IssueDetectionModelSelection | null,
  stored: IssueDetectionScheduleModel | undefined,
  createdSecretId: string | undefined,
  firstSecretId: string | undefined,
): string | undefined => {
  if (createdSecretId) {
    return createdSecretId;
  }
  if (selection?.mode === 'direct' && stored?.provider === selection.provider && stored.secretId) {
    return stored.secretId;
  }
  return firstSecretId;
};

/** The stored selection put back into the picker's shape, for prefilling the modal. */
const toSelection = (model: IssueDetectionScheduleModel | undefined): IssueDetectionModelSelection | undefined => {
  const fallbackProvider = ISSUE_DETECTION_PROVIDERS[0];
  if (model?.endpointName) {
    return {
      mode: 'endpoint',
      endpointName: model.endpointName,
      provider: fallbackProvider.id,
      model: fallbackProvider.defaultModel,
    };
  }
  if (model?.provider && model.model) {
    return { mode: 'direct', provider: model.provider, model: model.model };
  }
  return undefined;
};

/**
 * Toolbar control that turns automatic (scheduled) issue detection on/off, picks its cadence, and
 * optionally scopes each run to traces matching one or more key=value clauses over request metadata
 * and/or tags. The trigger opens a modal so the cadence and the trace filters are chosen and saved
 * together; "Off" pauses the schedule. Manual detection is unaffected.
 */
export const ExperimentIssuesScheduleControl = ({ experimentId }: ExperimentIssuesScheduleControlProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // Whether the footer's "View run history" link would lead where the user already is. Read off
  // the route rather than taken as a prop, so the link stays right by itself wherever this dialog
  // is opened from -- including a surface that does not exist yet.
  const isOnRunHistoryPage = Boolean(matchPath(RoutePaths.experimentPageTabIssueDetectionRuns, useLocation().pathname));
  // Replaces upstream's experiment-ACL check: OSS has no per-experiment write permission for
  // tags, but it does have servers that cannot run a schedule at all.
  const scheduledDetectionAvailable = useScheduledDetectionAvailable();
  const {
    setting,
    traceFilters,
    model,
    startTime: storedStartTime,
    weekday: storedWeekday,
    timezone: storedTimezone,
    nextRunMs,
    isLoading,
    isUpdating,
    loadError,
    reload,
    setSetting,
  } = useIssueDetectionSchedule({ experimentId });

  const [open, setOpen] = useState(false);
  const [cadence, setCadence] = useState<IssueDetectionScheduleSetting>('off');
  // Time of day the cadence fires, `HH:MM` in the schedule's timezone.
  const [startTime, setStartTime] = useState(DEFAULT_ISSUE_DETECTION_START_TIME);
  // Weekday a weekly cadence fires on, Monday as 0.
  const [weekday, setWeekday] = useState(DEFAULT_ISSUE_DETECTION_WEEKDAY);
  // The zone the time above is read in. Seeded from the schedule, or from this browser when the
  // schedule has none, which is the zone a save would have adopted anyway.
  const [timezone, setTimezone] = useState(() => resolveScheduleTimezone());
  const [filters, setFilters] = useState<ScheduleTraceFilter[]>([]);
  // The model the scheduled runs will use. Null until the stored selection (or the default) is
  // resolved, which needs the endpoint list.
  const [selection, setSelection] = useState<IssueDetectionModelSelection | null>(null);
  // 'apiKey' is the step shown when the chosen provider has no credential to run with. Saving a
  // schedule is not a run, so unlike the ad-hoc dialog -- which learns this from the server
  // rejecting a submission -- this is checked before the schedule is stored.
  const [view, setView] = useState<ModalView>('main');
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  // The key this draft has already stored in AI Gateway, if any. Kept because the schedule's two
  // tag writes can fail after the secret is created, and the secret is named randomly, so a retry
  // that created another one would leave an unused credential behind on every attempt.
  const [createdSecret, setCreatedSecret] = useState<{
    provider: string;
    apiKey: string;
    secretId: string;
  } | null>(null);

  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const {
    hasExistingSecrets,
    existingSecrets,
    isLoadingSecrets,
    envCredentialsAvailable,
    isLoadingProviderConfig,
    secretsError,
  } = useApiKeyConfiguration({
    provider: selection?.mode === 'direct' ? selection.provider : ISSUE_DETECTION_PROVIDERS[0].id,
  });
  const {
    mutate: createSecret,
    isLoading: isCreatingSecret,
    error: createSecretError,
    reset: resetCreateSecret,
  } = useCreateSecret();
  // Set when a clause holds a character the server cannot apply, so Save explains instead of
  // storing a filter that every run would ignore.
  const [filterError, setFilterError] = useState<string | null>(null);
  // True after a Save whose writes failed. Local (not the hook's `error`, which also covers load
  // failures) so it only reflects the in-modal save attempt, and it keeps the modal open for retry.
  const [saveFailed, setSaveFailed] = useState(false);

  const optionLabels: Record<IssueDetectionScheduleSetting, string> = {
    off: intl.formatMessage({
      defaultMessage: 'Off',
      description: 'Experiment issues tab > Scheduled detection option that disables automatic runs',
    }),
    weekly: intl.formatMessage({
      defaultMessage: 'Weekly',
      description: 'Experiment issues tab > Scheduled detection cadence option that runs once a week',
    }),
    daily: intl.formatMessage({
      defaultMessage: 'Daily',
      description: 'Experiment issues tab > Scheduled detection cadence option that runs once a day',
    }),
    hourly: intl.formatMessage({
      defaultMessage: 'Hourly',
      description: 'Experiment issues tab > Scheduled detection cadence option that runs every hour',
    }),
  };

  const triggerLabel =
    setting === 'off'
      ? intl.formatMessage({
          defaultMessage: 'Schedule detection',
          description: 'Experiment issues tab > Button that opens the scheduled-detection modal',
        })
      : intl.formatMessage(
          {
            defaultMessage: 'Scheduled: {cadence}',
            description: 'Experiment issues tab > Button showing the active scheduled-detection cadence',
          },
          { cadence: optionLabels[setting] },
        );

  // A schedule that is on shows a green checkmark, a change of shape as well as of colour. It sits in
  // the label, not in `icon`: a default Button paints its leading icon with the button's own colour,
  // which would hide the green.
  const isScheduled = setting !== 'off';
  const triggerIcon = isScheduled ? undefined : <ClockIcon />;
  // While saving, the button's spinner stands in for the checkmark.
  const triggerContent =
    isScheduled && !isUpdating ? (
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <CheckCircleIcon color="success" data-testid="schedule-on-icon" />
        {triggerLabel}
      </span>
    ) : (
      triggerLabel
    );

  // Seed the modal from the current schedule + filters so it opens showing what is active. With no
  // filters we show no rows — just the "Add filter" button — so the user opts in explicitly.
  const openModal = useCallback(() => {
    // Read the schedule again on the way in. The draft below is seeded from what is already
    // loaded, so a reply that lands mid-edit cannot disturb it -- what it refreshes is the next
    // run, which a save leaves unknown until the dispatcher has worked the new one out.
    reload();
    setCadence(setting);
    setStartTime(storedStartTime);
    setWeekday(storedWeekday);
    setTimezone(resolveScheduleTimezone(undefined, storedTimezone));
    setFilters(traceFilters.map((filter) => ({ ...filter })));
    setSaveFailed(false);
    setFilterError(null);
    setSelection(toSelection(model) ?? null);
    setView('main');
    setApiKeyDraft('');
    setCreatedSecret(null);
    resetCreateSecret();
    setOpen(true);
  }, [reload, setting, storedStartTime, storedWeekday, storedTimezone, traceFilters, model, resetCreateSecret]);
  const closeModal = useCallback(() => setOpen(false), []);

  // A schedule that names no model (or one saved before a model could be chosen) falls back to the
  // same default the ad-hoc dialog uses: the first AI Gateway endpoint, else the first provider.
  // Done in an effect so a modal opened before the endpoint list arrives still lands on it.
  useEffect(() => {
    if (!open || selection || isLoadingEndpoints) {
      return;
    }
    const provider = ISSUE_DETECTION_PROVIDERS[0];
    setSelection(
      endpoints.length > 0
        ? {
            mode: 'endpoint',
            endpointName: endpoints[0].name,
            provider: provider.id,
            model: provider.defaultModel,
          }
        : { mode: 'direct', provider: provider.id, model: provider.defaultModel },
    );
  }, [open, selection, isLoadingEndpoints, endpoints]);

  const changeFilter = useCallback((index: number, next: ScheduleTraceFilter) => {
    setFilterError(null);
    setFilters((prev) => prev.map((filter, i) => (i === index ? next : filter)));
  }, []);

  // Removing the last row collapses the section back to just the "Add filter" button.
  const removeFilter = useCallback((index: number) => setFilters((prev) => prev.filter((_, i) => i !== index)), []);

  const addFilter = useCallback(
    () => setFilters((prev) => (prev.length >= MAX_TRACE_FILTERS ? prev : [...prev, emptyFilterRow()])),
    [],
  );

  // Every zone the browser knows, built once the modal is opened rather than on every render of
  // the Issues tab: the list is a few hundred entries and each one is formatted to find its
  // current offset.
  const standardTimezones = useMemo(() => (open ? allTimezones() : []), [open]);
  const recommendedTimezonesLabel = intl.formatMessage({
    defaultMessage: 'Recommended',
    description: 'Experiment issues tab > Schedule modal heading above the suggested timezones',
  });
  const standardTimezonesLabel = intl.formatMessage({
    defaultMessage: 'Standard',
    description: 'Experiment issues tab > Schedule modal heading above the full timezone list',
  });
  // Built once per open and per chosen zone, not on every render: the config maps the whole
  // zone list, and the combobox re-renders on every keystroke in its search box. A stable
  // identity also lets `NavigableCombobox` keep its own memoized views.
  const suggestedTimezones = useMemo(() => recommendedTimezones(timezone), [timezone]);
  const timezoneConfig = useMemo<NavigableComboboxConfig<string>>(() => {
    const suggested = new Set(suggestedTimezones.map(({ zone }) => zone));
    const toItem = ({ zone, label }: { zone: string; label: string }): ComboboxMenuItem<string> => ({
      type: 'item',
      key: zone,
      label,
      value: zone,
    });
    return {
      initialViewId: 'main',
      views: [
        {
          id: 'main',
          items: [
            {
              type: 'section',
              key: 'recommended',
              label: recommendedTimezonesLabel,
            },
            ...suggestedTimezones.map(toItem),
            {
              type: 'section',
              key: 'standard',
              label: standardTimezonesLabel,
            },
            ...standardTimezones.filter(({ zone }) => !suggested.has(zone)).map(toItem),
          ],
        },
      ],
    };
  }, [suggestedTimezones, standardTimezones, recommendedTimezonesLabel, standardTimezonesLabel]);

  // What the saved schedule will do next, which is the question the dialog cannot otherwise
  // answer: the controls say "daily at 16:00", not "that means tomorrow".
  //
  // "as saved" is doing real work in that wording. This is the stored schedule's next run and it
  // does not follow the draft: only the server computes these, and it has not seen an unsaved
  // edit. Labelled plainly "Next run" it reads as a claim about the controls above it, so a draft
  // of "Weekly at 07:45" would sit beside a Tuesday at 12:00 and look like the dialog's answer
  // for what was just typed.
  //
  // Formatted in the zone the schedule stores, which is the zone the server read the instant in
  // -- not the draft's zone, which would show a different wall clock for the same instant while
  // still claiming to describe what is saved. `nextRunMs` is only ever read alongside a stored
  // zone, and this requires one rather than falling back to the browser's, which would name an
  // hour the server is not going to fire at.
  const nextRunLabel = useMemo(() => {
    if (nextRunMs === undefined || !storedTimezone) {
      return null;
    }
    const when = formatScheduleInstant(nextRunMs, storedTimezone, intl.locale);
    if (when === undefined) {
      // A zone this server can load but this browser cannot makes `Intl` throw. Unguarded that
      // took the whole Issues tab down rather than one line of it, since this runs whenever the
      // control renders and not only when the modal is open. The schedule stays editable and its
      // stored zone untouched; only the line is lost.
      return null;
    }
    return intl.formatMessage(
      {
        defaultMessage: 'Next run as saved: {when}',
        description:
          'Experiment issues tab > Schedule modal line stating when the schedule as currently saved next runs; it does not follow unsaved edits',
      },
      { when },
    );
  }, [nextRunMs, storedTimezone, intl]);

  const persistSchedule = useCallback(
    async (secretId?: string) => {
      // Blank-key rows are dropped; the hook/persistence also normalize, but trimming here keeps the
      // saved model clean.
      const cleaned = filters
        .map((filter) => ({
          field: filter.field,
          operator: filter.operator,
          key: filter.key.trim(),
          value: filter.value,
        }))
        .filter((filter) => filter.key.length > 0);
      // Both checks run before either tag is written: the server drops what they catch, so storing
      // it would leave the modal showing a filter that no run applies.
      if (cleaned.length > MAX_TRACE_FILTERS) {
        setFilterError(
          intl.formatMessage(
            {
              defaultMessage:
                'Remove {excess, plural, one {# filter} other {# filters}}. Scheduled runs apply at most {limit}.',
              description:
                'Experiment issues tab > Schedule modal error shown when a schedule carries more trace filters than the server applies',
            },
            { excess: cleaned.length - MAX_TRACE_FILTERS, limit: MAX_TRACE_FILTERS },
          ),
        );
        return;
      }
      const rejected = cleaned.map(unsupportedFilterCharacter).find(Boolean);
      if (rejected) {
        setFilterError(
          intl.formatMessage(
            {
              defaultMessage: 'Remove the {character} character from the filter. Scheduled runs cannot match on it.',
              description:
                'Experiment issues tab > Schedule modal error shown when a trace filter contains a character the server cannot apply',
            },
            { character: rejected },
          ),
        );
        return;
      }
      setFilterError(null);
      setSaveFailed(false);
      try {
        // Await the writes and only close on success. On failure keep the modal open with the draft
        // intact so the user can retry without re-entering the cadence/filters.
        await setSetting({
          setting: cadence,
          startTime,
          weekday,
          timezone,
          traceFilters: cleaned,
          model: selection
            ? toStoredModel(selection, resolveSecretId(selection, model, secretId, existingSecrets[0]?.secret_id))
            : undefined,
        });
        setOpen(false);
      } catch {
        setSaveFailed(true);
      }
    },
    [cadence, startTime, weekday, timezone, filters, selection, model, existingSecrets, setSetting, intl],
  );

  // Prompting for a key before these answer would ask for one the server may already have.
  const isCheckingCredentials = isLoadingSecrets || isLoadingProviderConfig;
  // Only act on a conclusive answer. A server that does not report its environment credentials, or
  // one whose gateway could not be reached, leaves this unknown -- and a schedule that could be
  // saved before this step existed must not become unsavable because a capability check failed.
  //
  // This deliberately treats "the secrets list could not be read" the same as "this server has no
  // environment credential": both mean nobody can say whether a usable key exists, so the step is
  // skipped rather than demanding a key that may already be stored. The cost is a schedule saved
  // without a key, whose first occurrence then fails with a credential error that names the
  // problem -- visible, and recoverable by saving a key -- rather than a prompt that cannot be
  // dismissed on a server that was working a moment ago.
  const credentialStateKnown = envCredentialsAvailable !== undefined && !secretsError;
  // Whether the chosen model can run at all. An AI Gateway endpoint carries its own credentials;
  // a provider's model needs either a key saved in AI Gateway or one in the server's environment.
  // Only asked of a schedule that will actually fire: turning one off needs no credential.
  const needsApiKey =
    cadence !== 'off' &&
    selection?.mode === 'direct' &&
    credentialStateKnown &&
    !hasExistingSecrets &&
    !envCredentialsAvailable;

  const handleSave = useCallback(() => {
    if (needsApiKey) {
      setView('apiKey');
      return;
    }
    return persistSchedule();
  }, [needsApiKey, persistSchedule]);

  // The key is saved in AI Gateway, not in the schedule: the dispatcher resolves a credential for
  // the provider at each fire time, so rotating the key later keeps the schedule running.
  const handleContinueAndSave = useCallback(() => {
    const trimmedKey = apiKeyDraft.trim();
    if (!selection || !trimmedKey) {
      return;
    }
    // Retrying after the secret was stored and the schedule write failed reuses that secret. Keyed
    // on the provider and the key itself, so editing either still stores the new one.
    if (createdSecret && createdSecret.provider === selection.provider && createdSecret.apiKey === trimmedKey) {
      // Not returned: every other path through this callback yields nothing, and returning a
      // promise from one of them alone makes it a function with inconsistent returns.
      void persistSchedule(createdSecret.secretId);
      return;
    }
    createSecret(
      {
        secret_name: generateRandomName(selection.provider),
        secret_value: { api_key: trimmedKey },
        provider: selection.provider,
      },
      // The schedule stores the secret just created, not whichever one a later read happens to
      // find first.
      {
        onSuccess: (response) => {
          setCreatedSecret({
            provider: selection.provider,
            apiKey: trimmedKey,
            secretId: response.secret.secret_id,
          });
          return persistSchedule(response.secret.secret_id);
        },
      },
    );
  }, [apiKeyDraft, selection, createdSecret, createSecret, persistSchedule]);

  // A schedule only runs on a server with the job system and its periodic tasks enabled, behind a
  // SQL backend store. Rather than hide the control, keep it visible but disabled with a tooltip
  // explaining why -- the affordance stays discoverable and any cadence already saved is still
  // readable. Offering a working-looking switch that silently never fires would be worse.
  // A failed read leaves the values above as fallbacks rather than what is stored, so the modal
  // must not open: saving the apparent "Off, no filters" state would replace a real schedule with
  // it. The reason and a retry sit on the trigger, since the modal that would otherwise carry them
  // is exactly what is being withheld.
  if (loadError) {
    const readFailed = intl.formatMessage({
      defaultMessage: "Couldn't read this experiment's detection schedule. Retry to edit it.",
      description:
        'Experiment issues tab > Tooltip explaining the scheduled-detection control is disabled because its schedule could not be read',
    });
    return (
      <Tooltip componentId={`${COMPONENT_ID}.trigger.load-failed-tooltip`} content={readFailed}>
        <Button componentId={`${COMPONENT_ID}.trigger.retry-load`} icon={<ClockIcon />} onClick={reload}>
          {intl.formatMessage({
            defaultMessage: 'Retry schedule',
            description:
              'Experiment issues tab > Button that retries reading the detection schedule after a failed read',
          })}
        </Button>
      </Tooltip>
    );
  }

  if (!scheduledDetectionAvailable) {
    // The server reports only whether it can run a schedule, not which requirement is missing, so
    // the reason has to cover every case `scheduled_detection_available()` answers no for. The
    // earlier wording named the backend store, job execution and periodic tasks, which reads as
    // wrong on a server that has all three and is turned down for its job execution engine.
    const disabledReason = intl.formatMessage({
      defaultMessage:
        'This MLflow server is not configured to run scheduled detection. It needs a SQL backend store, job execution and periodic tasks enabled, and a job execution engine that can run scheduled detection.',
      description:
        'Experiment issues tab > Tooltip explaining the scheduled-detection control is disabled because the server cannot run a schedule',
    });
    return (
      <Tooltip componentId={`${COMPONENT_ID}.trigger.disabled-tooltip`} content={disabledReason}>
        <Button componentId={`${COMPONENT_ID}.trigger`} icon={triggerIcon} disabled>
          {triggerContent}
        </Button>
      </Tooltip>
    );
  }

  const cadenceRowLabel = intl.formatMessage({
    defaultMessage: 'Cadence',
    description: 'Experiment issues tab > Schedule modal label for the run cadence',
  });
  const weekdayFieldLabel = intl.formatMessage({
    defaultMessage: 'Day',
    description: 'Experiment issues tab > Schedule modal label for the weekday a weekly run starts on',
  });
  const hourFieldLabel = intl.formatMessage({
    defaultMessage: 'Hour',
    description: 'Experiment issues tab > Schedule modal label for the hour a run starts at',
  });
  const minuteFieldLabel = intl.formatMessage({
    defaultMessage: 'Minute',
    description: 'Experiment issues tab > Schedule modal label for the minute a run starts at',
  });
  const timezoneRowLabel = intl.formatMessage({
    defaultMessage: 'Timezone',
    description: 'Experiment issues tab > Schedule modal label for the timezone the start time is read in',
  });
  const modelRowLabel = intl.formatMessage({
    defaultMessage: 'Model',
    description: 'Experiment issues tab > Schedule modal label for the model scheduled runs analyze with',
  });
  const filterSectionLabel = intl.formatMessage({
    defaultMessage: 'Additional trace filters (optional)',
    description: 'Experiment issues tab > Schedule modal label for the optional trace filters',
  });
  const filtersDisabled = cadence === 'off';
  const { hour, minute } = splitStartTime(startTime);
  // The quarter hours the dispatcher can fire at, plus the stored minute when it is not one of
  // them: a schedule set by hand to 09:08 is shown and saved as 09:08 rather than being moved to
  // a neighbouring quarter hour by having been looked at. Sorts correctly as text, zero-padded.
  const minuteOptions = startTimeMinuteOptions().includes(minute)
    ? startTimeMinuteOptions()
    : [...startTimeMinuteOptions(), minute].sort();
  const weekdayOptions = weekdayLabels(intl.locale);
  // An active schedule has to name the model it will run on. `selection` is null until the
  // endpoint list resolves, and a save made in that gap stores no model at all -- the run would
  // then use the server's default rather than what this dialog was about to show, and the
  // credential step would be skipped because it only applies to a direct provider selection.
  const awaitingSelection = cadence !== 'off' && !selection;
  const selectedProviderName =
    ISSUE_DETECTION_PROVIDERS.find((provider) => provider.id === selection?.provider)?.name ?? selection?.provider;

  return (
    <>
      <Button
        componentId={`${COMPONENT_ID}.trigger`}
        icon={triggerIcon}
        loading={isUpdating}
        // Disabled while the schedule is loading, so the modal never opens against unknown state.
        disabled={isLoading || isUpdating}
        onClick={openModal}
      >
        {triggerContent}
      </Button>
      <Modal
        componentId={`${COMPONENT_ID}.modal`}
        visible={open}
        onCancel={isUpdating || isCreatingSecret ? undefined : closeModal}
        footer={
          <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            {/* Where upstream puts "View run history". Upstream's opens the managed scheduler's
                own history; this one opens every detection run this experiment has recorded, the
                scheduled ones among them. Hidden on the API-key step, which is mid-save, and on
                the history page itself, where it would lead nowhere. */}
            {view === 'main' && !isOnRunHistoryPage && (
              <Link
                componentId={`${COMPONENT_ID}.modal.footer.run-history`}
                to={Routes.getIssueDetectionRunsRoute(experimentId)}
              >
                {intl.formatMessage({
                  defaultMessage: 'View run history',
                  description: "Experiment issues tab > Schedule modal link to this experiment's detection run history",
                })}
              </Link>
            )}
            <div css={{ marginLeft: 'auto' }}>
              {view === 'apiKey' ? (
                <>
                  <Button
                    componentId={`${COMPONENT_ID}.modal.footer.api-key-back`}
                    onClick={() => {
                      resetCreateSecret();
                      setView('main');
                    }}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Back',
                      description: 'Experiment issues tab > Schedule modal button that leaves the API key step',
                    })}
                  </Button>
                  <Button
                    componentId={`${COMPONENT_ID}.modal.footer.api-key-continue`}
                    type="primary"
                    loading={isCreatingSecret || isUpdating}
                    // `loading` hides the label but still accepts a click, and each click mints a
                    // fresh secret name, so a double-click would store two keys for one schedule.
                    disabled={!apiKeyDraft.trim() || isCreatingSecret || isUpdating}
                    onClick={handleContinueAndSave}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Continue and save',
                      description:
                        'Experiment issues tab > Schedule modal button that saves the API key and then the schedule',
                    })}
                  </Button>
                </>
              ) : (
                <>
                  <Button componentId={`${COMPONENT_ID}.modal.footer.cancel`} onClick={closeModal}>
                    {intl.formatMessage({
                      defaultMessage: 'Cancel',
                      description: 'Experiment issues tab > Schedule modal cancel button',
                    })}
                  </Button>
                  <Button
                    componentId={`${COMPONENT_ID}.modal.footer.save`}
                    type="primary"
                    loading={isUpdating}
                    // Also held while the credential check is in flight, so Save cannot ask for a
                    // key before the answer that says whether one is needed has arrived.
                    disabled={awaitingSelection || isCheckingCredentials}
                    onClick={handleSave}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Save',
                      description: 'Experiment issues tab > Schedule modal confirm button',
                    })}
                  </Button>
                </>
              )}
            </div>
          </div>
        }
        title={intl.formatMessage({
          defaultMessage: 'Schedule issue detection',
          description: 'Experiment issues tab > Schedule modal title',
        })}
        size="normal"
        css={{ width: `${MODAL_WIDTH}px !important` }}
        // Left at the design system's default rather than raised: the timezone combobox portals
        // its menu at `zIndexBase + 10`, with no way to lift it, so a modal above that height
        // renders over its own dropdown. Everything this dialog opens for itself is placed at
        // `+ 40` or beyond, so it still stacks over the modal from here.
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {view === 'main' && (
            <>
              {/* Above the controls, because it is the one line here that describes the schedule
                  as it stands rather than the draft being edited. Shown only while both the
                  saved schedule and the draft are running: a paused schedule has a next fire
                  time recorded that it will never reach, and a draft turned off is about to stop
                  the one that would. */}
              {setting !== 'off' && cadence !== 'off' && nextRunLabel && (
                <Typography.Hint>{nextRunLabel}</Typography.Hint>
              )}
              {/* The cadence row and the timezone under it share one column sized to the row's
                  own content, so the timezone control ends where the minute select ends instead
                  of running the width of the dialog. Measured from the row rather than set to a
                  number: the row's width depends on the weekday being there and on how long
                  "on" and "at" are in the current language, so a constant would be wrong in
                  some languages and on every cadence change. */}
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.md,
                  width: 'fit-content',
                }}
              >
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text color="secondary">{cadenceRowLabel}</Typography.Text>
                  {/* One row: the cadence, the weekday a weekly one falls on, and the time of day.
                    Wraps rather than overflowing, since a long translation of "on" or "at" can
                    push the minute select past the modal's width. */}
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: theme.spacing.sm,
                      flexWrap: 'wrap',
                    }}
                  >
                    <SimpleSelect
                      id={`${COMPONENT_ID}.modal.cadence`}
                      componentId={`${COMPONENT_ID}.modal.cadence`}
                      label={cadenceRowLabel}
                      width={CADENCE_WIDTH}
                      value={cadence}
                      onChange={({ target }) => {
                        if (isScheduleSetting(target.value)) {
                          setCadence(target.value);
                        }
                      }}
                      // The select's options popover defaults below the modal; lift it above the modal's zIndex.
                      contentProps={{ style: { zIndex: theme.options.zIndexBase + 40 } }}
                    >
                      {SETTING_OPTIONS.map((option) => (
                        <SimpleSelectOption key={option} value={option}>
                          {optionLabels[option]}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelect>
                    {cadence === 'weekly' && (
                      <>
                        <Typography.Text color="secondary">
                          {intl.formatMessage({
                            defaultMessage: 'on',
                            description:
                              'Experiment issues tab > Schedule modal word joining the cadence to the weekday it runs on, as in "Weekly on Monday"',
                          })}
                        </Typography.Text>
                        <SimpleSelect
                          id={`${COMPONENT_ID}.modal.weekday`}
                          componentId={`${COMPONENT_ID}.modal.weekday`}
                          label={weekdayFieldLabel}
                          width={WEEKDAY_WIDTH}
                          value={String(weekday)}
                          // Guarded rather than coerced: `Number('')` is `NaN`, which serializes to
                          // `null` and would come back from the server as Monday -- a silently
                          // different day from the one on screen.
                          onChange={({ target }) => {
                            const chosen = Number(target.value);
                            if (Number.isInteger(chosen) && chosen >= 0 && chosen <= 6) {
                              setWeekday(chosen);
                            }
                          }}
                          contentProps={{ style: { zIndex: theme.options.zIndexBase + 40 } }}
                        >
                          {weekdayOptions.map((label, index) => (
                            <SimpleSelectOption key={label} value={String(index)}>
                              {label}
                            </SimpleSelectOption>
                          ))}
                        </SimpleSelect>
                      </>
                    )}
                    <Typography.Text color="secondary">
                      {intl.formatMessage({
                        defaultMessage: 'at',
                        description:
                          'Experiment issues tab > Schedule modal word joining the cadence to the time it runs at, as in "Daily at 09:00"',
                      })}
                    </Typography.Text>
                    <SimpleSelect
                      id={`${COMPONENT_ID}.modal.start-hour`}
                      componentId={`${COMPONENT_ID}.modal.start-hour`}
                      label={hourFieldLabel}
                      width={TIME_PART_WIDTH}
                      value={hour}
                      disabled={filtersDisabled}
                      onChange={({ target }) => setStartTime(joinStartTime(target.value, minute))}
                      contentProps={{ style: { zIndex: theme.options.zIndexBase + 40 } }}
                    >
                      {startTimeHourOptions().map((option) => (
                        <SimpleSelectOption key={option} value={option}>
                          {option}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelect>
                    <Typography.Text color="secondary">:</Typography.Text>
                    <SimpleSelect
                      id={`${COMPONENT_ID}.modal.start-minute`}
                      componentId={`${COMPONENT_ID}.modal.start-minute`}
                      label={minuteFieldLabel}
                      width={TIME_PART_WIDTH}
                      value={minute}
                      disabled={filtersDisabled}
                      onChange={({ target }) => setStartTime(joinStartTime(hour, target.value))}
                      contentProps={{ style: { zIndex: theme.options.zIndexBase + 40 } }}
                    >
                      {minuteOptions.map((option) => (
                        <SimpleSelectOption key={option} value={option}>
                          {option}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelect>
                  </div>
                </div>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text color="secondary">{timezoneRowLabel}</Typography.Text>
                  <NavigableCombobox
                    componentId={`${COMPONENT_ID}.modal.timezone`}
                    config={timezoneConfig}
                    value={timezone}
                    // Clearing the field leaves the schedule on the zone it had rather than on none,
                    // since every stored schedule has to name one for its time to mean anything.
                    onChange={(next) => setTimezone(next ?? timezone)}
                    valueToDisplayString={(zone) => timezoneChoice(zone).label}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Search timezones',
                      description: 'Experiment issues tab > Schedule modal placeholder of the timezone search input',
                    })}
                    disabled={filtersDisabled}
                    clearOnOpen
                  />
                </div>
              </div>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text color="secondary">{modelRowLabel}</Typography.Text>
                {selection && (
                  <div css={{ width: MODEL_PICKER_WIDTH }}>
                    <IssueDetectionModelDropdown
                      endpoints={endpoints}
                      value={selection}
                      onChange={setSelection}
                      disabled={filtersDisabled}
                      // The modal raises its own stacking context, which this control's option list
                      // and its "Create endpoint" modal would otherwise render below.
                      overlayZIndex={theme.options.zIndexBase + 40}
                    />
                  </div>
                )}
                {cadence !== 'off' && (
                  // OSS only: the managed control has no equivalent line. A run here calls the
                  // model the owner picked above on every occurrence, and nothing else in the
                  // modal says that the cost recurs.
                  <Typography.Hint>
                    {intl.formatMessage({
                      defaultMessage: 'Scheduled runs incur model usage costs on every occurrence.',
                      description:
                        'Experiment issues tab > Schedule modal hint that a scheduled detection run spends on the chosen model each time it fires',
                    })}
                  </Typography.Hint>
                )}
              </div>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Typography.Text color="secondary">{filterSectionLabel}</Typography.Text>
                {filters.map((filter, index) => (
                  <TraceFilterRow
                    key={index}
                    index={index}
                    filter={filter}
                    disabled={filtersDisabled}
                    onChange={changeFilter}
                    onRemove={removeFilter}
                  />
                ))}
                <div>
                  <Button
                    componentId={`${COMPONENT_ID}.modal.filter-add`}
                    icon={<PlusIcon />}
                    // Held at the cap the server applies: a clause past it would show here as
                    // active and be dropped by every run.
                    disabled={filtersDisabled || filters.length >= MAX_TRACE_FILTERS}
                    onClick={addFilter}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'Add filter',
                      description:
                        'Experiment issues tab > Schedule modal button that adds another trace filter clause',
                    })}
                  </Button>
                </div>
              </div>
            </>
          )}
          {view === 'apiKey' && (
            <div
              data-testid="schedule-api-key-view"
              css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}
            >
              <Typography.Text bold>
                {intl.formatMessage({
                  defaultMessage: 'One last step to schedule issue detection',
                  description: 'Experiment issues tab > Schedule modal headline of the API key step',
                })}
              </Typography.Text>
              <Typography.Text color="secondary">
                {intl.formatMessage(
                  {
                    defaultMessage:
                      '{provider} needs an API key. Paste it once and MLflow saves it securely in AI Gateway, where every scheduled run reads it.',
                    description: 'Experiment issues tab > Schedule modal explanation of the API key step',
                  },
                  { provider: selectedProviderName },
                )}
              </Typography.Text>
              <Input
                componentId={`${COMPONENT_ID}.modal.api-key`}
                data-testid="schedule-api-key-input"
                type="password"
                value={apiKeyDraft}
                onChange={({ target }) => setApiKeyDraft(target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'API key',
                  description: 'Experiment issues tab > Schedule modal placeholder of the API key input',
                })}
              />
              {createSecretError && <FormUI.Message type="error" message={createSecretError.message} />}
            </div>
          )}
          {filterError && <FormUI.Message type="error" message={filterError} />}
          {saveFailed && (
            <FormUI.Message
              type="error"
              message={intl.formatMessage({
                defaultMessage: "Couldn't save the schedule. Please try again.",
                description: 'Experiment issues tab > Schedule modal error shown when saving the schedule fails',
              })}
            />
          )}
        </div>
      </Modal>
    </>
  );
};
