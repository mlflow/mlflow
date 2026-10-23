import {
  Alert,
  Button,
  CheckCircleIcon,
  Checkbox,
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
import { useCallback, useEffect, useState } from 'react';
import { useIntl } from 'react-intl';
import { generateRandomName } from '../../../common/utils/NameUtils';
import { useApiKeyConfiguration } from '../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { useCreateSecret } from '../../../gateway/hooks/useCreateSecret';
import { useEndpointsQuery } from '../../../gateway/hooks/useEndpointsQuery';
import {
  ISSUE_DETECTION_PROVIDERS,
  IssueDetectionModelDropdown,
  type IssueDetectionModelSelection,
} from '../../components/experiment-page/components/traces-v3/IssueDetectionModelDropdown';
import { unsupportedFilterCharacter } from './issueDetectionSchedule';
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
const SETTING_OPTIONS: IssueDetectionScheduleSetting[] = ['off', 'weekly', 'daily', 'hourly'];

const isScheduleSetting = (value: string): value is IssueDetectionScheduleSetting =>
  SETTING_OPTIONS.some((option) => option === value);

/** The modal's two steps: the schedule itself, and the API key the chosen provider needs. */
type ModalView = 'main' | 'apiKey';

/**
 * Width of the model picker, matching the one in the ad-hoc detection dialog: that modal is 520
 * wide with 32 of padding each side, and its picker takes half of what is left. Picked over
 * filling the row so the two dialogs present the same control at the same size.
 */
const MODEL_PICKER_WIDTH = 220;

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
          width={140}
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
          css={{ width: 160 }}
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
          css={{ width: 56 }}
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
          css={{ width: 160 }}
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

/** The picker's selection in the shape the schedule stores: an endpoint, or a provider's model. */
const toStoredModel = (selection: IssueDetectionModelSelection): IssueDetectionScheduleModel =>
  selection.mode === 'endpoint' && selection.endpointName
    ? { endpointName: selection.endpointName }
    : { provider: selection.provider, model: selection.model };

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
  // Replaces upstream's experiment-ACL check: OSS has no per-experiment write permission for
  // tags, but it does have servers that cannot run a schedule at all.
  const scheduledDetectionAvailable = useScheduledDetectionAvailable();
  const { setting, traceFilters, model, isLoading, isUpdating, loadError, retryLoad, setSetting } =
    useIssueDetectionSchedule({ experimentId });

  const [open, setOpen] = useState(false);
  const [cadence, setCadence] = useState<IssueDetectionScheduleSetting>('off');
  const [filters, setFilters] = useState<ScheduleTraceFilter[]>([]);
  // The model the scheduled runs will use. Null until the stored selection (or the default) is
  // resolved, which needs the endpoint list.
  const [selection, setSelection] = useState<IssueDetectionModelSelection | null>(null);
  // 'apiKey' is the step shown when the chosen provider has no credential to run with. Saving a
  // schedule is not a run, so unlike the ad-hoc dialog -- which learns this from the server
  // rejecting a submission -- this is checked before the schedule is stored.
  const [view, setView] = useState<ModalView>('main');
  const [apiKeyDraft, setApiKeyDraft] = useState('');

  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const { hasExistingSecrets, isLoadingSecrets, envCredentialsAvailable, isLoadingProviderConfig, secretsError } =
    useApiKeyConfiguration({
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
  // The owner approves what an unattended run does on every Save that schedules one, since each
  // Save is what puts a recurring, unwatched run in place.
  const [approvedUnattendedRun, setApprovedUnattendedRun] = useState(false);

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
    setCadence(setting);
    setFilters(traceFilters.map((filter) => ({ ...filter })));
    setSaveFailed(false);
    setFilterError(null);
    setApprovedUnattendedRun(false);
    setSelection(toSelection(model) ?? null);
    setView('main');
    setApiKeyDraft('');
    resetCreateSecret();
    setOpen(true);
  }, [setting, traceFilters, model, resetCreateSecret]);
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

  const addFilter = useCallback(() => setFilters((prev) => [...prev, emptyFilterRow()]), []);

  const persistSchedule = useCallback(async () => {
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
    // Checked before either tag is written: the server drops a clause carrying one of these, so
    // storing it would leave the modal showing a filter that no run applies.
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
        traceFilters: cleaned,
        model: selection ? toStoredModel(selection) : undefined,
      });
      setOpen(false);
    } catch {
      setSaveFailed(true);
    }
  }, [cadence, filters, selection, setSetting, intl]);

  // Prompting for a key before these answer would ask for one the server may already have.
  const isCheckingCredentials = isLoadingSecrets || isLoadingProviderConfig;
  // Only act on a conclusive answer. A server that does not report its environment credentials, or
  // one whose gateway could not be reached, leaves this unknown -- and a schedule that could be
  // saved before this step existed must not become unsavable because a capability check failed.
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
    createSecret(
      {
        secret_name: generateRandomName(selection.provider),
        secret_value: { api_key: trimmedKey },
        provider: selection.provider,
      },
      { onSuccess: () => persistSchedule() },
    );
  }, [apiKeyDraft, selection, createSecret, persistSchedule]);

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
        <Button componentId={`${COMPONENT_ID}.trigger.retry-load`} icon={<ClockIcon />} onClick={retryLoad}>
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
    const disabledReason = intl.formatMessage({
      defaultMessage:
        'This MLflow server cannot run scheduled detection. It needs a SQL backend store with job execution and periodic tasks enabled.',
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
  const modelRowLabel = intl.formatMessage({
    defaultMessage: 'Model',
    description: 'Experiment issues tab > Schedule modal label for the model scheduled runs analyze with',
  });
  const filterSectionLabel = intl.formatMessage({
    defaultMessage: 'Additional trace filters (optional)',
    description: 'Experiment issues tab > Schedule modal label for the optional trace filters',
  });
  const filtersDisabled = cadence === 'off';
  const needsApproval = cadence !== 'off' && !approvedUnattendedRun;
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
            {/* Upstream's "View run history" link sits here. It opens the managed scheduler's own
                run history, which OSS has no equivalent of yet, so the footer holds only the
                buttons for now. */}
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
                    disabled={!apiKeyDraft.trim()}
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
                    disabled={needsApproval || isCheckingCredentials}
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
        zIndex={theme.options.zIndexBase + 20}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {view === 'main' && (
            <>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text color="secondary">{cadenceRowLabel}</Typography.Text>
                <SimpleSelect
                  id={`${COMPONENT_ID}.modal.cadence`}
                  componentId={`${COMPONENT_ID}.modal.cadence`}
                  label={cadenceRowLabel}
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
                    disabled={filtersDisabled}
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
              {cadence !== 'off' && (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                  <Alert
                    componentId={`${COMPONENT_ID}.modal.unattended-run-notice`}
                    type="info"
                    closable={false}
                    message={intl.formatMessage({
                      defaultMessage: 'Scheduled runs analyze traces with nobody watching',
                      description:
                        'Experiment issues tab > Schedule modal notice title about what an unattended scheduled detection run does',
                    })}
                    description={intl.formatMessage({
                      defaultMessage:
                        "Each run samples this experiment's recent traces, sends their content to the model chosen above using the credential saved for it, and files what it finds in Issues. That spend recurs on this cadence. Trace content is written by your application and its users, so it is untrusted input to the model.",
                      description:
                        'Experiment issues tab > Schedule modal notice explaining that a scheduled detection run analyzes untrusted trace content unattended using the credential saved for the chosen model',
                    })}
                  />
                  <Checkbox
                    componentId={`${COMPONENT_ID}.modal.approve-unattended-run`}
                    isChecked={approvedUnattendedRun}
                    onChange={setApprovedUnattendedRun}
                  >
                    {intl.formatMessage({
                      defaultMessage: 'I understand that scheduled runs analyze traces and spend unattended',
                      description:
                        'Experiment issues tab > Schedule modal checkbox the owner must tick before saving a schedule',
                    })}
                  </Checkbox>
                </div>
              )}
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
