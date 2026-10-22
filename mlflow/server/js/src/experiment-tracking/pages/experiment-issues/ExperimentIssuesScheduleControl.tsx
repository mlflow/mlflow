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
import { useCallback, useState } from 'react';
import { useIntl } from 'react-intl';
import type { ScheduleTraceFilter, ScheduleTraceFilterField } from './issueDetectionSchedule';
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
  const { setting, traceFilters, isLoading, isUpdating, setSetting } = useIssueDetectionSchedule({
    experimentId,
  });

  const [open, setOpen] = useState(false);
  const [cadence, setCadence] = useState<IssueDetectionScheduleSetting>('off');
  const [filters, setFilters] = useState<ScheduleTraceFilter[]>([]);
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
    setApprovedUnattendedRun(false);
    setOpen(true);
  }, [setting, traceFilters]);
  const closeModal = useCallback(() => setOpen(false), []);

  const changeFilter = useCallback(
    (index: number, next: ScheduleTraceFilter) =>
      setFilters((prev) => prev.map((filter, i) => (i === index ? next : filter))),
    [],
  );

  // Removing the last row collapses the section back to just the "Add filter" button.
  const removeFilter = useCallback((index: number) => setFilters((prev) => prev.filter((_, i) => i !== index)), []);

  const addFilter = useCallback(() => setFilters((prev) => [...prev, emptyFilterRow()]), []);

  const handleSave = useCallback(async () => {
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
    setSaveFailed(false);
    try {
      // Await the writes and only close on success. On failure keep the modal open with the draft
      // intact so the user can retry without re-entering the cadence/filters.
      await setSetting({ setting: cadence, traceFilters: cleaned });
      setOpen(false);
    } catch {
      setSaveFailed(true);
    }
  }, [cadence, filters, setSetting]);

  // A schedule only runs on a server with the job system and its periodic tasks enabled, behind a
  // SQL backend store. Rather than hide the control, keep it visible but disabled with a tooltip
  // explaining why -- the affordance stays discoverable and any cadence already saved is still
  // readable. Offering a working-looking switch that silently never fires would be worse.
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
  const filterSectionLabel = intl.formatMessage({
    defaultMessage: 'Additional trace filters (optional)',
    description: 'Experiment issues tab > Schedule modal label for the optional trace filters',
  });
  const filtersDisabled = cadence === 'off';
  const needsApproval = cadence !== 'off' && !approvedUnattendedRun;

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
        onCancel={closeModal}
        footer={
          <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            {/* Upstream's "View run history" link sits here. It opens the managed scheduler's own
                run history, which OSS has no equivalent of yet, so the footer holds only the
                buttons for now. */}
            <div css={{ marginLeft: 'auto' }}>
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
                disabled={needsApproval}
                onClick={handleSave}
              >
                {intl.formatMessage({
                  defaultMessage: 'Save',
                  description: 'Experiment issues tab > Schedule modal confirm button',
                })}
              </Button>
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
                  description: 'Experiment issues tab > Schedule modal button that adds another trace filter clause',
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
                    "Each run samples this experiment's recent traces, sends their content to the detection model using the credentials configured on the MLflow server, and files what it finds in Issues. That spend recurs on this cadence. Trace content is written by your application and its users, so it is untrusted input to the model.",
                  description:
                    'Experiment issues tab > Schedule modal notice explaining that a scheduled detection run analyzes untrusted trace content unattended using the server credentials',
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
