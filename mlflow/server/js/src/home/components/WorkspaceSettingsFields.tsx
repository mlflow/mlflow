import { FormUI, RHFControlledComponents, useDesignSystemTheme } from '@databricks/design-system';
import { type FieldPath, type FieldValues, useFormContext } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { TraceArchivalRetentionInput } from '../../common/components/TraceArchivalRetentionInput';
import type { TraceArchivalRetentionUnit } from '../../common/utils/traceArchival';
import { validateTraceArchivalLocation } from '../../common/utils/traceArchival';

type WorkspaceSettingsFieldNames<TFieldValues extends FieldValues> = {
  description: FieldPath<TFieldValues>;
  artifactRoot: FieldPath<TFieldValues>;
  traceArchivalLocation: FieldPath<TFieldValues>;
};

type TraceArchivalRetentionFieldState = {
  amount: string;
  error?: string;
  onAmountChange: (value: string) => void;
  onUnitChange: (value: TraceArchivalRetentionUnit) => void;
  unit: TraceArchivalRetentionUnit;
};

type WorkspaceSettingsFieldsProps<TFieldValues extends FieldValues> = {
  idPrefix: string;
  componentId: string;
  fieldNames: WorkspaceSettingsFieldNames<TFieldValues>;
  traceArchivalRetention: TraceArchivalRetentionFieldState;
  descriptionAutoFocus?: boolean;
  showClearHint?: boolean;
  showTraceArchivalSettings?: boolean;
};

export const WorkspaceSettingsFields = <TFieldValues extends FieldValues>({
  idPrefix,
  componentId,
  fieldNames,
  traceArchivalRetention,
  descriptionAutoFocus = false,
  showClearHint = false,
  showTraceArchivalSettings = true,
}: WorkspaceSettingsFieldsProps<TFieldValues>) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { control, getFieldState, formState } = useFormContext<TFieldValues>();
  const locationFieldState = getFieldState(fieldNames.traceArchivalLocation, formState);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {showClearHint && (
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Clear any optional field and save to remove the workspace override."
            description="Hint for clearing optional values in edit workspace modal"
          />
        </FormUI.Hint>
      )}
      <div>
        <FormUI.Label htmlFor={`${idPrefix}.description`}>
          <FormattedMessage defaultMessage="Description" description="Label for workspace description field" />
        </FormUI.Label>
        <RHFControlledComponents.Input
          control={control}
          id={`${idPrefix}.description`}
          componentId={`${componentId}.description_input`}
          name={fieldNames.description}
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter workspace description',
            description: 'Placeholder for workspace description input',
          })}
          autoFocus={descriptionAutoFocus}
        />
      </div>
      <div>
        <FormUI.Label htmlFor={`${idPrefix}.artifact_root`}>
          <FormattedMessage
            defaultMessage="Default Artifact Root"
            description="Label for workspace artifact root field"
          />
        </FormUI.Label>
        <RHFControlledComponents.Input
          control={control}
          id={`${idPrefix}.artifact_root`}
          componentId={`${componentId}.artifact_root_input`}
          name={fieldNames.artifactRoot}
          placeholder={intl.formatMessage({
            defaultMessage: 'Enter default artifact root URI',
            description: 'Placeholder for workspace artifact root input',
          })}
        />
      </div>
      {showTraceArchivalSettings && (
        <CollapsibleSection
          componentId={`${componentId}.trace_archival_section`}
          title={intl.formatMessage({
            defaultMessage: 'Trace archival settings',
            description: 'Accordion title for trace archival workspace settings',
          })}
          defaultCollapsed
          forceOpen={Boolean(locationFieldState.error || traceArchivalRetention.error)}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <div>
              <FormUI.Label htmlFor={`${idPrefix}.trace_archival_location`}>
                <FormattedMessage
                  defaultMessage="Trace Archival Location"
                  description="Label for workspace trace archival location field"
                />
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Optional. Override where archived trace payloads are stored for this workspace. Leave blank to use the server default."
                  description="Hint for workspace trace archival location field"
                />
              </FormUI.Hint>
              <RHFControlledComponents.Input
                control={control}
                id={`${idPrefix}.trace_archival_location`}
                componentId={`${componentId}.trace_archival_location_input`}
                name={fieldNames.traceArchivalLocation}
                rules={{
                  validate: (value) => {
                    const result = validateTraceArchivalLocation((value as string | undefined) ?? '', intl);
                    return result.valid || result.error;
                  },
                }}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Enter trace archival location URI',
                  description: 'Placeholder for workspace trace archival location input',
                })}
                validationState={locationFieldState.error ? 'error' : undefined}
              />
              {locationFieldState.error && <FormUI.Message type="error" message={locationFieldState.error.message} />}
            </div>
            <div>
              <FormUI.Label htmlFor={`${idPrefix}.trace_archival_retention_amount`}>
                <FormattedMessage
                  defaultMessage="Trace Archival Retention"
                  description="Label for workspace trace archival retention field"
                />
              </FormUI.Label>
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Optional. Override how long traces stay in the tracking store before archival. Leave blank to use the server default."
                  description="Hint for workspace trace archival retention field"
                />
              </FormUI.Hint>
              <TraceArchivalRetentionInput
                amount={traceArchivalRetention.amount}
                amountInputId={`${idPrefix}.trace_archival_retention_amount`}
                componentId={`${componentId}.trace_archival_retention`}
                error={Boolean(traceArchivalRetention.error)}
                onAmountChange={traceArchivalRetention.onAmountChange}
                onUnitChange={traceArchivalRetention.onUnitChange}
                unitSelectorId={`${idPrefix}.trace_archival_retention_unit`}
                unit={traceArchivalRetention.unit}
              />
              {traceArchivalRetention.error && <FormUI.Message type="error" message={traceArchivalRetention.error} />}
            </div>
          </div>
        </CollapsibleSection>
      )}
    </div>
  );
};
