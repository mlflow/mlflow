import React, { useMemo } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Input,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { TraceArchivalRetentionUnit } from '../utils/traceArchival';

type TraceArchivalRetentionInputProps = {
  amount: string;
  amountInputId: string;
  /** For a label that isn't a `<label htmlFor>` of the amount input. */
  amountAriaLabelledBy?: string;
  amountAriaDescribedBy?: string;
  componentId: string;
  error?: boolean;
  onAmountBlur?: () => void;
  onAmountChange: (value: string) => void;
  onUnitChange: (value: TraceArchivalRetentionUnit) => void;
  unitSelectorId: string;
  unit: TraceArchivalRetentionUnit;
};

export const TraceArchivalRetentionInput = ({
  amount,
  amountInputId,
  amountAriaLabelledBy,
  amountAriaDescribedBy,
  componentId,
  error = false,
  onAmountBlur,
  onAmountChange,
  onUnitChange,
  unitSelectorId,
  unit,
}: TraceArchivalRetentionInputProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const unitOptions = useMemo(
    () => [
      {
        value: 'd' as const,
        label: intl.formatMessage({
          defaultMessage: 'Days',
          description: 'Trace archival retention unit option for days',
        }),
      },
      {
        value: 'h' as const,
        label: intl.formatMessage({
          defaultMessage: 'Hours',
          description: 'Trace archival retention unit option for hours',
        }),
      },
      {
        value: 'm' as const,
        label: intl.formatMessage({
          defaultMessage: 'Minutes',
          description: 'Trace archival retention unit option for minutes',
        }),
      },
    ],
    [intl],
  );

  const selectedOption = unitOptions.find((option) => option.value === unit) ?? unitOptions[0];
  const controlWidth = theme.spacing.xl * 10;

  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: theme.spacing.sm,
        width: '100%',
        maxWidth: controlWidth,
      }}
    >
      <Input
        componentId={`${componentId}.amount_input`}
        id={amountInputId}
        aria-labelledby={amountAriaLabelledBy}
        aria-describedby={amountAriaDescribedBy}
        aria-invalid={error || undefined}
        css={{ width: '100%' }}
        placeholder={intl.formatMessage({
          defaultMessage: 'Enter the retention',
          description: 'Placeholder for trace archival retention amount input',
        })}
        inputMode="numeric"
        value={amount}
        onChange={(e) => {
          onAmountChange(e.target.value);
        }}
        onBlur={onAmountBlur}
        validationState={error ? 'error' : undefined}
      />
      <DialogCombobox
        componentId={`${componentId}.unit_selector`}
        label={intl.formatMessage({
          defaultMessage: 'Trace archival retention unit',
          description: 'Label for trace archival retention unit selector',
        })}
        value={[selectedOption.value]}
      >
        <DialogComboboxTrigger
          id={unitSelectorId}
          allowClear={false}
          css={{ width: '100%' }}
          renderDisplayedValue={() => selectedOption.label}
          withInlineLabel={false}
        />
        <DialogComboboxContent matchTriggerWidth>
          <DialogComboboxOptionList>
            {unitOptions.map((option) => (
              <DialogComboboxOptionListSelectItem
                key={option.value}
                checked={selectedOption.value === option.value}
                title={option.label}
                value={option.value}
                onChange={() => onUnitChange(option.value as TraceArchivalRetentionUnit)}
              >
                {option.label}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
    </div>
  );
};
