import React, { useMemo } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { TimeUnit, isTimeUnitValid } from '../utils/timeUtils';

interface TimeUnitSelectorProps {
  value: TimeUnit;
  onChange: (unit: TimeUnit) => void;
  /** Start time in milliseconds - used to calculate valid time units */
  startTimeMs?: number;
  /** End time in milliseconds - used to calculate valid time units */
  endTimeMs?: number;
  /** Called when user clicks the clear button to reset to auto mode */
  onClear?: () => void;
  /** Whether to show the clear button (e.g., when user has manually changed the value) */
  allowClear?: boolean;
}

export const TimeUnitSelector: React.FC<TimeUnitSelectorProps> = ({
  value,
  onChange,
  startTimeMs,
  endTimeMs,
  onClear,
  allowClear = false,
}) => {
  const intl = useIntl();

  const label = intl.formatMessage({
    defaultMessage: 'Time Unit',
    description: 'Label for time unit selector',
  });

  // Only show time units that won't exceed the max data points limit
  const options = useMemo(() => {
    const allOptions = [
      {
        value: TimeUnit.Second,
        label: intl.formatMessage({ defaultMessage: 'Second', description: 'Time unit: second' }),
      },
      {
        value: TimeUnit.Minute,
        label: intl.formatMessage({ defaultMessage: 'Minute', description: 'Time unit: minute' }),
      },
      { value: TimeUnit.Hour, label: intl.formatMessage({ defaultMessage: 'Hour', description: 'Time unit: hour' }) },
      { value: TimeUnit.Day, label: intl.formatMessage({ defaultMessage: 'Day', description: 'Time unit: day' }) },
      {
        value: TimeUnit.Month,
        label: intl.formatMessage({ defaultMessage: 'Month', description: 'Time unit: month' }),
      },
      { value: TimeUnit.Year, label: intl.formatMessage({ defaultMessage: 'Year', description: 'Time unit: year' }) },
    ];
    return allOptions.filter((option) => isTimeUnitValid(startTimeMs, endTimeMs, option.value));
  }, [intl, startTimeMs, endTimeMs]);

  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <DialogCombobox componentId="mlflow.experiment.overview.time-unit-selector" label={label} value={[value]}>
      <DialogComboboxTrigger
        renderDisplayedValue={() => selectedOption?.label}
        allowClear={allowClear}
        onClear={onClear}
        data-testid="time-unit-select-dropdown"
      />
      <DialogComboboxContent>
        <DialogComboboxOptionList>
          {options.map((option) => (
            <DialogComboboxOptionListSelectItem
              key={option.value}
              checked={value === option.value}
              title={option.label}
              value={option.value}
              onChange={() => onChange(option.value)}
            >
              {option.label}
            </DialogComboboxOptionListSelectItem>
          ))}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
