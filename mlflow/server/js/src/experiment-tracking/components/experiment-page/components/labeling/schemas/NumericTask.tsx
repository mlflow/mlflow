/**
 * Numeric input task component for labeling schemas
 */

import React from 'react';
import { Input, useDesignSystemTheme } from '@databricks/design-system';

import type { LabelingSchema } from '../../../../types/labeling';

export const NumericTask = ({
  schema,
  value,
  setValue,
  saveValue,
}: {
  schema: Extract<LabelingSchema['schema'], { type: 'NUMERIC' }>;
  value: number | undefined;
  setValue: (newValue: number | undefined) => void;
  saveValue: (newValue: number | undefined) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const handleInputChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const { value: newValue } = e.target;
      if (newValue === '') {
        setValue(undefined);
      } else {
        const numValue = Number(newValue);
        if (!isNaN(numValue)) {
          setValue(numValue);
        }
      }
    },
    [setValue],
  );

  const handleSave = React.useCallback(() => {
    let clampedValue = value;
    if (clampedValue != null) {
      if (schema.min != null && clampedValue < schema.min) {
        clampedValue = schema.min;
      }
      if (schema.max != null && clampedValue > schema.max) {
        clampedValue = schema.max;
      }
    }
    setValue(clampedValue);
    saveValue(clampedValue);
  }, [value, setValue, saveValue, schema.min, schema.max]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
      }}
    >
      <Input
        componentId="mlflow.labeling.tasks.numeric"
        type="number"
        value={value ?? ''}
        onChange={handleInputChange}
        onBlur={handleSave}
        onPressEnter={handleSave}
        min={schema.min}
        max={schema.max}
        placeholder="Enter a number..."
      />
    </div>
  );
};
