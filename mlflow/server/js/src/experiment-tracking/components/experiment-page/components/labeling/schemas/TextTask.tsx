/**
 * Text input task component for labeling schemas
 */

import React from 'react';
import { Input } from '@databricks/design-system';

import type { LabelingSchema } from '../../../../types/labeling';

export const TextTask = ({
  schema,
  value,
  setValue,
  saveValue,
}: {
  schema: Extract<LabelingSchema['schema'], { type: 'TEXT' }>;
  value: string | undefined;
  setValue: (newValue: string | undefined) => void;
  saveValue: (newValue: string | undefined) => void;
}) => {
  const handleChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setValue(e.target.value || undefined);
    },
    [setValue],
  );

  const handleSave = React.useCallback(() => {
    saveValue(value);
  }, [value, saveValue]);

  return (
    <Input.TextArea
      componentId="mlflow.labeling.tasks.text"
      value={value ?? ''}
      onChange={handleChange}
      onBlur={handleSave}
      maxLength={schema.maxLength}
      rows={4}
      placeholder="Enter your feedback..."
    />
  );
};
