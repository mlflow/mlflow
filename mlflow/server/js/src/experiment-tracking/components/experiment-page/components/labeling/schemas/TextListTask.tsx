/**
 * Text list task component for labeling schemas (multiple text inputs)
 */

import React from 'react';
import { Button, Input, PlusIcon, CloseSmallIcon, useDesignSystemTheme } from '@databricks/design-system';

import type { LabelingSchema } from '../../../../types/labeling';

export const TextListTask = ({
  schema,
  value,
  setValue,
  saveValue,
}: {
  schema: Extract<LabelingSchema['schema'], { type: 'TEXT_LIST' }>;
  value: string[] | undefined;
  setValue: (newValue: string[] | undefined) => void;
  saveValue: (newValue: string[] | undefined) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const items = value ?? [];

  const addItem = React.useCallback(() => {
    const updated = [...items, ''];
    setValue(updated);
  }, [items, setValue]);

  const removeItem = React.useCallback(
    (index: number) => {
      const updated = items.filter((_, i) => i !== index);
      setValue(updated);
      saveValue(updated);
    },
    [items, setValue, saveValue],
  );

  const updateItem = React.useCallback(
    (index: number, text: string) => {
      const updated = [...items];
      updated[index] = text;
      setValue(updated);
    },
    [items, setValue],
  );

  const handleSave = React.useCallback(() => {
    saveValue(value);
  }, [value, saveValue]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {items.map((item, index) => (
        <div key={index} css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-start' }}>
          <Input.TextArea
            componentId="mlflow.labeling.tasks.text-list.input"
            value={item}
            onChange={(e) => updateItem(index, e.target.value)}
            onBlur={handleSave}
            maxLength={schema.maxLength}
            rows={2}
            placeholder="Enter text..."
          />
          <Button
            componentId="mlflow.labeling.tasks.text-list.remove"
            type="tertiary"
            icon={<CloseSmallIcon />}
            onClick={() => removeItem(index)}
          />
        </div>
      ))}
      <Button
        componentId="mlflow.labeling.tasks.text-list.add"
        type="tertiary"
        icon={<PlusIcon />}
        onClick={addItem}
      >
        Add item
      </Button>
    </div>
  );
};
