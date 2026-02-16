/**
 * Categorical (single-select) task component for labeling schemas
 */

import React from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Radio,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import type { LabelingSchema } from '../../../../types/labeling';

export const CategoricalTask = ({
  schema,
  value,
  setValue,
  saveValue,
}: {
  schema: Extract<LabelingSchema['schema'], { type: 'CATEGORICAL' }>;
  value: string | undefined;
  setValue: (newValue: string | undefined) => void;
  saveValue: (newValue: string | undefined) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  // Remove duplicates
  const options = Array.from(new Set(schema.options));

  const selectOption = React.useCallback(
    (o: string | undefined) => {
      setValue(o);
      saveValue(o);
    },
    [saveValue, setValue],
  );

  if (options.length === 0) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text>No options available</Typography.Text>
      </div>
    );
  }

  // Use Radio if there are 5 or fewer options
  if (options.length <= 5) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Radio.Group
          componentId="mlflow.labeling.tasks.categorical.radio"
          defaultValue={value}
          name="categorical-group"
          onChange={(e) => selectOption(e.target.value)}
        >
          {options.map((o) => (
            <Radio key={o} value={o}>
              {o}
            </Radio>
          ))}
        </Radio.Group>
      </div>
    );
  }

  // Use dropdown for more than 5 options
  return (
    <DialogCombobox
      componentId="mlflow.labeling.tasks.categorical.combobox"
      stayOpenOnSelection={false}
      scrollToSelectedElement
      value={value ? [value] : []}
    >
      <DialogComboboxTrigger
        controlled
        allowClear
        onClear={() => selectOption(undefined)}
        withChevronIcon
        withInlineLabel={false}
        placeholder="Select an option"
      />
      <DialogComboboxContent textOverflowMode="multiline" align="end">
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch>
            {options.map((o) => (
              <DialogComboboxOptionListSelectItem
                key={o}
                value={o}
                onChange={() => selectOption(o)}
                checked={value === o}
              />
            ))}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
