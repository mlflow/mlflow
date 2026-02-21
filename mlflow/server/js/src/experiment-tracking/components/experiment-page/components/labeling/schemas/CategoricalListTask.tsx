/**
 * Categorical list (multi-select) task component for labeling schemas
 */

import React from 'react';
import {
  Checkbox,
  CloseSmallIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Tooltip,
  Button,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import type { LabelingSchema } from '../../../../types/labeling';

export const CategoricalListTask = ({
  schema,
  value,
  setValue,
  saveValue,
}: {
  schema: Extract<LabelingSchema['schema'], { type: 'CATEGORICAL_LIST' }>;
  value: string[] | undefined;
  setValue: (newValue: string[] | undefined) => void;
  saveValue: (newValue: string[] | undefined) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  // Remove duplicates
  const options = Array.from(new Set(schema.options));

  const handleDialogClose = React.useCallback(() => {
    saveValue(value);
  }, [saveValue, value]);

  const toggleOption = React.useCallback(
    (o: string, doSubmit?: boolean) => {
      const isSelected = value?.includes(o) ?? false;
      const updated = isSelected ? value?.filter((s) => s !== o) : [...(value ?? []), o];
      setValue(updated);
      if (doSubmit) {
        saveValue(updated);
      }
    },
    [saveValue, value, setValue],
  );

  const removeSelected = React.useCallback(
    (o: string) => {
      const updated = value?.filter((s) => s !== o);
      setValue(updated);
      saveValue(updated);
    },
    [saveValue, value, setValue],
  );

  if (options.length === 0) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text>No options available</Typography.Text>
      </div>
    );
  }

  // Use checkboxes if there are 5 or fewer options
  if (options.length <= 5) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {options.map((o) => (
          <Checkbox
            key={o}
            componentId="mlflow.labeling.tasks.categorical-list.checkbox"
            isChecked={value?.includes(o) ?? false}
            onChange={() => toggleOption(o, true)}
          >
            {o}
          </Checkbox>
        ))}
      </div>
    );
  }

  // Use dropdown for more than 5 options
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {/* Display selected items */}
      <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap', alignItems: 'center' }}>
        {value?.map((s) => (
          <div
            key={s}
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
              alignItems: 'center',
              backgroundColor: theme.colors.actionIconBackgroundPress,
              borderRadius: theme.legacyBorders.borderRadiusLg,
            }}
          >
            <span css={{ paddingLeft: theme.spacing.sm }}>{s}</span>
            <Tooltip
              componentId="mlflow.labeling.tasks.categorical-list.remove-tooltip"
              content="Remove"
            >
              <Button
                onClick={() => removeSelected(s)}
                css={{ padding: `0 ${theme.spacing.sm}px` }}
                size="small"
                componentId="mlflow.labeling.tasks.categorical-list.remove-button"
                icon={<CloseSmallIcon />}
              />
            </Tooltip>
          </div>
        ))}
      </div>
      <div css={{ width: 'fit-content', flexShrink: 0 }}>
        <DialogCombobox
          label="Select"
          componentId="mlflow.labeling.tasks.categorical-list.combobox"
          multiSelect
          stayOpenOnSelection
          onOpenChange={(open) => !open && handleDialogClose()}
          value={value}
        >
          <DialogComboboxTrigger allowClear={false} showTagAfterValueCount={0} />
          <DialogComboboxContent textOverflowMode="multiline" align="end">
            <DialogComboboxOptionList>
              <DialogComboboxOptionListSearch>
                {options.map((o) => (
                  <DialogComboboxOptionListSelectItem
                    key={o}
                    value={o}
                    onChange={() => toggleOption(o, false)}
                    checked={value?.includes(o) ?? false}
                  />
                ))}
              </DialogComboboxOptionListSearch>
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      </div>
    </div>
  );
};
