import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
} from '@databricks/design-system';

import type { InputCategorical } from '../types';

/**
 * The categorical widget's value type depends on `input.multi_select`:
 * a single string for single-select, an array of strings for multi-select.
 * Callers should narrow with the `input.multi_select` flag before reading.
 */
export type LabelSchemaCategoricalValue = string | string[] | null | undefined;

export interface LabelSchemaInputCategoricalProps {
  input: InputCategorical;
  value: LabelSchemaCategoricalValue;
  onChange: (value: string | string[]) => void;
  disabled?: boolean;
  componentId: string;
  label?: string;
}

/**
 * Categorical labeling widget. Renders as a `DialogCombobox` — single-
 * select by default, multi-select when `input.multi_select === true`.
 * Options render in their authored order.
 */
export const LabelSchemaInputCategorical = ({
  input,
  value,
  onChange,
  disabled,
  componentId,
  label,
}: LabelSchemaInputCategoricalProps) => {
  const multiSelect = input.multi_select === true;
  // The DialogCombobox `value` prop always wants a string[]; lift the
  // single-select value into a singleton (or empty) array here. The
  // `typeof value === 'string'` guard (rather than a truthiness check)
  // keeps a falsy-but-valid selected value rendering as selected; only
  // null/undefined ("not yet reviewed") collapses to an empty array.
  let comboValue: string[];
  if (multiSelect) {
    comboValue = Array.isArray(value) ? value : [];
  } else if (typeof value === 'string') {
    comboValue = [value];
  } else {
    comboValue = [];
  }

  const handleSingleSelectChange = (option: string) => {
    onChange(option);
  };

  const handleMultiSelectChange = (option: string) => {
    const current = Array.isArray(value) ? value : [];
    const next = current.includes(option) ? current.filter((o) => o !== option) : [...current, option];
    onChange(next);
  };

  return (
    <DialogCombobox componentId={componentId} value={comboValue} label={label ?? ''} multiSelect={multiSelect}>
      <DialogComboboxTrigger
        disabled={disabled}
        withInlineLabel={false}
        placeholder={multiSelect ? 'Select one or more' : 'Select an option'}
        width="100%"
      />
      <DialogComboboxContent matchTriggerWidth>
        <DialogComboboxOptionList>
          {input.options.map((option) =>
            multiSelect ? (
              <DialogComboboxOptionListCheckboxItem
                key={option}
                value={option}
                checked={(Array.isArray(value) ? value : []).includes(option)}
                onChange={handleMultiSelectChange}
              />
            ) : (
              <DialogComboboxOptionListSelectItem
                key={option}
                value={option}
                checked={value === option}
                onChange={handleSingleSelectChange}
              />
            ),
          )}
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
