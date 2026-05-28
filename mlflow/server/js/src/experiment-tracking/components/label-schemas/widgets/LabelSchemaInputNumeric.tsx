import { Input } from '@databricks/design-system';

import type { InputNumeric } from '../types';

export interface LabelSchemaInputNumericProps {
  input: InputNumeric;
  value: number | null | undefined;
  onChange: (value: number | null) => void;
  disabled?: boolean;
  componentId: string;
}

/**
 * Numeric labeling widget. Renders as a number input bounded by the
 * schema's `min_value` / `max_value` (each optional independently).
 *
 * For feedback-type schemas the server enforces that both bounds are
 * set; for expectation-type schemas either bound may be missing and
 * the corresponding HTML attribute is simply omitted.
 */
const buildRangePlaceholder = (min: number | undefined, max: number | undefined): string | undefined => {
  if (min != null && max != null) {
    return `${min} – ${max}`;
  }
  if (min != null) {
    return `≥ ${min}`;
  }
  if (max != null) {
    return `≤ ${max}`;
  }
  return undefined;
};

export const LabelSchemaInputNumeric = ({
  input,
  value,
  onChange,
  disabled,
  componentId,
}: LabelSchemaInputNumericProps) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (raw === '') {
      onChange(null);
      return;
    }
    const parsed = Number(raw);
    if (Number.isNaN(parsed)) {
      // Reject NaN at the widget boundary; callers see only valid numbers
      // or null. This avoids polluting the form state with NaN.
      return;
    }
    onChange(parsed);
  };

  return (
    <Input
      componentId={componentId}
      type="number"
      value={value ?? ''}
      onChange={handleChange}
      disabled={disabled}
      min={input.min_value}
      max={input.max_value}
      placeholder={buildRangePlaceholder(input.min_value, input.max_value)}
    />
  );
};
