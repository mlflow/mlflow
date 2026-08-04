import { Alert, useDesignSystemTheme } from '@databricks/design-system';

import type { LabelSchemaInput } from '../types';
import { LabelSchemaInputCategorical } from './LabelSchemaInputCategorical';
import { LabelSchemaInputNumeric } from './LabelSchemaInputNumeric';
import { LabelSchemaInputPassFail } from './LabelSchemaInputPassFail';
import { LabelSchemaInputText } from './LabelSchemaInputText';

/**
 * Strongly-typed value carrier matching the discriminated `LabelSchemaInput`
 * oneof: callers should narrow on the schema's input variant before
 * reading the value.
 *
 * - `pass_fail`: boolean (true = positive, false = negative)
 * - `categorical` single-select: string
 * - `categorical` multi-select: string[]
 * - `numeric`: number
 * - `text`: string
 *
 * `null` / `undefined` represent the "not yet reviewed" state and are
 * legitimate values across all variants; the renderer's `onChange`
 * accepts them so the cleared-numeric / unselected-categorical signal
 * propagates to the parent rather than being silently coerced.
 */
export type LabelSchemaValue = boolean | string | string[] | number | null | undefined;

export interface LabelSchemaInputRendererProps {
  input: LabelSchemaInput;
  value: LabelSchemaValue;
  onChange: (value: LabelSchemaValue) => void;
  disabled?: boolean;
  /**
   * Stable, PII-free component identifier prefix. The renderer appends
   * the input-variant name so the resulting `componentId` is
   * disambiguated across schemas with different input types.
   */
  componentId: string;
  /** Optional label displayed on the underlying control (categorical only). */
  label?: string;
  /**
   * Schema instruction. For the text variant it is used as the textarea
   * placeholder when the field is editable; hidden when disabled so an
   * empty read-only field stays blank.
   */
  instruction?: string;
}

/**
 * Dispatch a `LabelSchemaInput` to the matching widget. Exactly one oneof
 * variant is set; render a named error if none is, rather than blank-rendering.
 */
export const LabelSchemaInputRenderer = ({
  input,
  value,
  onChange,
  disabled,
  componentId,
  label,
  instruction,
}: LabelSchemaInputRendererProps) => {
  const { theme } = useDesignSystemTheme();

  if (input.pass_fail) {
    return (
      <LabelSchemaInputPassFail
        input={input.pass_fail}
        value={typeof value === 'boolean' ? value : null}
        onChange={onChange}
        disabled={disabled}
        componentId={`${componentId}.pass-fail`}
      />
    );
  }
  if (input.categorical) {
    // multi_select is immutable after creation, so a stored value always
    // matches the widget's shape; guard by type only (a value left over from
    // a different variant resets to null, as the other widgets do).
    const isMultiSelect = input.categorical.multi_select === true;
    const categoricalValue = isMultiSelect
      ? Array.isArray(value)
        ? value
        : null
      : typeof value === 'string'
        ? value
        : null;
    return (
      <LabelSchemaInputCategorical
        input={input.categorical}
        value={categoricalValue}
        onChange={onChange}
        disabled={disabled}
        componentId={`${componentId}.categorical`}
        label={label}
      />
    );
  }
  if (input.numeric) {
    return (
      <LabelSchemaInputNumeric
        input={input.numeric}
        value={typeof value === 'number' ? value : null}
        onChange={onChange}
        disabled={disabled}
        componentId={`${componentId}.numeric`}
      />
    );
  }
  if (input.text) {
    return (
      <LabelSchemaInputText
        input={input.text}
        value={typeof value === 'string' ? value : null}
        onChange={onChange}
        disabled={disabled}
        componentId={`${componentId}.text`}
        placeholder={disabled ? undefined : instruction}
      />
    );
  }
  return (
    <Alert
      componentId={`${componentId}.invalid-input`}
      type="error"
      message="Invalid label schema: input has no variant set (expected one of pass_fail, categorical, numeric, text)."
      closable={false}
      css={{ color: theme.colors.textValidationDanger }}
    />
  );
};
