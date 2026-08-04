import { Input } from '@databricks/design-system';

import type { InputText } from '../types';

export interface LabelSchemaInputTextProps {
  input: InputText;
  value: string | null | undefined;
  onChange: (value: string) => void;
  disabled?: boolean;
  componentId: string;
  /**
   * Placeholder rendered inside the empty textarea. The review surface
   * passes the schema's instruction here so the prompt lives inside the
   * box the reviewer types into rather than as a separate line above it.
   */
  placeholder?: string;
}

/**
 * Free-form text labeling widget. Renders a multi-line textarea bounded
 * by the schema's optional `max_length` (the HTML `maxLength` attribute
 * is omitted when unset). Supported for both feedback and expectation
 * schemas.
 */
export const LabelSchemaInputText = ({
  input,
  value,
  onChange,
  disabled,
  componentId,
  placeholder,
}: LabelSchemaInputTextProps) => {
  return (
    <Input.TextArea
      componentId={componentId}
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      maxLength={input.max_length}
      placeholder={placeholder}
      rows={3}
    />
  );
};
