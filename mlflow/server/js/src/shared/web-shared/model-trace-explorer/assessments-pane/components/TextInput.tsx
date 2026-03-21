import { Input } from '@databricks/design-system';

import type { AssessmentValueInputFieldProps } from './types';

export const TextInput = ({
  value,
  valueError,
  setValue,
  setValueError,
  isSubmitting,
}: AssessmentValueInputFieldProps) => {
  return (
    <div>
      <Input
        data-testid="assessment-value-string-input"
        componentId="shared.model-trace-explorer.assessment-value-string-input"
        value={String(value)}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => {
          setValue(e.target.value);
          setValueError(null);
        }}
        disabled={isSubmitting}
        allowClear
      />
      {valueError && <div css={{ marginTop: '8px', color: 'red' }}>{valueError}</div>}
    </div>
  );
};
