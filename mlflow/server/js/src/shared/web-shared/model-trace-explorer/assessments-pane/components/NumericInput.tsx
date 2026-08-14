import { Input } from '@databricks/design-system';

import type { AssessmentValueInputFieldProps } from './types';

export const NumericInput = ({
  value,
  valueError,
  setValue,
  setValueError,
  isSubmitting,
}: AssessmentValueInputFieldProps) => {
  return (
    <div>
      <Input
        data-testid="assessment-value-number-input"
        componentId="shared.model-trace-explorer.assessment-value-number-input"
        value={String(value)}
        onKeyDown={(e) => e.stopPropagation()}
        onChange={(e) => {
          setValue(e.target.value ? Number(e.target.value) : '');
          setValueError(null);
        }}
        type="number"
        disabled={isSubmitting}
        allowClear
      />
      {valueError && <div css={{ marginTop: '8px', color: 'red' }}>{valueError}</div>}
    </div>
  );
};
